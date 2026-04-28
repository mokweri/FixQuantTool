# TileCNN Graph Exporter & Digital Twin

This document explains the two hardware-verification systems added to FixQuantTool:

1. **`TileCNNGraphExporter`** — exports a quantized model as a hardware-ready binary artifact bundle (weights, inputs, and bit-exact reference outputs) for use in TileCNN C-simulation and host-side verification.
2. **`convert_to_tilecnn_model()`** — builds a PyTorch model that is a **bit-exact digital twin** of the TileCNN FPGA hardware, enabling full-dataset accuracy evaluation without synthesizing the bitstream.

---

## Background: Why Two Systems?

The FixQuantTool inference pipeline produces two distinct model representations:

| Representation | Method | Behaviour |
|---|---|---|
| **HLS Emulation** | `convert_to_emu_model()` | Runs sequentially: `Conv → Add → ReLU` as separate nodes. Matches QAT training math exactly. |
| **TileCNN Digital Twin** | `convert_to_tilecnn_model()` | Fuses `Conv + residual-Add + ReLU` into a single kernel, mirrors the TileCNN HLS `EMIT_LOOP`. Bit-identical to FPGA hardware. |

The small (~0.3%) accuracy gap between the two is the natural result of differences in integer rounding when the residual add is fused into the convolution write-back stage rather than applied as a separate sequential step.

---

## 1. TileCNN Graph Exporter

### Purpose

`TileCNNGraphExporter` (in `src/fixquant/export/tilecnn_exporter.py`) translates a quantized PyTorch model into the [TileCNN Graph Handoff Specification](../graph_handoff_spec.md) format. It produces a self-contained directory containing:

```
<export_dir>/
├── graph.json          # Graph topology, tensor metadata, fractional scales
├── inputs/             # INT8 boundary input activation files (.int8.bin)
├── params/             # INT8 weight and bias files (.int8.bin)
└── refs/               # Bit-exact reference output files (.int8.bin)
```

### Usage

```bash
# Export all predefined hardware test cases
python tools/export_hw_testcases.py

# Export a single test case
python tools/export_hw_testcases.py --test_case single_conv_relu
```

Or programmatically:

```python
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter

infer_proc = InferProcessor(qat_model, config)
emu_model   = infer_proc.convert_to_emu_model()

inspector = StdModelInspector(emu_model, default_input_frac=5)
inspector.register_activation_hooks(inspector.topological_order(),
                                    capture_input=True, capture_output=True)
with torch.no_grad():
    inspector.run_and_capture(input_image)

exporter = TileCNNGraphExporter(inspector, model_name="resnet50")
exporter.export("outputs/hw_testcases/my_subgraph",
                subgraph_nodes=["conv1", "maxpool", "layer1_0_conv1"])
```

### What the exporter does

1. **Graph translation** — Walks the model in topological order and lowers each operation into a `graph.json` node (`conv2d`, `linear`, `maxpool2d`, `gap2d`), recording strides, padding, dilation, groups, and tensor IDs.
2. **Parameter extraction** — Writes the already-INT8-quantized `w_int8` and `b_int8` buffers directly to `.int8.bin` files, along with their fractional metadata. No double-quantization occurs.
3. **Boundary input extraction** — Captures live activations via `torch.fx` hooks during a single forward pass and saves them at the correct fractional scale.
4. **Bit-exact reference generation** — After writing `graph.json`, calls `_write_tilecnn_bitexact_references()` which **re-simulates the entire subgraph** using the exported binary files and TileCNN's exact integer arithmetic (fused residual add, fixed-point GAP reciprocal). The resulting reference outputs **exactly match** what the TileCNN C++ kernel produces, eliminating false mismatches in C-simulation.

### Why bit-exact references matter

The PyTorch emulation model runs operations sequentially. TileCNN fuses the residual add into the convolution write-back stage. Because integer arithmetic is not associative, the sequential and fused orderings can differ by ±1 LSB. If you export the sequential PyTorch output as the hardware reference, every residual block will show a mismatch in C-simulation even when the hardware is correct.

The exporter's reference rewrite step avoids this by computing references with the same fused arithmetic the hardware uses.

### Reference arithmetic details

The core bit-exact functions used by both the exporter and the digital twin are:

#### Convolution (`_tilecnn_conv2d`)
```
acc = int64_matmul(weight, unfold(ifm))
shift_out = frac_w + frac_in - frac_out
s1  = (acc >> (shift_out - 1)) + 1          # half-step rounding
bias_adj = bias << (frac_out - frac_b + 1)  # align bias to (frac_out+1)
out = (s1 + bias_adj + 1) >> 1              # second half-step
out = clamp(out, -128, 127)
```

#### Global Average Pooling (`_tilecnn_gap`)
Uses a fixed-point reciprocal with `GAP_SCALE_FRAC_BITS = 16` of precision:
```
gap_mul = round((1 << (16 + frac_out - frac_in)) / num_pixels)
out = clamp((sum * gap_mul + (1 << 15)) >> 16, -128, 127)
```

#### Residual Add (`_tilecnn_residual_add`)
```
residual_shift = frac_out - frac_residual
out = clamp(main + signed_shift(residual, residual_shift), -128, 127)
out = relu(out)  # if post_add_relu
```

---

## 2. TileCNN Digital Twin (`convert_to_tilecnn_model`)

### Purpose

`InferProcessor.convert_to_tilecnn_model()` builds a runnable PyTorch model that faithfully reproduces every integer arithmetic step the TileCNN FPGA hardware performs. Running this through `deploy_eval.py` gives the **exact Top-1 / Top-5 accuracy the FPGA will achieve**, before any synthesis or board bring-up.

### Usage

```bash
# Evaluate with the TileCNN digital twin (bit-identical to FPGA accuracy)
python tools/deploy_eval.py --model_type tilecnn

# Evaluate with the sequential HLS emulation (matches QAT training)
python tools/deploy_eval.py --model_type emu
```

Or programmatically:

```python
from fixquant.graph.inference_processor import InferProcessor

infer_proc  = InferProcessor(qat_model, config)
tc_model    = infer_proc.convert_to_tilecnn_model()

tc_model.eval()
with torch.no_grad():
    logits = tc_model(image_tensor)  # full float32 input, float32 output
```

The model accepts standard float32 images (the `InputQuantizer` handles quantization at the entry point) and returns float32 logits (the `OutputDequantizer` handles de-scaling at the exit).

### How it works

The conversion is a multi-pass `torch.fx` graph transformation:

#### Pass 1 — Pre-scan for residual add fusion targets
Identifies every `AddWithMetadata` node whose main-branch argument traces back to a `Conv2d`. Records:
- the conv to promote (`conv_name`)
- the residual tensor arg (`residual_arg`)
- the fractional scales of both inputs and the output
- whether the add is immediately followed by a ReLU (`post_add_relu`)

#### Pass 2 — Module replacement
Replaces all modules in-place using `replace_node_module()`:

| Original Module | Replacement | Notes |
|---|---|---|
| `nn.Conv2d` (non-fused) | `TileCNNConv2d` | Plain conv, no residual |
| `nn.Conv2d` (fused with Add) | `TileCNNConv2d(residual_add=True)` | Fuses add and optional relu |
| `nn.Linear` | `TileCNNLinear` | Same two-step rounding as conv |
| `nn.MaxPool2d` | `TileCNNMaxPool` | With post-pool frac shift |
| `nn.AdaptiveAvgPool2d` | `TileCNNGAP` | Fixed-point reciprocal GAP |
| `nn.ReLU` (intermediate) | `HLSRelu` | Preserves int8 dtype |

#### Pass 3 — Graph rewiring
- Rewires the fused `TileCNNConv2d` node to accept `(main_input, residual_input)` as its two positional arguments.
- Moves out-of-order residual nodes (e.g. `downsample_0`) to appear before their consuming conv node in topological order.
- Bypasses absorbed post-Add ReLU nodes (re-routes their consumers to the fused conv output) and erases dead nodes.

#### Pass 4 — I/O boundary injection
- Inserts `InputQuantizer` after the graph input to convert `float32 → int8`.
- Inserts `OutputDequantizer` before the graph output to convert `int8 → float32`.

### TileCNN-specific modules

All fused modules live in `src/fixquant/emulation/fxp_emu_modules.py`:

| Class | Description |
|---|---|
| `TileCNNConv2d` | Bit-exact conv with optional fused residual add and relu |
| `TileCNNLinear` | Bit-exact fully-connected layer |
| `TileCNNGAP` | Bit-exact global average pool with fixed-point reciprocal |
| `TileCNNMaxPool` | Max-pool with optional post-pool fractional shift |

### Accuracy comparison (ResNet-50, ImageNet-mini, 40 batches)

| Model | Top-1 Acc | Top-5 Acc |
|---|---|---|
| QAT (floating-point training) | ~70% | ~90% |
| HLS Emulation (`emu`) | 69.8% | 90.0% |
| **TileCNN Digital Twin** | **69.5%** | **89.8%** |
| FP32 Baseline | ~76% | ~93% |

The 0.3% gap between `emu` and `tilecnn` reflects the ±1 LSB rounding differences in the fused residual-add path across 16 bottleneck blocks.

---

## Correctness Guarantees

The following invariants are enforced:

1. **No double-quantization** — `save_activation()` in `StdModelInspector` detects if a tensor is already an integer dtype and writes it directly without calling `to_int_tensor()` again.
2. **Correct fractional scales for HLS modules** — `get_quant_params()` in `StdModelInspector` reads `fin`, `fout`, `fw`, `fb` directly from `HLSConv2d` / `TileCNNConv2d` attributes, so boundary inputs are always saved at the correct scale.
3. **Bit-exact references** — `_write_tilecnn_bitexact_references()` runs after every export and overwrites initial references with fused-hardware-accurate values.
4. **Topological validity** — The digital-twin graph passes `torch.fx.Graph.lint()` before being returned.
