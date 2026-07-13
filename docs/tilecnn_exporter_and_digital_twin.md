# TileCNN Graph Exporter & Digital Twin

*Updated 2026-07: the `convert_to_emu_model()` / `convert_to_tilecnn_model()`
APIs and `TileCNNConv2d`/`HLSConv2d` classes described in earlier versions were
consolidated in commit `a30fc02` into a single
`InferProcessor.convert_to_hardware_model(backend=...)` and the `Hardware*`
module family. This revision matches the current code; bit-exactness is now
enforced by `tests/test_kernels.py` and `tests/test_golden.py`.*

This document explains the two hardware-verification systems in FixQuantTool:

1. **`TileCNNGraphExporter`** — exports a quantized model as a hardware-ready binary artifact bundle (weights, inputs, and bit-exact reference outputs) for use in TileCNN C-simulation and host-side verification.
2. **`convert_to_hardware_model()`** — builds a PyTorch model from bit-exact integer `Hardware*` modules (the **digital twin**), enabling full-dataset accuracy evaluation without synthesizing the bitstream.

---

## Background: Two Levels of Fusion

The hardware model built in PyTorch is **sequential**: convs, residual adds
(`HardwareElementwiseAdd`), ReLU/ReLU6 and pools are separate int8 nodes. The
real TileCNN hardware *fuses* the residual add (and post-add activation) into
the convolution write-back stage. Because integer arithmetic is not
associative, the sequential and fused orderings can differ by ±1 LSB per
residual block.

That fusion is applied at **export level**: the exporter folds Add/ReLU nodes
into the preceding conv's `post_ops` in `graph.json`, and
`_write_tilecnn_bitexact_references()` recomputes all reference outputs with
the fused arithmetic — so the shipped references are exactly what the FPGA
computes, even where the sequential PyTorch model differs by an LSB.
The `backend` argument (`"tilecnn"` default, `"hls"`) is metadata the exporter
uses to classify modules; the integer arithmetic is identical.

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
hw_model   = infer_proc.convert_to_hardware_model()

inspector = StdModelInspector(hw_model, default_input_frac=infer_proc.input_frac or 5)
inspector.register_activation_hooks(inspector.topological_order(),
                                    capture_input=True, capture_output=True)
with torch.no_grad():
    inspector.run_and_capture(input_image)

exporter = TileCNNGraphExporter(inspector, model_name="resnet50")
exporter.export("outputs/hw_testcases/my_subgraph",
                subgraph_nodes=["conv1", "maxpool", "layer1_0_conv1"])
```

### What the exporter does

1. **Graph translation** — Walks the model in topological order and lowers each operation into a `graph.json` node (`conv2d`, `linear`, `maxpool2d`, `gap2d`), recording strides, padding, dilation, groups, and tensor IDs. Standalone Add/ReLU/ReLU6 nodes are folded into the preceding conv's `post_ops` (`relu`, `relu6`, `residual_add`, `post_add_relu`, `post_add_relu6`) — ReLU6 is never silently degraded to `relu`.
2. **Parameter extraction** — Writes the already-INT8-quantized `w_int8` and `b_int8` buffers directly to `.int8.bin` files, along with their fractional metadata. No double-quantization occurs.
3. **Boundary input extraction** — Captures live activations via `torch.fx` hooks during a single forward pass and saves them at the correct fractional scale.
4. **Legality checks** — `_check_shift_legality()` verifies every derived `shift_out`, `bias_shift`, residual shift and GAP shift is in range before anything is written; missing quantization params raise instead of falling back to defaults.
5. **Bit-exact reference generation** — After writing `graph.json`, calls `_write_tilecnn_bitexact_references()` which **re-simulates the entire subgraph** using the exported binary files and TileCNN's exact integer arithmetic (fused residual add, grouped/depthwise conv, ReLU6 clamp, fixed-point GAP reciprocal). The resulting reference outputs **exactly match** what the TileCNN C++ kernel produces, eliminating false mismatches in C-simulation. (Grouped conv and the `relu6` post-op are schema **v1.1 extensions** — see `graph_handoff_spec.md`; the HLS kernels must implement them before MobileNet graphs run on the FPGA.)

### Why bit-exact references matter

The PyTorch emulation model runs operations sequentially. TileCNN fuses the residual add into the convolution write-back stage. Because integer arithmetic is not associative, the sequential and fused orderings can differ by ±1 LSB. If you export the sequential PyTorch output as the hardware reference, every residual block will show a mismatch in C-simulation even when the hardware is correct.

The exporter's reference rewrite step avoids this by computing references with the same fused arithmetic the hardware uses.

### Reference arithmetic details

The core bit-exact functions used by both the exporter and the digital twin are:

#### Convolution (`_tilecnn_conv2d`)
```
acc = int64_matmul(weight, unfold(ifm))
shift_out = frac_w + frac_in - frac_out
s1  = (acc >> (shift_out - 1))              # shift to (frac_out+1) scale (truncate)
bias_adj = bias << (frac_out - frac_b + 1)  # align bias to (frac_out+1)
out = (s1 + bias_adj + 1) >> 1              # single round-half-up to frac_out
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

## 2. Digital Twin (`convert_to_hardware_model`)

### Purpose

`InferProcessor.convert_to_hardware_model(backend="tilecnn")` builds a runnable PyTorch model from bit-exact integer modules — the same arithmetic the TileCNN FPGA hardware performs per kernel. Running this through `deploy_eval.py` gives the FPGA-level Top-1 / Top-5 accuracy before any synthesis or board bring-up (up to the ±1 LSB residual-fusion difference described in the Background section).

### Usage

```bash
python tools/deploy_eval.py --model resnet50 --model_type tilecnn   # or emu
```

Or programmatically:

```python
from fixquant.graph.inference_processor import InferProcessor

infer_proc = InferProcessor(qat_model, config)
hw_model   = infer_proc.convert_to_hardware_model(backend="tilecnn")

hw_model.eval()
with torch.no_grad():
    logits = hw_model(image_tensor)  # float32 input, float32 output
```

The model accepts standard float32 images (the `InputQuantizer` handles quantization at the entry point) and returns float32 logits (the `OutputDequantizer` handles de-scaling at the exit).

### How it works

Starting from `convert_to_std_model()`'s plain-module graph, the converter:

1. Extracts per-layer quantization params via `generate_qconfig()`. Missing
   entries **raise** — there are no silent frac defaults. The network input
   frac comes from the learned input QuantStub.
2. Replaces each module in place (`replace_node_module`):

   | Original Module | Replacement | Notes |
   |---|---|---|
   | `nn.Conv2d` | `HardwareConv2d` | int8, two-step EMIT_LOOP rounding; supports `groups` (depthwise) |
   | `nn.Linear` | `HardwareLinear` | Same two-step rounding as conv |
   | `nn.MaxPool2d` | `HardwareMaxPool2d` | `post_pool_shift = frac_out − frac_in` |
   | `nn.AdaptiveAvgPool2d` (global) | `HardwareGAP` | Fixed-point reciprocal GAP |
   | `AddWithMetadata` | `HardwareElementwiseAdd` | Aligns both inputs to `frac_out` with `_signed_shift` |
   | `nn.ReLU` / `nn.ReLU6` | `HardwareRelu` / `HardwareRelu6` | int8; ReLU6 clamps at `min(127, 6·2^frac)` |

3. Injects `InputQuantizer` (`float32 → int8`) after the graph input and
   `OutputDequantizer` (`int8 → float32`) before the output.

Residual adds stay as separate sequential nodes here; the conv-level fusion
exists only in the exported `graph.json` post_ops and its reference generator.

### Accuracy comparison

The previously published numbers (ResNet-50, ImageNet-mini, 40 batches: emu
69.8/90.0, twin 69.5/89.8) predate the 2026-07 rework.

**2026-07-13 rounding-bias fix.** The two-step requant carried *two* `+1`s (one
in `s1`, one in the final shift) where correct round-half-up needs only one. The
extra `+1` added a constant +0.5/layer output bias that compounded through depth
— negligible for shallow ResNet, but ~14 points for MobileNet-V2. The `s1` `+1`
was removed here **and in the HLS kernel** (`output_postproc.cpp` acc_quantize /
emit_tile, `dw_stage.cpp`, and `tilecnn_utils` `runtime_reference.cpp`) so the
twin stays bit-exact with the hardware. After the fix the twin matches its QAT
model: MobileNet-V2 twin 56.0→**69.7** (QAT 69.9), ResNet-50 twin 69.5→**72.5**
(QAT 72.6). Current numbers are tracked in [baselines.md](baselines.md).

---

## Correctness Guarantees

The following invariants are enforced:

1. **Kernel bit-exactness under test** — `tests/test_kernels.py` asserts the `Hardware*` modules and the exporter's `_tilecnn_*` reference kernels agree bit-exactly (all shift signs, groups, relu/relu6, residual adds); `tests/test_golden.py` pins the arithmetic to a committed integer golden file.
2. **No double-quantization** — `save_activation()` in `StdModelInspector` detects if a tensor is already an integer dtype and writes it directly without calling `to_int_tensor()` again.
3. **Correct fractional scales** — `get_quant_params()` in `StdModelInspector` reads `fin`, `fout`, `fw`, `fb` directly from `HardwareConv2d`-family attributes, so boundary inputs are always saved at the correct scale.
4. **Bit-exact references** — `_write_tilecnn_bitexact_references()` runs after every export and overwrites initial references with fused-hardware-accurate values; `_check_shift_legality()` rejects out-of-range shifts first.
5. **Topological validity** — The converted graph passes `torch.fx.Graph.lint()` before being returned.
