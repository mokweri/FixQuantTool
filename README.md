# FixQuantTool

A fixed-point quantization toolkit for Quantization-Aware Training (QAT), bit-exact hardware emulation, and FPGA deployment.

## Overview

FixQuantTool provides an end-to-end workflow for deploying neural networks on fixed-point FPGA accelerators:

1. **Quantization-Aware Training (QAT)** — Fine-tune a pretrained model with learnable quantization thresholds (TQT-based approach) so quantization noise is minimized.
2. **Inference Conversion** — Convert a QAT-trained model to a standard `nn.Module` with folded batch normalization and fixed-point quantized parameters.
3. **Hardware Emulation** — Bit-exact emulation of HLS/FPGA convolution kernels (rounding, saturation, ReLU) for validation before synthesis.
4. **TileCNN Digital Twin** — A PyTorch model that fuses residual additions and uses hardware-accurate GAP/MaxPool kernels, producing bit-identical results to the real FPGA accelerator.
5. **Hardware Test-Case Export** — Extracts per-layer weights, inputs, and bit-exact reference outputs as binary blobs for use by the TileCNN C-simulation testbench.

## Features

- **Fixed-point quantization** with configurable bit-widths and fractional positions
- **TQT (Trained Quantization Thresholds)** for learning optimal quantization ranges
- **Automatic Conv-BN fusion** via `torch.fx` graph transformations
- **Fused ConvBN module** with proper handling of frozen/running BN statistics
- **Multiple rounding modes** (round-nearest, round-to-zero, truncation, convergent)
- **Model introspection** for graph analysis, activation capture, and parameter export
- **HLS sequential emulation** — `HLSConv2d` / `HLSLinear` reproducing the exact two-step rounding of the hardware `EMIT_LOOP`
- **TileCNN digital twin** — Fused `Conv + residual-Add + ReLU` modules and hardware-accurate GAP, bit-identical to the FPGA
- **TileCNN graph exporter** — Produces `graph.json` + binary artifacts satisfying the TileCNN Graph Handoff Specification, with bit-exact references overwritten using fused hardware arithmetic
- **ONNX export** with fixed-point metadata attributes
- **Distributed training** support via Horovod

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU (recommended)
- Conda environment with PyTorch (the project uses the `Obed_Cuda` conda environment)

### Install (development mode)

```bash
conda activate Obed_Cuda
cd FixQuantTool
pip install -e .
```

This makes the `fixquant` package importable from anywhere while allowing in-place edits.

### Dependencies

Core dependencies are listed in `pyproject.toml` and `requirements.txt`:

- `torch >= 2.0`, `torchvision >= 0.15`
- `numpy`, `pyyaml`, `onnx`, `tqdm`, `Pillow`
- Optional: `horovod` (for distributed training)

## Project Structure

```
FixQuantTool/
├── pyproject.toml              # Package metadata (PEP 621)
├── setup.cfg                   # Backward-compat packaging config
├── requirements.txt            # Dependency list
│
├── src/fixquant/               # Main Python package
│   ├── quantization/           # Core quantization modules
│   │   ├── fix_ops.py          # Fixed-point operations & rounding
│   │   ├── fused_conv_bn.py    # Conv-BN fusion module
│   │   ├── qat_modules.py      # QAT-aware layer wrappers
│   │   ├── fxp_modules.py      # Fixed-point emulation modules
│   │   ├── tqt_ops.py          # TQT threshold initialization
│   │   └── tqt_quantizer.py    # Learnable threshold quantizer
│   │
│   ├── graph/                  # FX graph processing
│   │   ├── qat_processor.py    # QatProcessor: model → QAT model
│   │   └── inference_processor.py  # InferProcessor: QAT → std/emu/tilecnn
│   │
│   ├── emulation/              # Hardware emulation
│   │   ├── fxp_emu_modules.py  # HLS & TileCNN bit-exact modules
│   │   │                         #   HLSConv2d, HLSLinear          (sequential)
│   │   │                         #   TileCNNConv2d, TileCNNLinear  (fused twin)
│   │   │                         #   TileCNNGAP, TileCNNMaxPool    (hw-accurate)
│   │   ├── model_introspector.py   # Graph inspection & activation export
│   │   └── model_transforms.py # Model-to-emulation conversion helpers
│   │
│   ├── export/                 # Hardware artifact export
│   │   └── tilecnn_exporter.py # TileCNNGraphExporter + bit-exact ref rewrite
│   │
│   ├── data/                   # Dataset providers
│   │   ├── imagenet.py         # ImageNet data provider
│   │   ├── cifar10.py          # CIFAR-10 data provider
│   │   ├── cifar100.py         # CIFAR-100 data provider
│   │   └── data_utils.py       # Quick data loader utilities
│   │
│   ├── training/               # Training orchestration
│   │   ├── run_config.py       # Training configuration
│   │   ├── run_manager.py      # Training loop manager
│   │   └── distributed_run_manager.py  # Horovod DDP support
│   │
│   ├── models/                 # Model architectures
│   │   ├── resnet.py           # ImageNet-scale ResNet
│   │   └── cifar/              # CIFAR-specific models
│   │
│   └── utils/                  # General utilities
│       ├── common_tools.py     # Metrics, logging helpers
│       └── pytorch_utils.py    # Optimizer, checkpoint utilities
│
├── tools/                      # CLI entry-point scripts
│   ├── qat_train.py            # Run QAT training
│   ├── qat_test.py             # Evaluate a QAT model
│   ├── deploy_eval.py          # Convert QAT → inference & evaluate
│   │                             #   --model_type emu      (HLS emulation)
│   │                             #   --model_type tilecnn  (FPGA digital twin)
│   ├── export_hw_testcases.py  # Export TileCNN binary test artifacts
│   ├── train.py                # Standard FP training
│   ├── train_cifar.py          # CIFAR training
│   ├── ddp_train_hvd.py        # Distributed training (Horovod)
│   ├── hw_layer_test_gen.py    # Generate per-layer test data
│   └── print_model_graph.py    # Dump model graph summary
│
├── configs/                    # Configuration files
│   ├── quant_config.yaml       # Layer replacement mapping
│   └── qconfig_files/          # Saved quantization configs
│
├── docs/                       # Technical documentation
│   ├── conv_fused.md           # Conv-BN fusion details
│   ├── qmodules.md             # Quantized module reference
│   ├── tqt.md                  # TQT quantizer details
│   └── tilecnn_exporter_and_digital_twin.md  # ★ Exporter & Digital Twin
│
├── graph_handoff_spec.md       # TileCNN Graph Handoff Specification
├── scripts/                    # HPC job scripts
├── assets/                     # Test images
├── checkpoints/                # Saved model weights (gitignored)
├── outputs/                    # Runtime outputs (gitignored)
└── qat_models/                 # QAT checkpoints (gitignored)
```

## Quick Start

### 1. QAT Training

```bash
conda activate Obed_Cuda
python tools/qat_train.py \
    --dataset imagenet \
    --dataroot /path/to/imagenet \
    --n_epochs 10 \
    --init_lr 1e-5
```

### 2. Evaluate with HLS Emulation (matches QAT training)

```bash
python tools/deploy_eval.py \
    --dataset imagenet \
    --dataroot /path/to/imagenet \
    --model_type emu
```

### 3. Evaluate with TileCNN Digital Twin (bit-identical to FPGA)

```bash
python tools/deploy_eval.py \
    --dataset imagenet \
    --dataroot /path/to/imagenet \
    --model_type tilecnn
```

The digital twin fuses residual additions into the convolution write-back stage and uses hardware-accurate GAP/MaxPool kernels, reproducing the exact integer arithmetic of the TileCNN FPGA accelerator.

**Expected accuracy (ResNet-50, ImageNet-mini, 40 validation batches):**

| Model | Top-1 | Top-5 |
|---|---|---|
| HLS Emulation (`emu`) | 69.8% | 90.0% |
| TileCNN Digital Twin (`tilecnn`) | 69.5% | 89.8% |

The ~0.3% gap is the real rounding difference from fused vs. sequential residual-add arithmetic across 16 ResNet bottleneck blocks.

### 4. Export Hardware Test Cases

```bash
python tools/export_hw_testcases.py
```

Produces self-contained test artifact directories under `outputs/hw_testcases/`, each containing:

- `inputs/*.int8.bin` — boundary input activations at the correct fractional scale
- `params/*.int8.bin` — INT8 weights and biases
- `refs/*.int8.bin` — bit-exact reference outputs matching TileCNN hardware semantics
- `graph.json` — graph topology with full quantization metadata

## Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `FIXQUANT_DATA_DIR` | Path to dataset directory | `/home/obed/Documents/imagenet-mini` |

### Quantization Config

The quantization configuration is defined in `configs/quant_config.yaml`. It specifies:
- Layer replacement mappings (which standard layers get replaced with QAT equivalents)
- Default quantization bit-widths

## Programmatic Usage

### Full Pipeline (QAT → TileCNN Digital Twin)

```python
import torch
import yaml
import torchvision.models as models
from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor

# Load config
with open("configs/quant_config.yaml") as f:
    config = yaml.safe_load(f)

# Build and quantize model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
qat = QatProcessor(model, config)
qat_model = qat.quantize()
qat.freeze()
qat.load_qat_weights("qat_models/checkpoint/resnet50_best.pth.tar")

# ── Option A: HLS sequential emulation (matches QAT training math) ──────
infer     = InferProcessor(qat_model, config)
emu_model = infer.convert_to_emu_model()      # pure INT8, sequential

# ── Option B: TileCNN digital twin (bit-identical to FPGA hardware) ──────
infer2   = InferProcessor(qat_model, config)
tc_model = infer2.convert_to_tilecnn_model()  # fused residual-add + hw GAP

# Both accept float32 input and return float32 logits
tc_model.eval()
with torch.no_grad():
    logits = tc_model(image_tensor)
```

### Exporting Hardware Test Artifacts

```python
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter

inspector = StdModelInspector(emu_model, default_input_frac=5)
all_nodes = inspector.topological_order()
inspector.register_activation_hooks(all_nodes, capture_input=True, capture_output=True)

with torch.no_grad():
    inspector.run_and_capture(input_image)

exporter = TileCNNGraphExporter(inspector, model_name="resnet50")
exporter.export("outputs/hw_testcases/my_subgraph",
                subgraph_nodes=["conv1", "maxpool", "layer1_0_conv1"])
```

The exporter automatically rewrites the reference files using TileCNN's fused bit-exact arithmetic, so C-simulation results will match without false mismatches.

## Documentation

- [QAT Training Guide](QAT.md)
- [Deployment Guide](DEPLOY.md)
- [Fused Conv-BN](docs/conv_fused.md)
- [Quantized Modules](docs/qmodules.md)
- [TQT Quantizer](docs/tqt.md)
- [TileCNN Exporter & Digital Twin](docs/tilecnn_exporter_and_digital_twin.md) ★ new
- [TileCNN Graph Handoff Specification](graph_handoff_spec.md)

## License

MIT