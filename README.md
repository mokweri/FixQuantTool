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
- **Bit-exact hardware modules** — `HardwareConv2d` / `HardwareLinear` / `HardwareGAP` / `HardwareMaxPool2d` reproducing the exact two-step rounding of the hardware `EMIT_LOOP` (guarded by kernel + golden tests)
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
│   │   ├── equalization.py     # BN fold + cross-layer equalization + bias corr.
│   │   └── tqt_quantizer.py    # Learnable threshold quantizer + calibration
│   │
│   ├── graph/                  # FX graph processing
│   │   ├── qat_processor.py    # QatProcessor: model → QAT model; preflight_check
│   │   └── inference_processor.py  # InferProcessor: QAT → std/hardware model
│   │
│   ├── diagnostics.py          # Per-layer quant reports, threshold logs, parity sweep
│   │
│   ├── emulation/              # Hardware emulation
│   │   ├── fxp_emu_modules.py  # Bit-exact TileCNN modules (HardwareConv2d, ...)
│   │   └── model_introspector.py   # Graph inspection & activation export
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
├── tools/                      # CLI entry-point scripts (all take --model)
│   ├── qat_train.py            # Run QAT training (--cle, --calib_batches, ...)
│   ├── qat_test.py             # Evaluate a QAT checkpoint
│   ├── deploy_eval.py          # Convert QAT → hardware model & evaluate
│   ├── layer_sensitivity.py    # Per-layer quantization sensitivity probes
│   ├── export_hw_testcases.py  # Export TileCNN subgraph test artifacts (resnet50)
│   ├── export_tilecnn_graph.py # Export a full model as a TileCNN graph bundle
│   ├── export_fixA_refactor_testcases.py  # URAM-refactor test cases (resnet50)
│   ├── train.py                # Float (non-QAT) training / baseline eval
│   ├── print_model_graph.py    # Dump model graph summary
│   └── archive/                # Retired scripts (see archive/README.md)
│
├── tests/                      # pytest suite (kernel bit-exactness, QAT flow,
│                                 # export, golden regression) — run: pytest tests/
├── configs/                    # Configuration files
│   ├── quant_config.yaml       # Layer replacement mapping, freeze_bn_delay
│   └── qconfig_files/          # Saved quantization configs
│
├── docs/                       # Technical documentation
│   ├── quantization_repo_analysis_and_roadmap.md  # ★ Analysis & roadmap
│   ├── improvements_2026-07.md # Phase 0-5 rework: what changed and why
│   ├── baselines.md            # Reproducibility baselines & commands
│   ├── conv_fused.md           # Conv-BN fusion details
│   ├── qmodules.md             # Quantized module reference
│   ├── tqt.md                  # TQT quantizer details
│   └── tilecnn_exporter_and_digital_twin.md  # Exporter & Digital Twin
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
    --model resnet50 \
    --dataroot /path/to/imagenet \
    --n_epochs 10 \
    --init_lr 1e-5

# MobileNetV2: fold BN + cross-layer equalization first (recommended)
python tools/qat_train.py --model mobilenet_v2 --cle --n_epochs 10
```

Calibration runs on `--calib_batches` batches with an MSE fix-position search;
TQT thresholds freeze after `--threshold_freeze_frac` of the epochs. Per-epoch
threshold/frac state is logged to `<save_dir>/logs/quant_thresholds.csv`.

### 2. Evaluate with the TileCNN Digital Twin (bit-identical to FPGA)

```bash
python tools/deploy_eval.py \
    --model resnet50 \
    --dataroot /path/to/imagenet \
    --model_type tilecnn
```

### 3. Run the test suite

```bash
python -m pytest tests/            # add -m "not slow" to skip full-MobileNet tests
```

The digital twin fuses residual additions into the convolution write-back stage and uses hardware-accurate GAP/MaxPool kernels, reproducing the exact integer arithmetic of the TileCNN FPGA accelerator.

Accuracy baselines (per model: float / PTQ / QAT / twin) are tracked in
[docs/baselines.md](docs/baselines.md). Numbers recorded before the 2026-07
rework are not comparable (see [docs/improvements_2026-07.md](docs/improvements_2026-07.md)).

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
qat.load_qat_weights("qat_models/resnet50/checkpoint/model_best.pth.tar")
qat.freeze()

# Bit-exact INT8 hardware model (Hardware* modules, TileCNN arithmetic)
infer    = InferProcessor(qat_model, config)
hw_model = infer.convert_to_hardware_model(backend="tilecnn")   # or "hls"

# Accepts float32 input, returns float32 logits (I/O quantizers injected)
hw_model.eval()
with torch.no_grad():
    logits = hw_model(image_tensor)
```

### Exporting Hardware Test Artifacts

```python
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter

inspector = StdModelInspector(hw_model, default_input_frac=infer.input_frac or 5)
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

- [Repo Analysis & Roadmap](docs/quantization_repo_analysis_and_roadmap.md)
- [2026-07 Rework (Phases 0–5)](docs/improvements_2026-07.md)
- [Reproducibility Baselines](docs/baselines.md)
- [Arrhenius (NAISS) GPU Guide](docs/arrhenius_gpu_guide.md) + [job template](scripts/jobscript_arrhenius.sh)
- [QAT Training Guide](QAT.md)
- [Deployment Guide](DEPLOY.md)
- [Fused Conv-BN](docs/conv_fused.md)
- [Quantized Modules](docs/qmodules.md)
- [TQT Quantizer](docs/tqt.md)
- [TileCNN Exporter & Digital Twin](docs/tilecnn_exporter_and_digital_twin.md)
- [TileCNN Graph Handoff Specification](graph_handoff_spec.md)

## License

MIT