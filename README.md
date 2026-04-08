# FixQuantTool

A fixed-point quantization toolkit for Quantization-Aware Training (QAT), bit-exact hardware emulation, and FPGA deployment.

## Overview

FixQuantTool provides an end-to-end workflow for deploying neural networks on fixed-point FPGA accelerators:

1. **Quantization-Aware Training (QAT)** — Fine-tune a pretrained model with learnable quantization thresholds (TQT-based approach) so that quantization noise is minimized.
2. **Inference Conversion** — Convert a QAT-trained model to a standard `nn.Module` with folded batch normalization and fixed-point quantized parameters.
3. **Hardware Emulation** — Bit-exact emulation of HLS/FPGA convolution kernels (including rounding, saturation, and ReLU behavior) for validation before synthesis.
4. **Test Data Generation** — Extract per-layer weights, biases, and activations as binary blobs for use by an FPGA accelerator testbench.

## Features

- **Fixed-point quantization** with configurable bit-widths and fractional positions
- **TQT (Trained Quantization Thresholds)** for learning optimal quantization ranges
- **Automatic Conv-BN fusion** via `torch.fx` graph transformations
- **Fused ConvBN module** with proper handling of frozen/running BN statistics
- **Multiple rounding modes** (round-nearest, round-to-zero, truncation, convergent)
- **Model introspection** for graph analysis, activation capture, and parameter export
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
│   │   └── inference_processor.py  # InferProcessor: QAT → inference
│   │
│   ├── emulation/              # Hardware emulation
│   │   ├── fxp_emu_modules.py  # HLS-accurate Conv2d emulation
│   │   ├── model_introspector.py   # Graph inspection & activation export
│   │   └── model_transforms.py # Model-to-emulation conversion
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
│   ├── train.py                # Standard FP training
│   ├── train_cifar.py          # CIFAR training
│   ├── ddp_train_hvd.py        # Distributed training (Horovod)
│   ├── hw_fxp_test.py          # INT8 convolution emulation test
│   ├── hw_layer_test_gen.py    # Generate per-layer test data
│   └── print_model_graph.py    # Dump model graph summary
│
├── configs/                    # Configuration files
│   ├── quant_config.yaml       # Layer replacement mapping
│   └── qconfig_files/          # Saved quantization configs
│
├── docs/                       # Technical documentation
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

### 2. Deploy & Evaluate

```bash
python tools/deploy_eval.py \
    --dataset imagenet \
    --dataroot /path/to/imagenet
```

### 3. Generate HW Test Data

```bash
python tools/hw_layer_test_gen.py
```

### 4. Run HW Emulation Test

```bash
python tools/hw_fxp_test.py \
    --qparams_json outputs/hw_data_files/conv1/qparams.json \
    --weights_file outputs/hw_data_files/conv1/weights.data \
    --bias_file outputs/hw_data_files/conv1/bias.data \
    --activation_file outputs/hw_data_files/conv1/input.data \
    --emu hls --compare_float_ref
```

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

```python
import torch
import yaml
import torchvision.models as models
from fixquant.graph import QatProcessor, InferProcessor

# Load config
with open("configs/quant_config.yaml") as f:
    config = yaml.safe_load(f)

# Create QAT model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
qat = QatProcessor(model, config)
qat_model = qat.quantize()

# ... train the model ...

# Convert to inference
infer = InferProcessor(qat_model, config)
std_model = infer.convert_to_std_model()
qconfig = infer.generate_qconfig()
```

## Documentation

- [QAT Training Guide](QAT.md)
- [Deployment Guide](DEPLOY.md)
- [Fused Conv-BN](docs/conv_fused.md)
- [Quantized Modules](docs/qmodules.md)
- [TQT Quantizer](docs/tqt.md)

## License

MIT