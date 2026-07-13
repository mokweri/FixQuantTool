# Quantization-Aware Training (QAT) Script
## Overview

This script is designed to perform quantization-aware training (QAT) on deep learning models. 
Here we use Fixed Point Quantization

## Usage

To run the script, simply execute the Python file. The script accepts various command-line arguments to customize the training process.

### Command-Line Arguments

#### Model & quantization

* `model`: Model to quantize — `resnet18 | resnet50 | vgg16 | mobilenet_v2` (default: mobilenet_v2)
* `cle`: Fold BN + cross-layer equalization before QAT; replaces ReLU6 with ReLU (recommended for MobileNet)
* `bias_corr`: Empirical bias correction after calibration (requires `--cle`)
* `calib_batches`: Calibration batches, MSE fix-position search (default: 20)
* `calib_scope`: Width of the fix-position search (default: 5)
* `threshold_freeze_frac`: Fraction of epochs after which TQT thresholds freeze (default: 0.7)

#### Hyperparameters

* `train_batch_size`: Training batch size (default: 50)
* `test_batch_size`: Testing batch size (default: 50)
* `valid_size`: Validation size (default: None)
* `n_epochs`: Number of training epochs (default: 10)
* `warmup-epochs`: Number of warmup epochs (default: 0)
* `warmup_lr`: Warmup learning rate (default: -1)
* `init_lr`: Initial learning rate (default: 1e-5)
* `quantizer_lr`: Initial learning rate of quantizer (default: 1e-2)
* `quantizer_lr_decay`: Learning rate decay ratio of quantizer (default: 0.5)

#### Performance Options

* `n_worker`: Number of worker threads (default: 8)
* `pin-memory`: Enable pinning memory (default: True)
* `device`: Device to use for training (default: "cuda")
* `gpus`: GPU IDs to use for training (default: '0')

#### Horovod Settings (Not used for now)

* `fp16-allreduce`: Enable FP16 compression during allreduce (default: False)
* `independent_distributed_sampling`: Enable independent distributed sampling (default: False)
* `dynamic_batch_size`: Dynamic batch size (default: 1)

#### Misc. Options

* `dataset`: Dataset to use for training (default: "imagenet")
* `dataroot`: Root directory of the dataset (default: `$FIXQUANT_DATA_DIR` or `/home/obed/Documents/imagenet-mini`)
* `display_freq`: Display training metrics every n steps (default: 100)
* `validation_frequency`: Validate model every n epochs (default: 1)
* `save_dir`: Directory to save trained models (default: './qat_models')
* `output_dir`: Directory to save QAT result (default: 'qat_result')
* `manual_seed`: Manual seed for reproducibility (default: 0)

### Example Usage

```bash
python tools/qat_train.py --model resnet50 --dataroot /path/to/imagenet
python tools/qat_train.py --model mobilenet_v2 --cle --n_epochs 10
```

Checkpoints are saved under `<save_dir>/<model>/checkpoint/model_best.pth.tar`;
per-epoch threshold/frac logs under `<save_dir>/<model>/logs/quant_thresholds.csv`.

> **`--cle` is part of the model definition, not just a training option.** With
> `--cle` the model is BN-folded and equalized (ReLU6→ReLU) *before* quantization,
> producing a BN-free `QuantizedConv2d` graph. A checkpoint trained this way only
> loads into an equally-equalized model, so every downstream tool that rebuilds
> the model — `qat_test.py`, `deploy_eval.py`, `export_tilecnn_graph.py`,
> `print_model_graph.py` — must be given the same `--cle` flag. Loading a CLE
> checkpoint without `--cle` (or vice-versa) now raises an explanatory error
> instead of a wall of missing/unexpected keys.

### QAT Process
The script performs the following QAT process:

1. Loads a pre-trained model (`--model`) and dataset; seeds are pinned via `--manual_seed`.
2. (Optional, `--cle`) folds BN and applies cross-layer equalization on the float model.
3. Initializes a `QatProcessor` with the model and `configs/quant_config.yaml`; `quantize()` replaces standard modules with QAT modules (Conv-BN fusion, quantized layers, residual adds, ReLU/ReLU6 range bounds, input stub).
4. **Calibrates** over `--calib_batches` batches: every TQT quantizer collects a value reservoir and sets its threshold via an MSE fix-position search.
5. (Optional, `--bias_corr`) applies per-channel bias correction against the float reference.
6. Writes a post-calibration quantizer report (`<save_dir>/<model>_calib_report.csv`).
7. Trains with two Adam parameter groups (weights at `init_lr`, thresholds at `quantizer_lr`, no weight-decay on thresholds); thresholds are frozen after `threshold_freeze_frac` of the epochs; BN folding freezes after `freeze_bn_delay` steps (quant_config.yaml).
8. Validates every `validation_frequency` epochs and checkpoints per epoch.

### Notes
* The script looks for `configs/quant_config.yaml` relative to the project root.
* The `QatProcessor` class (in `fixquant.graph.qat_processor`) is responsible for quantizing and calibrating the model.
* The `RunManager` class (in `fixquant.training`) is responsible for managing the training process.

### Troubleshooting
* If you encounter any issues while running the script, check the logs for error messages.
* Make sure `configs/quant_config.yaml` is correctly formatted and present.
* Set the `FIXQUANT_DATA_DIR` environment variable to override the default dataset path.