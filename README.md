# Quantization-Aware Training (QAT) Script
## Overview

This script is designed to perform quantization-aware training (QAT) on deep learning models. 
Here we use Fixed Point Quantization

## Requirements

* Python 3.x
* PyTorch 1.x
* Torchvision
* YAML

## Usage

To run the script, simply execute the Python file. The script accepts various command-line arguments to customize the training process.

### Command-Line Arguments

#### Hyperparameters

* `train_batch_size`: Training batch size (default: 100)
* `test_batch_size`: Testing batch size (default: 100)
* `valid_size`: Validation size (default: None)
* `n_epochs`: Number of training epochs (default: 2)
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
* `dataroot`: Root directory of the dataset (default: "/home/obed/Documents/imagenet-mini")
* `display_freq`: Display training metrics every n steps (default: 100)
* `validation_frequency`: Validate model every n epochs (default: 1)
* `save_dir`: Directory to save trained models (default: './qat_models')
* `output_dir`: Directory to save QAT result (default: 'qat_result')
* `manual_seed`: Manual seed for reproducibility (default: 0)

### Example Usage

```bash
python qat_train.py --dataset imagenet --dataroot /path/to/imagenet --save_dir ./qat_models
```

### QAT Process
The script performs the following QAT process:

1. Loads a pre-trained model (e.g., ResNet18) and a dataset (e.g., ImageNet).
2. Creates a calibration dataset using the `build_sub_train_loader` method.
3. Initializes a `QatProcessor` object with the model and a quantization configuration file (quant_config.yaml).
4. Quantizes the model using the `quantize` method. - replaces standard modules with qat modules
5. Calibrates the quantized model using the calibration dataset and the `calibrate` method.
6. Loads pre-trained QAT weights (if available) using the `load_qat_weights` method.
7. Freezes the quantized model using the freeze method.
8. Creates a `RunConfig` object with the command-line arguments and initializes a `RunManager` object.
9. Validates the quantized model using the `validate` method.

### Notes
* The script assumes that the `quant_config.yaml` file is present in the `quantization/utils` directory.
* The QatProcessor class is responsible for quantizing and calibrating the model.
* The RunManager class is responsible for managing the training process.

### Troubleshooting
* If you encounter any issues while running the script, please check the logs for error messages.
* Make sure that the `quant_config.yaml` file is correctly formatted and present in the required directory.