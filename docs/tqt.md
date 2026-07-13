# TQT Quantization Module

This module implements fake quantization using Trained Quantization Thresholds (TQT) for quantization-aware training (QAT) in PyTorch. The code simulates quantize–dequantize operations with symmetric fixed-point quantization using power-of-2 scaling factors. It includes a base `FakeQuantizer`, a derived `TQTQuantizer` that learns thresholds, and a custom autograd function `TQTQuantize` for the quantization operation.

## Table of Contents

- [Overview](#overview)
- [Module Components](#module-components)
  - [FakeQuantizer](#fakequantizer)
  - [TQTQuantizer](#tqtquantizer)
  - [TQTQuantize](#tqtquantize)
- [Usage Example](#usage-example)
- [References](#references)
- [License](#license)

## Overview

This module provides the following functionality:
- **Fake Quantization:** Simulates the quantize–dequantize process during training. 
- The general operation is:
x_out = (clamp(round(x / scale + zero_point), quant_min, quant_max) - zero_point) * scale

where symmetric quantization is assumed (zero_point = 0) and scaling is performed using a power-of-2 factor.

- **Trained Quantization Thresholds (TQT):** The quantization thresholds are learned during training to better match the distribution of activations or weights.

## Module Components

### FakeQuantizer

`FakeQuantizer` is an abstract base class derived from `nn.Module` and is designed to simulate quantization during training.

#### Key Points:
- **Buffers:**
- `quant_enabled`: Indicates if quantization is enabled (stored as a uint8 buffer for DDP compatibility).
- `bitwidth`: The number of bits for quantization.
- `domain`: The quantization domain, computed as \(2^{(\text{bitwidth} - 1)}\).

- **Methods:**
- `forward(x)`: Abstract; should be implemented by subclasses.
- `_save_to_state_dict(...)`: Customizes state saving by removing the quantization buffers.
- `_load_from_state_dict(...)`: Customizes state loading to ignore certain keys (`bitwidth`, `quant_enabled`, and `domain`).

### TQTQuantizer

`TQTQuantizer` extends `FakeQuantizer` to implement the TQT method. It is used to quantize weights or activations.

#### Initialization:
- **Parameters:**
- `bitwidth`: Bitwidth for quantization.
- `tensor_type`: Either `'weight'` or `'act'`. An error is raised for any other value.
- `method`: Optional; defaults to 3 for weights and 2 for activations.

- **Attributes:**
- `tensor_type` and `method`: Define the quantization configuration.
- `quantize_fn_cls`: Set to `TQTQuantize` (the custom autograd function).
- `log_threshold`: A learnable parameter that holds the logarithm of the quantization threshold.
- `warmup_enabled`: A buffer used to control the initial threshold calibration.
- `_forward_fn`: A function pointer that determines which quantization strategy to use. Initially set to `_quantize_with_warmup`.

#### Key Methods:
- `_init_threshold(x)`: Initializes the quantization threshold based on input statistics. Uses different schemes:
- For **weights**, it uses a 3-standard-deviation method.
- For **activations**, it uses a KL-divergence based approach.
- `_forward_pass_input(x, log_threshold, domain, method)`: A fallback that passes the input unchanged (used when quantization is disabled).
- `_quantize(x, log_threshold, domain, method)`: Applies the quantization operation using the `TQTQuantize` autograd function.
- `_quantize_with_warmup(x, log_threshold, domain, method)`: One-shot threshold initialization on the first observed tensor, then quantizes.
- `forward(x)`: Calls the current forward function to quantize the input.
- `enable_warmup(enabled=True)` / `disable_warmup()`: Enable or disable the warmup phase.
- `enable_quant(enabled=True)`: Toggle quantization on/off (off = float passthrough). Used by `tools/layer_sensitivity.py` for one-layer-at-a-time probes.
- `freeze_quant(frozen=True)` / `unfreeze_quant()`: Freeze or unfreeze updates to the learned threshold. `RunManager` freezes all thresholds after `threshold_freeze_frac` of the QAT epochs so exported frac bits are stable.
- `export_quant_info()`: Exports the quantization information as a list `[bitwidth, fp]`, where `fp = bitwidth - 1 - ceil(log_threshold)`.
- `_save_to_state_dict(...)` and `_load_from_state_dict(...)`: Manage saving and restoring the quantizer state while handling custom buffers.

#### Multi-batch calibration (added 2026-07)

`QatProcessor.calibrate(loader, device, max_batches, scope)` drives these:

- `start_calibration(max_samples, per_batch)`: switches the forward into
  collection mode — each batch contributes a random subsample of values to a
  bounded reservoir while quantization continues with the current threshold.
- `finish_calibration(scope=5)`: picks the fractional position by the MSE
  search in `fix_ops.find_fix_pos` (the Vitis "diffs" method) over the whole
  reservoir, sets `log_threshold = bitwidth - 1 - frac` (an exact power of 2),
  and returns the frac. Replaces the old behavior where only the first batch
  mattered.

#### Bounded activation ranges (added 2026-07)

`bounded_range` (e.g. `(0.0, 6.0)`) declares the analytic output bounds of the
tensor being quantized. Set by `ActRangePass` for conv outputs whose only
consumer is a ReLU6 (or `(0, None)` for ReLU). Warmup then initializes the
threshold at the bound and calibration samples are clipped to it, so the
quantizer spends its range on values that survive the activation instead of the
raw pre-activation distribution.

### TQTQuantize

`TQTQuantize` is a custom autograd function that performs the quantization operation.

#### Forward Pass:
- **Inputs:**
- `x`: The input tensor.
- `logt`: Logarithm of the quantization threshold.
- `domain`: The quantization domain (e.g., \(2^{(\text{bitwidth} - 1)}\)).
- `method`: The quantization method identifier.

- **Operations:**
- Computes the scale as \( \text{scale} = 2^{\lceil \text{logt} \rceil} / \text{domain} \).
- Defines `quant_max` and `quant_min` for symmetric quantization.
- Applies quantization using the helper function `fix_quantize_tensor` (from `quantization.fix_ops`).

#### Backward Pass:
- Computes gradients for both the input tensor and the log threshold.
- Handles rounding behavior such that edge cases (e.g., rounding -1.5 to -1) are consistent with hardware implementations.
- Adjusts gradients based on whether values fall within or outside the quantization range.

## Usage Example

Below is a simple example demonstrating how to use the TQT quantizer:

```python
import torch
from fixquant.quantization.tqt_quantizer import TQTQuantizer

# Instantiate a TQTQuantizer for weight quantization
tqtq = TQTQuantizer(bitwidth=8, tensor_type='weight')

# Create a sample tensor
float_tensor = torch.tensor([[0.5, -0.75, 1.25],
                            [0.1,  0.3, -0.2]], dtype=torch.float32)

# Print the initial state dictionary
print("Initial state dict:")
print(tqtq.state_dict())

# Perform quantization on the input tensor
qtensor = tqtq.forward(float_tensor)
print("Quantized Tensor:")
print(qtensor)

# Export quantization information (bitwidth and fixed-point parameter)
quant_info = tqtq.export_quant_info()
print("Quantization Info:", quant_info)

# Save the quantizer state to disk
torch.save(tqtq.state_dict(), 'tqt.pth')

# Load the quantizer state from disk
tqtq.load_state_dict(torch.load('tqt.pth'))

```
### References

Trained Quantization Thresholds (TQT) Paper
Quantizing Convolutional Neural Networks for Low-Power High-Throughput Inference Engines