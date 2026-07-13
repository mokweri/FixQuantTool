
---

# QModules

This module provides quantized versions of commonly used neural network layers for FixedPointQAT in PyTorch. 
By integrating fake quantization through the TQT quantizer, it simulates low-precision arithmetic during training.

## Table of Contents

- [Overview](#overview)
- [Setup](#Setup)
- [Module Components](#module-components)
  - [_QuantizedConvNd](#_quantizedconvnd)
  - [QuantizedConv2d](#quantizedconv2d)
  - [QuantizedLinear](#quantizedlinear)
  - [Pooling Layers](#pooling-layers)
    - [QMaxPool2D](#qmaxpool2d)
    - [QAdaptiveAvgPool2d](#qadaptiveavgpool2d)
  - [Element-wise Addition](#qelementwiseadd)
  - [QuantStubC](#quantstubc)
- [Usage Examples](#usage-examples)
- [Exporting Quantization Information](#exporting-quantization-information)
- [State Management](#state-management)


## Overview

This module implements a series of quantized layers that extend PyTorch’s built-in modules. Each layer integrates a TQT-based fake quantizer for weights, biases, and activations, allowing the network to mimic the behavior of fixed-point arithmetic during training. This helps reduce the accuracy gap when deploying the trained model in a low-precision environment.

## Setup

All classes live in `src/fixquant/quantization/qat_modules.py`; the quantizer
they embed is `fixquant.quantization.tqt_quantizer.TQTQuantizer` (see
[tqt.md](tqt.md)). Modules are normally created for you by
`QatProcessor.quantize()` — direct use is only needed for experiments.

## Module Components

### _QuantizedConvNd

This internal class extends PyTorch’s `_ConvNd` to add quantization support.

- **Attributes:**
  - `weight_quantizer`: Quantizes the convolution weights.
  - `bias_quantizer`: Quantizes the biases (if present).
  - `act_quantizer`: Quantizes the activations.
  - `_mod_name`: A user-defined identifier for the module.
  
- **Key Methods:**
  - `extra_repr()`: Returns a string with extra information, including the module name.
  - `state_dict()`: Overridden to save the state of the quantizers along with layer parameters.
  - `_load_from_state_dict()`: Loads quantizer states when restoring a saved model.

### QuantizedConv2d

A quantized version of PyTorch’s `nn.Conv2d` that incorporates fake quantization.

- **Key Methods:**
  - `forward(input: Tensor)`: Applies convolution with quantized weights, biases, and activation.
  - `from_float(mod)`: Converts a standard floating point convolution layer to its quantized counterpart.
  - `export_quant_info()`: Returns quantization parameters (e.g., fractional bits) for weights, bias, and activations.

### QuantizedLinear

A quantized implementation of `nn.Linear` for fully connected layers.

- **Key Methods:**
  - `forward(input)`: Performs linear transformation with quantized weights and activation.
  - `from_float(mod)`: Converts a floating point linear module to a quantized version.
  - `export_quant_info()`: Exports quantization settings for analysis.
  - Overridden `state_dict()` and `_load_from_state_dict()` to include quantizer state.

### Pooling Layers

#### QMaxPool2D

A quantized version of `nn.MaxPool2d` that applies activation quantization after pooling.

- **Key Methods:**
  - `forward(input)`: Executes max pooling followed by quantization.
  - `from_float(mod)`: Converts a standard max pooling layer into a quantized layer.
  - `export_quant_info()`: Exports quantization info for activations.

#### QAdaptiveAvgPool2d

A quantized version of `AdaptiveAvgPool2d`. Also substituted for *functional*
`F.adaptive_avg_pool2d` calls (used by torchvision MobileNetV2) by
`ReplaceFunctionalPoolPass`, so the GAP output always carries a quantizer.

- **Key Methods:**
  - `forward(input)`: Performs adaptive average pooling then applies quantization.
  - `from_float(mod)`: Converts a floating point adaptive average pooling layer.
  - `export_quant_info()`: Returns quantization information.

### QElementwiseAdd

Residual addition followed by activation quantization. Since 2026-07 it models
the hardware alignment: when `align_inputs=True` (default), both inputs are
first rounded onto the *output* frac grid (STE, no clamp — the hardware aligns
in int32 and only saturates after the add), matching the TileCNN residual
`_signed_shift`.

- **Key Methods:**
  - `forward(x1, x2)`: Aligns both inputs to the output grid, sums, then quantizes.
  - `export_quant_info()`: Exports quantization parameters for the output.

### QuantStubC

A module-based quantization stub.

- **Attributes:**
  - `quantizer`: A TQTQuantizer instance for activation quantization.
  - `_mod_name`: Defaulted to `"QuantStubC"`.
  
- **Key Methods:**
  - `forward(x)`: Applies quantization to the input tensor.
  - `export_quant_info()`: Exports the activation quantization details.
  - Overridden `state_dict()` and `_load_from_state_dict()` to include the quantizer state.

## Usage Examples

Below is an example demonstrating how to convert a floating point convolution layer into a quantized convolution layer:

```python
import torch
import torch.nn as nn
from fixquant.quantization.qat_modules import QuantizedConv2d

# Define a standard (floating point) convolution layer
float_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# Convert the float module to a quantized version
quant_conv = QuantizedConv2d.from_float(float_conv)

# Forward pass with a dummy input
input_tensor = torch.randn(1, 3, 32, 32)
output = quant_conv(input_tensor)
print(output.shape)
```

You can similarly convert linear and pooling layers using their respective `from_float` methods.

## Exporting Quantization Information

Each quantized layer provides an `export_quant_info()` method, which returns the quantization parameters (fractional bits) for weights, biases, and activations. This can be used for further analysis or debugging:

```python
frac_w, frac_b, frac_out = quant_conv.export_quant_info()
print(f"Weight fractional bits: {frac_w}, Bias fractional bits: {frac_b}, Activation fractional bits: {frac_out}")
```

## State Management

To ensure proper saving and restoration of both layer parameters and quantizer states, the modules override `state_dict()` and `_load_from_state_dict()`. This allows the full state (including quantization information) to be saved with:

```python
# Save state
torch.save(model.state_dict(), "quant_model.pth")

# Load state
model.load_state_dict(torch.load("quant_model.pth"))
```
