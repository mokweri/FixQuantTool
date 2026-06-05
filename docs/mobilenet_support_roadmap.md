# MobileNet Support Roadmap (Fixed-Point Emulation & QAT)

## 1. Executive Summary

The `FixQuantTool` currently provides robust Quantization-Aware Training (QAT) and bit-exact fixed-point inference emulation for ResNet-style architectures. While the ultimate hardware exporter (`TileCNN`) has its own limitations, enabling the **Fixed-Point Quantization and Emulation** pass for the MobileNet family (MobileNetV1, V2, and V3) requires specific developments in our QAT library, FX graph processing, and bit-exact emulation layers.

This roadmap details the necessary upgrades to the `QatProcessor`, `InferProcessor`, and the underlying fixed-point operators to successfully quantize and emulate MobileNets before any hardware export considerations.

## 2. Gap Analysis (Quantization & Emulation)

An investigation of the current fixed-point quantization codebase reveals the following gaps regarding MobileNet support:

### 2.1. Depthwise Convolution Emulation
- **Current State**: The underlying PyTorch QAT modules (e.g., `QuantizedConv2d`) pass the `groups` argument to `F.conv2d`, meaning naive depthwise execution might run.
- **Missing Development**: Validation and potential refinement of per-channel quantization statistics for depthwise convolutions (`groups == in_channels`). Because depthwise layers don't accumulate across channels, their activation distributions and quantization scale behaviors differ significantly from standard pointwise/spatial convolutions. The `fused_conv_bn` logic and fixed-point `FxP_QConv2D` modules need rigorous testing to ensure bit-exact emulation correctly models depthwise channel isolation.

### 2.2. Advanced Activations (ReLU6, Hardswish, Hardsigmoid)
- **Current State**: The framework relies on standard `nn.ReLU` processing.
- **Missing Development**: 
  - **MobileNetV1/V2**: Require `ReLU6`. The dynamic range of `ReLU6` is bounded at 6.0, meaning the quantization parameters (max values and scales) need a custom `QReLU6` module that forces this fixed-point saturation natively, rather than relying on dynamic activation profiling.
  - **MobileNetV3**: Requires `Hardswish` and `Hardsigmoid`. These activations use complex piecewise functions. Emulating them in bit-exact fixed-point requires custom QAT layers (`QHardswish`, `QHardsigmoid`) that approximate the functions using fixed-point shifts, additions, and multiplications (to mirror how they are eventually implemented in hardware).

### 2.3. Squeeze-and-Excitation (SE) / Element-wise Multiplication (MobileNetV3)
- **Current State**: The quantization modules currently support channel-wise addition via `QElementwiseAdd`, but lack multiplication. `QAdaptiveAvgPool2d` is available.
- **Missing Development**: SE blocks scale spatial channels by an activation vector ($x \cdot scale$). A `QElementwiseMul` module must be developed for the QAT and Emulation passes. This includes implementing the fixed-point scaling math (quantizing the product of two quantized tensors) and updating the FX graph parsers (`QatProcessor` and `InferProcessor`) to trace and replace standard PyTorch multiplications with `QElementwiseMul`.

### 2.4. Model Definitions & Graph Parsing Rules
- **Current State**: The `src/fixquant/models/` directory natively supports ResNet templates, and the graph processors have hardcoded matching logic for standard `Conv-BN-ReLU` blocks and `Add` operations.
- **Missing Development**: 
  - Native `MobileNet` definitions (or robust `torchvision.models.mobilenet` parsers).
  - Expanded graph fusion rules in `inference_processor.py` to identify and fuse `Conv-BN-ReLU6`, `Conv-BN-Hardswish`, and correctly trace the SE block multipliers.

---

## 3. Development Roadmap

### Phase 1: Foundational Support (MobileNetV1)
*Goal: Enable QAT and fixed-point emulation for Depthwise Convolutions.*

1. **Depthwise Validation**:
   - Write comprehensive unit tests for `QuantizedConv2d` and `FxP_QConv2D` with `groups = in_channels`.
   - Ensure the `fused_conv_bn` logic correctly calculates and applies folded batch-norm scales for depthwise filters.
2. **Model Definition**:
   - Introduce `mobilenetv1.py` to `src/fixquant/models/` and verify the `QatProcessor` successfully converts it.
3. **Emulation Verification**:
   - Validate that the inference processor correctly converts the graph into bit-exact emulation nodes.

### Phase 2: MobileNetV2 & ReLU6
*Goal: Support inverted residuals and bounded activations.*

1. **QReLU6 Implementation**:
   - Develop `QReLU6` and `FxP_ReLU6` modules. Ensure the fixed-point saturation logic explicitly caps at the quantized equivalent of 6.0.
2. **Graph Processor Upgrades**:
   - Update `inference_processor.py` and `qat_processor.py` to recognize `nn.ReLU6`, swap it with `QReLU6`, and handle `Conv2d + BatchNorm2d + ReLU6` fusion patterns.
3. **Model Integration**:
   - Introduce `mobilenetv2.py` and run end-to-end QAT and Fixed-Point inference tests.

### Phase 3: MobileNetV3 (SE Blocks & Advanced Activations)
*Goal: Support channel-wise multiplication, Hardswish, and Hardsigmoid.*

1. **Element-wise Multiplication**:
   - Create `QElementwiseMul` and `FxP_ElementwiseMul` modules. Define the bit-exact rules for multiplying two quantized tensors and shifting to the output scale.
2. **New Activations**:
   - Implement `QHardswish` and `QHardsigmoid`, focusing on fixed-point friendly approximations (e.g., using integer adds/shifts instead of floating-point divisions).
3. **Graph Processor Upgrades**:
   - Train the FX graph tracer to identify SE block multiplications and swap them with the new `QElementwiseMul` operator.
4. **End-to-End MobileNetV3**:
   - Port MobileNetV3, run QAT, and successfully evaluate the emulated fixed-point accuracy.
