# MobileNet Support Roadmap (Fixed-Point Emulation & QAT)

> **Status update (2026-07, see `improvements_2026-07.md`):** Phases 1–2 below
> are implemented, in places differently from the original plan:
>
> - **Depthwise conv** — supported end to end: QAT (`FusedConvBN`/
>   `QuantizedConv2d` pass `groups` through), `HardwareConv2d`, the exporter
>   reference kernels, and `graph.json` (spec v1.1). Tested in
>   `tests/test_qat_flow.py` / `tests/test_export.py`.
> - **ReLU6** — no separate `QReLU6` module was needed: `ActRangePass` bounds
>   the producing conv's activation quantizer to [0, 6], `HardwareRelu6`
>   clamps at `min(127, 6·2^frac)`, and the exporter emits a true `relu6`
>   post-op. Alternatively `--cle` folds BN, replaces ReLU6 with ReLU and
>   equalizes weights (recommended for MobileNetV2).
> - **Functional `adaptive_avg_pool2d`** (torchvision MobileNetV2) — handled
>   by `ReplaceFunctionalPoolPass`.
> - **QAT stability** — the accuracy decay had a training-bug root cause (BN
>   freeze + optimizer detachment), fixed in `fused_conv_bn.py`.
> - **Still open:** Phase 3 (MobileNetV3: `QElementwiseMul`, Hardswish/
>   Hardsigmoid, SE blocks) and hardware-side (HLS) implementation of the
>   spec v1.1 grouped-conv/relu6 extensions.

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

### Phase 1: Foundational Support (MobileNetV2)
*Goal: Enable QAT and fixed-point emulation for MobileNet architectures from Torchvision.*

> [!NOTE]
> `torchvision.models` natively only ships with `mobilenet_v2` and `mobilenet_v3`, lacking a direct `mobilenet_v1` implementation. We will use `mobilenet_v2` as the foundational baseline.

1. **Depthwise Validation**:
   - `QatProcessor` natively handles `groups=in_channels` perfectly via `FusedConvBN`.
2. **ReLU6 and AdaptiveAvgPool Support**:
   - Introduce `HLSRelu6` and map `nn.ReLU6` in `inference_processor.py`.
   - Add support for functional `F.adaptive_avg_pool2d` mapping in `inference_processor.py`.
3. **Emulation Verification**:
   - Validate that the inference processor correctly converts the graph into bit-exact emulation nodes.

### Phase 2: Further Model Optimizations
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
