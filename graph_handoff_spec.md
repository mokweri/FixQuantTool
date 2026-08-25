# TileCNN Quantized Graph Handoff Specification

This document defines the file and metadata contract between FixQuantTool export tool and TileCNN(FPGA Accelerator).

The goal is to let the quantization tool export a complete quantized ResNet-like
graph in a simple, inspectable format. TileCNN then imports that graph, packs
tensors into accelerator layouts, generates per-layer schedules and descriptors,
allocates runtime buffers, and launches the existing `tile_conv` kernel.

This specification is intentionally graph-level. It replaces the current
single-layer `details.txt` fixture style for end-to-end models, while preserving
the same core assumptions: signed int8 tensors, file-backed weights, explicit
fixed-point metadata, and canonical software tensor layouts.

## Ownership Boundary

The quantization/export tool must provide:

- graph topology
- tensor shapes
- tensor file paths
- canonical tensor layouts
- quantization fractional bits
- signed int8 activation, weight, and bias files
- optional file-backed validation inputs and references

TileCNN is responsible for:

- converting activations from `CHW` into the packed accelerator layout
- converting convolution weights from `OIHW` into the bank-friendly URAM layout
- converting linear weights from `OI` into a `1x1` convolution-compatible layout
- widening int8 biases to int32 internally
- applying TileCNN's hardware bias pre-shift internally
- deriving `shift_out`, `shift_bias`, `gap_mul`, and residual alignment shifts
- generating URAM transfer descriptors and compute descriptors
- allocating and reusing device buffers
- lowering graph nodes into one or more `tile_conv` launches

The quantization tool must not pre-pack tensors for TileCNN unless a future
schema version explicitly adds a packed export mode.

## Recommended Directory Layout

Each exported model should be a directory with one graph JSON file and separate
binary tensor files.

```text
resnet18_int8_tilecnn/
  manifest.json
  graph.json

  inputs/
    input_0.int8.bin

  params/
    conv1.weight.int8.bin
    conv1.bias.int8.bin
    layer1.0.conv1.weight.int8.bin
    layer1.0.conv1.bias.int8.bin
    layer1.0.conv2.weight.int8.bin
    layer1.0.conv2.bias.int8.bin
    layer2.0.downsample.0.weight.int8.bin
    layer2.0.downsample.0.bias.int8.bin
    ...
    fc.weight.int8.bin
    fc.bias.int8.bin

  refs/
    logits.int8.bin
    optional/
      conv1_out.int8.bin
      layer1.0_out.int8.bin
      gap_out.int8.bin
```

Use separate files per tensor. This makes debugging, partial regeneration, and
per-layer comparison much easier than a single monolithic binary archive.

## Model Package Manifest

Released-model exports include `manifest.json` with schema
`tilecnn.model-package.v1`. This package-level record identifies the model-zoo
release, FixQuant version and Git revision, checkpoint and configuration
checksums, reference-image preprocessing, and checksums for `graph.json` and
the exported validation inputs and references.

`manifest.json` describes how the package was produced. `graph.json` remains
the compiler input and graph/tensor contract. Consumers may ignore the package
manifest when reading legacy fixtures, but release-oriented workflows should
validate it before accepting a newly generated package.

## Binary File Format

All binary files are raw arrays with no header.

All values are stored in little-endian machine order where that matters. For
int8 files, this is just byte order.

| Tensor kind | Dtype | Layout | File contents |
| --- | --- | --- | --- |
| Activations | signed int8 | `CHW` | `C * H * W` bytes |
| Conv weights | signed int8 | `OIHW` | `Cout * Cin * Kh * Kw` bytes |
| Linear weights | signed int8 | `OI` | `Cout * Cin` bytes |
| Biases | signed int8 | `O` | `Cout` bytes |
| References | signed int8 | `CHW` or `C` | output tensor bytes |

Biases must be exported as signed int8. TileCNN will widen each bias lane to
int32 and apply the hardware bias alignment shift internally before writing the
bias buffer consumed by the kernel.

## Graph JSON Overview

The model handoff file must be named `graph.json` unless the TileCNN importer is
given another path explicitly.

Top-level structure:

```json
{
  "schema": "tilecnn.graph.v1",
  "model": {},
  "target": {},
  "graph": {},
  "tensors": {},
  "nodes": []
}
```

Required top-level fields:

- `schema`: must be `"tilecnn.graph.v1"` for this version.
- `model`: descriptive metadata.
- `target`: global assumptions for the exported graph.
- `graph`: names of graph inputs, outputs, and optional output references.
- `tensors`: dictionary of all graph, parameter, input, and reference tensors.
- `nodes`: topologically ordered operation list.

## Top-Level Metadata

Example:

```json
{
  "schema": "tilecnn.graph.v1",
  "model": {
    "name": "resnet18",
    "producer": "fixed_point_quantizer",
    "producer_version": "0.1.0",
    "source_framework": "pytorch"
  },
  "target": {
    "bitwidth": 8,
    "signed": true,
    "activation_layout": "CHW",
    "conv_weight_layout": "OIHW",
    "linear_weight_layout": "OI",
    "bias_dtype": "int8"
  },
  "graph": {
    "inputs": ["input"],
    "outputs": ["logits"],
    "references": {
      "logits": "logits_ref"
    }
  },
  "tensors": {},
  "nodes": []
}
```

Target constraints for `tilecnn.graph.v1`:

- `bitwidth` must be `8`.
- `signed` must be `true`.
- `activation_layout` must be `"CHW"`.
- `conv_weight_layout` must be `"OIHW"`.
- `linear_weight_layout` must be `"OI"`.
- `bias_dtype` must be `"int8"`.

## Tensor Records

All tensors referenced by nodes must appear in the `tensors` dictionary.

### Activation/Input Tensor

```json
"input": {
  "kind": "input",
  "dtype": "int8",
  "layout": "CHW",
  "shape": [3, 224, 224],
  "frac": 7,
  "file": "inputs/input_0.int8.bin"
}
```

Fields:

- `kind`: `"input"`, `"activation"`, `"param"`, or `"reference"`.
- `dtype`: `"int8"` for this schema version.
- `layout`: `"CHW"` for activations.
- `shape`: `[C, H, W]`.
- `frac`: fixed-point fractional bits for this tensor.
- `file`: relative path, required for file-backed inputs and references.

Intermediate activation tensors do not need a file unless they are exported as
debug references.

```json
"layer1.0.out": {
  "kind": "activation",
  "dtype": "int8",
  "layout": "CHW",
  "shape": [64, 56, 56],
  "frac": 5
}
```

### Convolution Weight Tensor

```json
"layer1.0.conv1_w": {
  "kind": "param",
  "dtype": "int8",
  "layout": "OIHW",
  "shape": [64, 64, 3, 3],
  "frac": 8,
  "file": "params/layer1.0.conv1.weight.int8.bin"
}
```

Shape is `[Cout, Cin, Kh, Kw]`.

### Linear Weight Tensor

```json
"fc_w": {
  "kind": "param",
  "dtype": "int8",
  "layout": "OI",
  "shape": [1000, 512],
  "frac": 8,
  "file": "params/fc.weight.int8.bin"
}
```

Shape is `[Cout, Cin]`. TileCNN lowers this to a `1x1` convolution with input
shape `[Cin, 1, 1]` and output shape `[Cout, 1, 1]`.

### Bias Tensor

```json
"layer1.0.conv1_b": {
  "kind": "param",
  "dtype": "int8",
  "layout": "O",
  "shape": [64],
  "frac": 6,
  "file": "params/layer1.0.conv1.bias.int8.bin"
}
```

Biases are exported as signed int8. TileCNN widens to int32 and computes the
hardware-aligned bias value using:

```text
shift_bias = frac_out - frac_b + 1
```

If `shift_bias >= 0`, TileCNN left-shifts the widened bias. If `shift_bias < 0`,
TileCNN right-shifts it.

### Reference Tensor

```json
"logits_ref": {
  "kind": "reference",
  "dtype": "int8",
  "layout": "CHW",
  "shape": [1000, 1, 1],
  "frac": 2,
  "file": "refs/logits.int8.bin"
}
```

References are optional but strongly recommended. At minimum, provide a final
graph output reference for bring-up. Intermediate references are useful for
debugging graph scheduler and residual ordering issues.

Final output references should be connected through `graph.references`, where
each key is a graph output tensor and each value is a tensor record with
`kind: "reference"`.

## Node Ordering

The `nodes` array must be in topological execution order.

Rules:

- A node may only consume tensors produced by earlier nodes, graph inputs, or
  parameter tensors.
- Residual add nodes must appear after both the main input and residual input
  tensors are available.
- Projection shortcut convolutions must appear before the conv node that uses
  their output as a residual input.
- The first TileCNN graph runtime will execute nodes sequentially in this order.

Parallel branch scheduling may be added later, but this schema does not require
the exporter to encode parallelism.

## Graph Nodes vs TileCNN Launches

The JSON file describes the model as a semantic graph. TileCNN then lowers that
semantic graph into one or more hardware launches.

Do not use the JSON `nodes` array to describe TileCNN's final fused launch plan.
For example, a ResNet stem should be exported as:

```text
conv2d -> maxpool2d
```

not as a single pre-fused `conv2d_maxpool2d` node.

TileCNN's graph compiler is responsible for recognizing legal patterns and
fusing them into a single `tile_conv` launch when the current hardware supports
that fusion.

This distinction gives us three benefits:

- the exported graph remains close to the original neural network semantics
- unfused or future fallback implementations remain possible
- validation can compare either semantic node outputs or fused launch outputs

In this document:

- **graph node** means an operation exported in `graph.json`
- **TileCNN launch** means one invocation of the `tile_conv` hardware kernel
- **fusion** means TileCNN combines multiple graph nodes into one launch

Residual add is the deliberate v1 exception. Instead of exporting a standalone
`add` graph node, the exporter should attach the residual input and residual
post-op flags to the terminal `conv2d` node of a residual block. This matches
the current TileCNN kernel contract, where residual loading and addition are
part of the convolution epilogue.

## Supported Ops in Schema v1

The initial graph importer should support:

- `conv2d`
- `maxpool2d`
- `gap2d`
- `linear`

TileCNN lowering rules:

| Graph pattern | TileCNN lowering |
| --- | --- |
| `conv2d` | one `tile_conv` launch |
| `conv2d` with ReLU | `tile_conv` with `RELU_ON` |
| `conv2d` with residual add | `tile_conv` with `ADD_ENABLE` |
| `conv2d` with residual add and ReLU | `ADD_ENABLE` plus `POST_ADD_RELU` |
| `conv2d` followed by fusible `maxpool2d` | one `tile_conv` launch with `POOL_ENABLE` |
| `conv2d` followed by fusible `gap2d` | one `tile_conv` launch with `GAP_ENABLE` |
| `linear` | one `tile_conv` launch as `1x1` convolution |

For v1, standalone non-fused maxpool and standalone non-fused GAP may be rejected
by the TileCNN importer unless a software or hardware fallback is implemented.

## Conv2D Node

Example:

```json
{
  "id": "layer1.0.conv1",
  "op": "conv2d",
  "inputs": {
    "ifm": "conv1_pool_out",
    "weight": "layer1.0.conv1_w",
    "bias": "layer1.0.conv1_b"
  },
  "outputs": {
    "ofm": "layer1.0.conv1_out"
  },
  "attrs": {
    "kernel": [3, 3],
    "stride": [1, 1],
    "padding": [1, 1, 1, 1],
    "dilation": [1, 1],
    "groups": 1
  },
  "post_ops": {
    "relu": true
  }
}
```

Quantization is primarily read from the input, weight, bias, and output tensor
records:

```text
frac_in  = tensors[inputs.ifm].frac
frac_w   = tensors[inputs.weight].frac
frac_b   = tensors[inputs.bias].frac
frac_out = tensors[outputs.ofm].frac
```

TileCNN derives:

```text
shift_out  = (frac_in + frac_w) - frac_out
shift_bias = frac_out - frac_b + 1
```

Current TileCNN constraints:

- `groups` must be `1` (schema v1 hardware; see the v1.1 extension below).
- `dilation` must be `[1, 1]`.
- `kernel[0] == kernel[1]`.
- `stride[0] == stride[1]`.
- padding should be symmetric for current runtime compatibility.

### Schema v1.1 extensions (proposed — required for MobileNet)

The exporter and the bit-exact software reference already implement these; the
hardware importer/kernels must adopt them before MobileNet-class graphs can run
on the accelerator:

- **Grouped / depthwise convolution**: `attrs.groups` may be any divisor of the
  channel counts; `groups == in_channels` denotes depthwise. The weight tensor
  shape is `[O, I/groups, kH, kW]`. All shift/bias semantics are unchanged.
- **`relu6` post-op**: `post_ops.relu6` (and `post_ops.post_add_relu6` after a
  fused residual add) clamps the int8 output to
  `[0, min(127, round(6 * 2^frac_out))]`. `relu` and `relu6` are mutually
  exclusive. A producer must never lower ReLU6 to plain `relu` unless
  `6 * 2^frac_out >= 127`, in which case they are equivalent.

For `padding`, use `[top, bottom, left, right]`. For symmetric padding, this is
normally `[p, p, p, p]`.

## Residual Add

Residual addition should be represented as a fused post-op on the terminal
convolution of the block.

Example identity block terminal conv:

```json
{
  "id": "layer1.0.conv2_add",
  "op": "conv2d",
  "inputs": {
    "ifm": "layer1.0.conv1_out",
    "weight": "layer1.0.conv2_w",
    "bias": "layer1.0.conv2_b",
    "residual": "conv1_pool_out"
  },
  "outputs": {
    "ofm": "layer1.0.out"
  },
  "attrs": {
    "kernel": [3, 3],
    "stride": [1, 1],
    "padding": [1, 1, 1, 1],
    "dilation": [1, 1],
    "groups": 1
  },
  "post_ops": {
    "residual_add": true,
    "post_add_relu": true
  }
}
```

TileCNN derives residual alignment from tensor fractional bits:

```text
residual_shift = frac_out - frac_residual
```

where:

```text
frac_out      = tensors[outputs.ofm].frac
frac_residual = tensors[inputs.residual].frac
```

If the quantization tool uses a different residual alignment rule, it may
optionally provide an explicit override:

```json
"post_ops": {
  "residual_add": true,
  "residual_shift": 0,
  "post_add_relu": true
}
```

If present, TileCNN should verify that the override is consistent with the
tensor fractional bits or emit a warning.

Residual shape requirements:

- residual tensor shape must match the output tensor shape
- residual tensor dtype must be signed int8
- residual tensor layout must be `CHW`

## Projection Shortcuts

Projection shortcuts should be represented as normal `conv2d` nodes whose output
is later consumed as the residual input.

Example:

```json
{
  "id": "layer2.0.downsample.0",
  "op": "conv2d",
  "inputs": {
    "ifm": "layer1.out",
    "weight": "layer2.0.downsample.0_w",
    "bias": "layer2.0.downsample.0_b"
  },
  "outputs": {
    "ofm": "layer2.0.skip"
  },
  "attrs": {
    "kernel": [1, 1],
    "stride": [2, 2],
    "padding": [0, 0, 0, 0],
    "dilation": [1, 1],
    "groups": 1
  },
  "post_ops": {
    "relu": false
  }
}
```

Then the terminal block convolution consumes `"layer2.0.skip"` as its residual.

## MaxPool2D Node

Example:

```json
{
  "id": "maxpool",
  "op": "maxpool2d",
  "inputs": {
    "ifm": "conv1_out"
  },
  "outputs": {
    "ofm": "conv1_pool_out"
  },
  "attrs": {
    "kernel": [3, 3],
    "stride": [2, 2],
    "padding": [1, 1, 1, 1]
  }
}
```

Current TileCNN maxpool support is specialized for:

- kernel `[3, 3]`
- stride `[2, 2]`
- padding `[1, 1, 1, 1]`

Preferred lowering:

```text
conv2d -> maxpool2d
```

can be fused into the preceding `conv2d` if:

- the maxpool input is the conv output
- no other node consumes the intermediate conv output
- quantization format is unchanged or representable by `post_pool_shift`

If a maxpool cannot be fused, the v1 TileCNN runtime may reject the graph.

## GAP2D Node

Example:

```json
{
  "id": "avgpool",
  "op": "gap2d",
  "inputs": {
    "ifm": "layer4.out"
  },
  "outputs": {
    "ofm": "gap_out"
  }
}
```

Current TileCNN GAP support is intended to be fused with the preceding
convolution when the full convolution output fits in one spatial tile.

TileCNN derives:

```text
gap_mul = round(2^(GAP_SCALE_FRAC_BITS + frac_out_gap - frac_out_conv) / (H * W))
```

where:

```text
frac_out_conv = tensors[inputs.ifm].frac
frac_out_gap  = tensors[outputs.ofm].frac
H * W         = spatial size of the GAP input
```

For schema v1, a non-fused `gap2d` may be rejected unless a TileCNN fallback is
implemented.

## Linear Node

Example:

```json
{
  "id": "fc",
  "op": "linear",
  "inputs": {
    "ifm": "gap_out",
    "weight": "fc_w",
    "bias": "fc_b"
  },
  "outputs": {
    "ofm": "logits"
  }
}
```

Input tensor shape should be `[Cin, 1, 1]`.

Weight tensor shape should be `[Cout, Cin]` with layout `OI`.

Output tensor shape should be `[Cout, 1, 1]`.

TileCNN lowers `linear` into a `1x1` convolution. This path is a compatibility
path for end-to-end ResNet execution, not an optimized classifier backend.

## Full Minimal Example

```json
{
  "schema": "tilecnn.graph.v1",
  "model": {
    "name": "tiny_residual_tail",
    "producer": "fixed_point_quantizer",
    "producer_version": "0.1.0",
    "source_framework": "pytorch"
  },
  "target": {
    "bitwidth": 8,
    "signed": true,
    "activation_layout": "CHW",
    "conv_weight_layout": "OIHW",
    "linear_weight_layout": "OI",
    "bias_dtype": "int8",
    "channel_pack_in": 16,
    "channel_pack_out": 16
  },
  "graph": {
    "inputs": ["input"],
    "outputs": ["block_out"]
  },
  "tensors": {
    "input": {
      "kind": "input",
      "dtype": "int8",
      "layout": "CHW",
      "shape": [64, 56, 56],
      "frac": 5,
      "file": "inputs/input_0.int8.bin"
    },
    "conv1_w": {
      "kind": "param",
      "dtype": "int8",
      "layout": "OIHW",
      "shape": [64, 64, 3, 3],
      "frac": 8,
      "file": "params/conv1.weight.int8.bin"
    },
    "conv1_b": {
      "kind": "param",
      "dtype": "int8",
      "layout": "O",
      "shape": [64],
      "frac": 3,
      "file": "params/conv1.bias.int8.bin"
    },
    "conv1_out": {
      "kind": "activation",
      "dtype": "int8",
      "layout": "CHW",
      "shape": [64, 56, 56],
      "frac": 5
    },
    "conv2_w": {
      "kind": "param",
      "dtype": "int8",
      "layout": "OIHW",
      "shape": [64, 64, 3, 3],
      "frac": 8,
      "file": "params/conv2.weight.int8.bin"
    },
    "conv2_b": {
      "kind": "param",
      "dtype": "int8",
      "layout": "O",
      "shape": [64],
      "frac": 3,
      "file": "params/conv2.bias.int8.bin"
    },
    "block_out": {
      "kind": "activation",
      "dtype": "int8",
      "layout": "CHW",
      "shape": [64, 56, 56],
      "frac": 5
    }
  },
  "nodes": [
    {
      "id": "conv1",
      "op": "conv2d",
      "inputs": {
        "ifm": "input",
        "weight": "conv1_w",
        "bias": "conv1_b"
      },
      "outputs": {
        "ofm": "conv1_out"
      },
      "attrs": {
        "kernel": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "groups": 1
      },
      "post_ops": {
        "relu": true
      }
    },
    {
      "id": "conv2_add",
      "op": "conv2d",
      "inputs": {
        "ifm": "conv1_out",
        "weight": "conv2_w",
        "bias": "conv2_b",
        "residual": "input"
      },
      "outputs": {
        "ofm": "block_out"
      },
      "attrs": {
        "kernel": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "groups": 1
      },
      "post_ops": {
        "residual_add": true,
        "post_add_relu": true
      }
    }
  ]
}
```

## Validation Requirements

The exporter should validate before writing files:

- all node inputs and outputs reference known tensors
- `nodes` are topologically ordered
- every parameter tensor has a file
- every file has exactly the expected byte count
- conv weight shape matches conv attrs and input/output channels
- linear weight shape matches input/output channels
- bias shape matches output channels
- residual input shape matches node output shape
- all `frac` values are present
- all dtypes are supported by this schema
- all current TileCNN constraints are respected, or explicitly marked as
  requiring a future fallback

Expected byte counts:

```text
activation CHW int8: C * H * W
conv weight OIHW int8: Cout * Cin * Kh * Kw
linear weight OI int8: Cout * Cin
bias O int8: Cout
reference CHW int8: C * H * W
```

## Importer Behavior Expected in TileCNN

The TileCNN graph importer should:

1. Parse `graph.json`.
2. Validate schema, tensor records, node order, and file sizes.
3. Load file-backed graph inputs, params, and references.
4. Lower fusible graph patterns into per-layer `tile_conv` launch plans.
5. Pack IFM tensors as needed.
6. Pack weights into bank-friendly URAM layout.
7. Widen and shift int8 biases into the existing `biases_vector_dt` format.
8. Generate `TileSchedule` for each launch.
9. Allocate activation, residual, descriptor, weight, and bias buffers.
10. Run the graph sequentially.
11. Compare final output and optional intermediate outputs against references.

## Versioning Notes

This is schema version `tilecnn.graph.v1`.

Future schema versions may add:

- packed tensor export
- single archive files with byte offsets
- grouped convolution
- non-symmetric padding
- standalone pooling
- standalone GAP
- dedicated classifier/GEMV backend
- per-tensor scale/zero-point quantization metadata
- multiple graph inputs
- dynamic runtime inputs without file-backed activation data

For v1, keep the contract small, explicit, and close to the current TileCNN
engine so that end-to-end ResNet support can be built without destabilizing the
working per-layer accelerator.
