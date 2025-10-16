# Hardware test artifacts (binary + metadata)

This folder contains binary blobs for hardware/FPGA tests and compact metadata to reconstruct and interpret them.
All tensors are saved as raw int8 bytes (no embedded metadata).

Contents (typical)
- test_image_*.data — quantized activation for a chosen layer (int8)
- weights_*.data — quantized Conv/Linear weights (int8)
- biases_*.data — quantized bias (int8)
- <layer>_qparams.json — quantization + shape + layer attributes for that test
- model_graph.txt — human-readable summary of the standard model graph (types, shapes, quant params)
- model_graph.json — machine-readable dump of the graph (nodes + edges and details)

How these files are generated
- Use hw_emulation/hw_layer_test_gen.py to export weights/biases, capture a real activation, and write <layer>_qparams.json.
- Use hw_emulation/print_model_graph.py to print/dump the full standard-model graph with shapes and quant info.

Binary format (all .data files)
- dtype: int8, raw bytes
- layout: contiguous (row-major / C order)
- no header or metadata; use the accompanying JSON to infer shapes and quantization parameters

File specifics
1) test_image_... .data
- Content: input activation to the selected layer, captured from a real forward pass, quantized to int8 using frac_in
- Shape: activation_in_dims from <layer>_qparams.json
  - Usually CHW after removing the single batch dim, e.g., [C, H, W]
- Size in bytes: product(C, H, W)

2) weights_... .data
- Content: quantized weights of the selected layer, saved as int8 using frac_w
- Layout:
  - Conv2d: OIHW (out_channels, in_channels, kernel_h, kernel_w)
  - Linear: (out_features, in_features)
- Shape: weights_dim from <layer>_qparams.json (may be a subset of the original weights)

3) biases_... .data
- Content: quantized bias (int8) using frac_b
- Shape: [out_channels] (Conv2d) or [out_features] (Linear)
  - If weights were subset, bias length matches the subset out_channels

4) <layer>_qparams.json (example keys)
{
  "layer": "layer1_0_conv2",
  "frac_w": 9,
  "frac_b": 5,
  "frac_in": 5,
  "frac_out": 4,
  "activation_in_dims": [64, 64, 64],
  "weights_dim": [64, 64, 3, 3],
  "conv_params": { "padding": [1, 1], "stride": [1, 1], "kernel_size": [3, 3] }
}
- Meaning:
  - frac_w, frac_b: fractional bits for weights/bias
  - frac_in: fractional bits used to quantize the input activation saved to test_image_*.data
  - frac_out: layer output fractional bits
  - activation_in_dims: shape of the saved activation (most often CHW)
  - weights_dim: effective saved weight shape (after any subsetting)
  - conv_params: only for Conv2d layers; stride/padding/kernel_size as integer tuples

Interpreting the data
- Reconstruct from .data with numpy:
  - Activation: shape = activation_in_dims from JSON (e.g., [C, H, W])
  - Weights: shape = weights_dim from JSON (Conv: OIHW; Linear: [out_features, in_features])
  - Bias: length equals weights_dim[0]
- De-quantize to approximate float: real ≈ int_value / (2^frac)
  - Use frac_in for activations, frac_w for weights, frac_b for biases, frac_out for layer outputs

Python snippets
```python
import numpy as np, json

# Load metadata
with open("layer1_0_conv2_qparams.json") as f:
    meta = json.load(f)

# Activation (int8 -> int array -> reshape)
act = np.fromfile("test_image_64x64x64.data", dtype=np.int8)
act = act.reshape(meta["activation_in_dims"])  # CHW
act_f = act.astype(np.float32) / (2 ** meta["frac_in"])  # optional de-quant

# Weights
w = np.fromfile("weights_64x64x3x3.data", dtype=np.int8)
w = w.reshape(meta["weights_dim"])  # Conv: OIHW
w_f = w.astype(np.float32) / (2 ** meta["frac_w"])  # optional de-quant

# Bias
b = np.fromfile("biases_1x64.data", dtype=np.int8)
b_f = b.astype(np.float32) / (2 ** meta["frac_b"])  # optional de-quant
```

Graph dumps
- model_graph.txt: ordered list of modules with predecessors/successors, quant params, and observed input/output shapes.
- model_graph.json: nodes array with detailed per-layer info plus edges [{from,to}].
  - Use hw_emulation/print_model_graph.py to regenerate.

Notes
- All values are int8; ensure your downstream pipeline interprets shapes and frac_* correctly.
- If you subset weights (e.g., to smaller O/I/kH/kW), weights_dim reflects the subset, and biases are trimmed to match O.
- Additional reference outputs (e.g., classification logits) may appear as .data files; interpret with the corresponding frac_out and shape for that head.
