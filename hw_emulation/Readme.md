# Hardware emulation tools (graph, activations, and test generation)

This folder contains utilities to inspect the converted standard model (FX graph), capture real activations, and generate single-layer hardware test artifacts.
It pairs with the binary outputs and metadata stored in `../hw_data_files`.

What’s here
- hw_layer_test_gen.py — generate a single-layer test:
  - Saves int8 weights, bias, and the input activation for a chosen layer.
  - Writes a JSON with quant params and shapes (see `../hw_data_files/Readme.md`).
  - Uses the standard model produced by `InferProcessor.convert_to_std_model()`.
- model_introspector.py — StdModelInspector class:
  - Builds a graph view with predecessors/successors (handles branches/residual adds).
  - Exposes quant params per layer (frac_w, frac_b, frac_out, frac_in list).
  - Captures input/output activations (records shapes), and saves them as int8.
  - Saves quantized weights/biases (with optional subsetting).
  - Dumps a full graph summary to text/JSON.
- print_model_graph.py — CLI to print/dump the model graph:
  - Prints Conv/Linear layers in topological order to pick names quickly.
  - Optionally runs one forward pass to record shapes and dumps:
    - `../hw_data_files/model_graph.txt`
    - `../hw_data_files/model_graph.json`
  - Paths are resolved relative to the repo; accepts optional overrides.
- FxP_emu_modules.py, hw_fxp_test.py — fixed‑point emulation helpers/tests (optional).
- model_transforms.py — utility transforms for emulation/tooling (optional).
- Param_extractor.py — deprecated; replaced by StdModelInspector. Use `inspector.save_layer_params()` and `inspector.save_activation()` instead.

Quick start
1) Discover layer names and graph details
```bash
python hw_emulation/print_model_graph.py --list_layers_only
# Or dump shapes + full graph summaries
python hw_emulation/print_model_graph.py
```
Outputs: `../hw_data_files/model_graph.txt` and `../hw_data_files/model_graph.json`.

2) Generate a single-layer hardware test
- Open `hw_emulation/hw_layer_test_gen.py` and set constants:
  - LAYER_NAME, SUBSET_SHAPE, output file paths, and TEST_IMAGE_PATH (optional).
- Run:
```bash
python hw_emulation/hw_layer_test_gen.py
```
Outputs go to `../hw_data_files/`:
- weights_*.data, biases_*.data, test_image_*.data (all raw int8)
- <layer>_qparams.json with:
  - frac_w, frac_b, frac_in, frac_out
  - activation_in_dims, weights_dim
  - conv_params (Conv2d only: padding, stride, kernel_size)

Binary format and metadata
- All `.data` are raw int8, contiguous (row‑major, no header). Use the JSON to reconstruct shapes.
- See `../hw_data_files/Readme.md` for exact layouts, examples, and de‑quantization tips.

Defaults and paths
- By default, tools load:
  - QAT checkpoint: `qat_models/checkpoint/resnet50_best.pth.tar`
  - Quant config: `quantization/utils/quant_config.yaml`
- Outputs are written under `../hw_data_files` from this folder.
- If `TEST_IMAGE_PATH` is missing, a random input (1×3×224×224) is used to capture activations.

Tips and gotchas
- Residual/branching: `frac_in` becomes a list (one per predecessor). The generator uses the first by default; adjust if needed.
- Layer not found: ensure `LAYER_NAME` matches the names printed by `print_model_graph.py`.
- Conv weight subsetting: when `SUBSET_SHAPE` is set, bias is trimmed to match the subset out_channels.
- ReLU naming: standard-model conversion assigns unique names per call site, so preds/succs won’t collide.

Deprecation
- `Param_extractor.py` is superseded by `StdModelInspector`. Prefer:
  - `inspector.save_layer_params(name, weight_file, bias_file, target_weight_shape)`
  - `inspector.register_activation_hooks([name]); inspector.save_activation(name, filepath, which="input")`

Troubleshooting
- If checkpoints/configs are missing, update paths or pass overrides to `print_model_graph.py`.
- For custom models, ensure `InferProcessor.convert_to_std_model()` supports all modules used.

License/notes
- These utilities are intended for internal hardware/FPGA test generation and emulation workflows.

