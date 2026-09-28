# Using FixQuant with DeepTile

FixQuant is the model-preparation component of the
[DeepTile](https://github.com/mokweri/DeepTile) framework. FixQuant owns training,
quantization, the integer digital twin, and model export. DeepTile owns
hardware-specific legality checks, packing, tiling, scheduling, descriptor
generation, and FPGA execution. Their external boundary is the
[DeepTile Graph Handoff Specification](../graph_handoff_spec.md).

DeepTile was previously named TileCNN. FixQuant's former names still work as
deprecated aliases: `tools/export_tilecnn_graph.py`,
`tools/check_tilecnn_legality.py`, `fixquant.export.tilecnn_exporter`,
`TileCNNGraphExporter`, the `tilecnn` hardware backend and `--model_type
tilecnn`. Model-zoo releases made before the rename keep
`evaluation/tilecnn_metrics.json` and `tilecnn` metric keys, and older policy
files may use `maximum_tilecnn_*` keys; all are read as before. The model ID
`vgg16_tilecnn` is a released name and does not change.

## Development inside the DeepTile repository

DeepTile pins FixQuant as the root-level `FixQuant/` submodule. Use the pinned
submodule as the active editable installation, so that the exporter DeepTile
calls is the one recorded in its source state:

```bash
git submodule update --init FixQuant
python -m pip install --no-deps -e ./FixQuant
```

## Framework entry points

From the parent DeepTile repository, fetch a released model and export its
ModelPackage with:

```bash
make -C scripts model-fetch \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
make -C scripts model-export \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
```

The validated ModelPackage is written below `build/models/` using the same
release identity. These targets wrap `scripts/model_zoo.sh fetch` and
`tools/export_deeptile_graph.py`, described in the main README.

## Related documentation

- [DeepTile exporter and integer digital twin](deeptile_exporter_and_digital_twin.md)
- [DeepTile Graph Handoff Specification](../graph_handoff_spec.md)
- [Deployment and graph export](../DEPLOY.md)
