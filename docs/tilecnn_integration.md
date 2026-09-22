# Using FixQuant with TileCNN

FixQuant is the model-preparation component of the
[TileCNN](https://github.com/mokweri/TileCNN) framework. FixQuant owns training,
quantization, the integer digital twin, and model export. TileCNN owns
hardware-specific legality checks, packing, tiling, scheduling, descriptor
generation, and FPGA execution. Their external boundary is the
[TileCNN Graph Handoff Specification](../graph_handoff_spec.md).

## Development inside the TileCNN repository

TileCNN pins FixQuant as the root-level `FixQuant/` submodule. Use the pinned
submodule as the active editable installation, so that the exporter TileCNN
calls is the one recorded in its source state:

```bash
git submodule update --init FixQuant
python -m pip install --no-deps -e ./FixQuant
```

## Framework entry points

From the parent TileCNN repository, fetch a released model and export its
ModelPackage with:

```bash
make -C scripts model-fetch \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
make -C scripts model-export \
    ZOO_MODEL=resnet50/imagenet1k/int8-tqt@v1.0.0
```

The validated ModelPackage is written below `build/models/` using the same
release identity. These targets wrap `scripts/model_zoo.sh fetch` and
`tools/export_tilecnn_graph.py`, described in the main README.

## Related documentation

- [TileCNN exporter and integer digital twin](tilecnn_exporter_and_digital_twin.md)
- [TileCNN Graph Handoff Specification](../graph_handoff_spec.md)
- [Deployment and graph export](../DEPLOY.md)
