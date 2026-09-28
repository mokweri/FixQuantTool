"""The names TileCNN used before it was renamed DeepTile keep working."""
import importlib
import sys

import pytest
import torch

from fixquant.emulation.fxp_emu_modules import HardwareConv2d
from fixquant.export import deeptile_exporter


def test_former_exporter_module_is_an_alias():
    sys.modules.pop("fixquant.export.tilecnn_exporter", None)
    with pytest.warns(DeprecationWarning):
        legacy = importlib.import_module("fixquant.export.tilecnn_exporter")
    assert legacy.TileCNNGraphExporter is deeptile_exporter.DeepTileGraphExporter
    assert legacy._tilecnn_conv2d is deeptile_exporter._deeptile_conv2d
    assert legacy._tilecnn_gap is deeptile_exporter._deeptile_gap


def test_former_backend_name_selects_the_deeptile_backend():
    weight = torch.zeros(2, 1, 1, 1, dtype=torch.int8)
    assert HardwareConv2d(weight, None, backend="tilecnn").backend == "deeptile"
    assert HardwareConv2d(weight, None).backend == "deeptile"


def test_exports_use_the_deeptile_identifiers():
    assert deeptile_exporter.GRAPH_SCHEMA == "deeptile.graph.v1"
    assert deeptile_exporter.MODEL_PACKAGE_SCHEMA == "deeptile.model-package.v1"
    assert set(deeptile_exporter.GRAPH_SCHEMAS) == {"tilecnn.graph.v1", "deeptile.graph.v1"}
