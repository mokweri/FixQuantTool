"""TileCNN exporter tests: full-graph export of a depthwise/ReLU6 model,
relu6/groups in graph.json, legality checks, and CLE numerics."""

import json

import pytest
import torch
import yaml
from pathlib import Path

from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter, _check_shift_legality

from tests.models import TinyMobileBlockNet, synthetic_loader

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/quant_config.yaml"))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    proc = QatProcessor(TinyMobileBlockNet(), CONFIG)
    qat = proc.quantize()
    proc.calibrate(synthetic_loader(), "cpu")
    proc.freeze()

    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()

    inspector = StdModelInspector(hw, default_input_frac=infer.input_frac or 5)
    nodes = inspector.topological_order()
    inspector.register_activation_hooks(nodes, capture_input=True, capture_output=True)
    with torch.no_grad():
        inspector.run_and_capture(torch.randn(1, 3, 32, 32))

    out_dir = tmp_path_factory.mktemp("export")
    exporter = TileCNNGraphExporter(inspector, model_name="tiny_mobile_block",
                                    default_input_frac=infer.input_frac or 5)
    exporter.export(str(out_dir))
    return out_dir, json.load(open(out_dir / "graph.json"))


def test_export_writes_graph_and_artifacts(exported):
    out_dir, graph = exported
    assert graph["schema"] == "tilecnn.graph.v1"
    assert (out_dir / "graph.json").exists()
    for ref_id in graph["graph"]["references"].values():
        f = out_dir / graph["tensors"][ref_id]["file"]
        assert f.exists() and f.stat().st_size > 0


def test_export_depthwise_conv_node(exported):
    _, graph = exported
    dw_nodes = [n for n in graph["nodes"] if n["op"] == "conv2d" and n["attrs"]["groups"] > 1]
    assert len(dw_nodes) == 1
    dw = dw_nodes[0]
    w_shape = graph["tensors"][dw["inputs"]["weight"]]["shape"]
    assert w_shape == [16, 1, 3, 3]


def test_export_relu6_post_op_preserved(exported):
    _, graph = exported
    relu6_nodes = [n for n in graph["nodes"] if n.get("post_ops", {}).get("relu6")]
    relu_nodes = [n for n in graph["nodes"] if n.get("post_ops", {}).get("relu")]
    assert len(relu6_nodes) == 2, "expand + depthwise convs must carry relu6, not relu"
    assert len(relu_nodes) >= 1     # stem conv carries plain relu


def test_export_all_params_int8_with_frac(exported):
    _, graph = exported
    for tid, spec in graph["tensors"].items():
        assert spec["dtype"] == "int8", tid
        assert isinstance(spec["frac"], int), tid


def test_legality_check_rejects_bad_shift():
    graph = {
        "tensors": {
            "in": {"frac": 40}, "w": {"frac": 7}, "b": {"frac": 7}, "out": {"frac": -10},
        },
        "nodes": [{
            "id": "conv", "op": "conv2d",
            "inputs": {"ifm": "in", "weight": "w", "bias": "b"},
            "outputs": {"ofm": "out"}, "post_ops": {},
        }],
    }
    with pytest.raises(ValueError, match="shift_out"):
        _check_shift_legality(graph)


def test_cle_preserves_function_and_equalizes():
    """CLE itself is mathematically exact through ReLU: compare the equalized
    model against a BN-folded + ReLU6→ReLU (but NOT equalized) reference so
    only the equalization step is under test."""
    import copy
    from torch.fx.experimental.optimization import fuse
    from fixquant.quantization.equalization import (
        cross_layer_equalize, replace_relu6_with_relu)

    torch.manual_seed(0)
    model = TinyMobileBlockNet().eval()
    # make the dw ranges pathologically unbalanced, like folded MobileNet
    with torch.no_grad():
        model.dw.weight[::2] *= 20.0

    ref_model = replace_relu6_with_relu(fuse(copy.deepcopy(model)))
    eq_model = replace_relu6_with_relu(fuse(copy.deepcopy(model)))

    def dw_ratio(gm):
        dw = next(m for m in gm.modules()
                  if isinstance(m, torch.nn.Conv2d) and m.groups > 1)
        r = dw.weight.detach().abs().amax(dim=(1, 2, 3))
        return (r.max() / r.min().clamp_min(1e-12)).item()

    ratio_before = dw_ratio(eq_model)
    cross_layer_equalize(eq_model, iterations=2)
    ratio_after = dw_ratio(eq_model)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = ref_model(x)
        got = eq_model(x)
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()
    assert ratio_after < ratio_before / 2, (ratio_before, ratio_after)
