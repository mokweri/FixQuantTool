"""EfficientNet-Lite0: structure, DeepTile legality, and the timm conversion.

The structural tests check the float model directly. The conversion test proves
the plain definition reproduces timm's logits. The slow test drives the real
pipeline -- QAT -> hardware model -> ModelPackage export -- and runs the
standalone acceptance checker over the result.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from fixquant.graph.qat_processor import preflight_check
from fixquant.models.efficientnet_lite import (
    EFFICIENTNET_LITE0_CHECKPOINT,
    EfficientNetLite,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load(open(REPO_ROOT / "configs/quant_config.yaml"))
WEIGHT_BUFFER_BUDGET = 512


def depthwise_convs(model):
    return [m for m in model.modules()
            if isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels]


def test_topology_matches_efficientnet_lite0():
    model = EfficientNetLite()
    assert sum(p.numel() for p in model.parameters()) == 4652008
    kernels = [m.kernel_size[0] for m in depthwise_convs(model)]
    assert kernels.count(3) == 7 and kernels.count(5) == 9
    blocks = [b for stage in model.blocks for b in stage]
    assert len(blocks) == 16
    assert sum(b.has_skip for b in blocks) == 9


def test_every_activation_is_a_separate_relu6():
    model = EfficientNetLite()
    # FixQuant folds each (Conv2d, BatchNorm2d) pair; an activation fused into
    # a batch-norm subclass would be dropped silently. Only plain modules here.
    assert all(type(m) is nn.BatchNorm2d
               for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    assert sum(isinstance(m, nn.ReLU6) for m in model.modules()) == 33
    assert preflight_check(model, raise_on_error=False) == []


def test_every_layer_fits_the_weight_buffer():
    for name, m in EfficientNetLite().named_modules():
        if isinstance(m, nn.Conv2d):
            cost = math.ceil(m.in_channels // m.groups / 16) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            cost = math.ceil(m.in_features / 16)
        else:
            continue
        assert cost <= WEIGHT_BUFFER_BUDGET, name


def test_converted_checkpoint_reproduces_timm():
    timm = pytest.importorskip("timm")
    if not EFFICIENTNET_LITE0_CHECKPOINT.is_file():
        pytest.skip("converted checkpoint not present")
    reference = timm.create_model("efficientnet_lite0", pretrained=False).eval()
    state = torch.load(EFFICIENTNET_LITE0_CHECKPOINT, map_location="cpu",
                       weights_only=False)["state_dict"]
    reference.load_state_dict(state, strict=True)
    model = EfficientNetLite().eval()
    model.load_state_dict(state, strict=True)
    inputs = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(reference(inputs), model(inputs))


@pytest.mark.slow
@pytest.mark.parametrize("cle", [False, True], ids=["plain", "cle_keep_relu6"])
def test_export_produces_a_legal_deeptile_package(tmp_path, cle):
    """Full pipeline on random weights: the graph the FPGA would be handed."""
    from fixquant.graph.qat_processor import QatProcessor
    from fixquant.graph.inference_processor import InferProcessor
    from fixquant.emulation.model_introspector import StdModelInspector
    from fixquant.export.deeptile_exporter import DeepTileGraphExporter

    torch.manual_seed(0)
    model = EfficientNetLite(num_classes=10)
    if cle:
        from fixquant.quantization.equalization import equalize_model
        model = equalize_model(model, replace_relu6=False)
    proc = QatProcessor(model, CONFIG)
    qat = proc.quantize()
    calib = [(torch.randn(2, 3, 224, 224), torch.randint(0, 10, (2,)))]
    proc.calibrate(calib, "cpu")
    proc.freeze()

    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()
    inspector = StdModelInspector(hw, default_input_frac=infer.input_frac or 5)
    inspector.collect_all_shapes(torch.randn(1, 3, 224, 224))

    out_dir = tmp_path / "package"
    DeepTileGraphExporter(inspector, model_name="efficientnet_lite0",
                          default_input_frac=infer.input_frac or 5).export(str(out_dir))

    graph = json.loads((out_dir / "graph.json").read_text())
    ops = [n["op"] for n in graph["nodes"]]
    assert ops.count("conv2d") == 49      # stem, 2 + 15 x 3 block convs, head
    assert ops.count("gap2d") == 1
    assert ops.count("linear") == 1
    posts = [n.get("post_ops", {}) for n in graph["nodes"]]
    assert sum(p.get("relu6", False) for p in posts) == 33
    assert not any(p.get("relu", False) for p in posts)
    assert sum(p.get("residual_add", False) for p in posts) == 9
    kernels = {n["attrs"]["kernel"][0] for n in graph["nodes"] if n["op"] == "conv2d"}
    assert kernels == {1, 3, 5}

    from tools.export_deeptile_graph import write_package_manifest

    class _Args:
        model = "efficientnet_lite0"
    write_package_manifest(out_dir, REPO_ROOT, _Args(), out_dir / "absent.pth.tar",
                           REPO_ROOT / "configs/quant_config.yaml",
                           out_dir / "absent.JPEG")

    check = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/check_deeptile_legality.py"),
         str(out_dir)],
        capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
