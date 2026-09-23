"""TileCNN legality of the vgg16_tilecnn variant, end to end.

The structural tests assert the two fabric limits that forced the architecture
change (pooling geometry, weight-buffer footprint) directly on the float model.
The slow test drives the real pipeline -- QAT -> hardware model -> ModelPackage
export -- and runs the standalone acceptance checker over the result.
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

from fixquant.models import get_model
from fixquant.models.vgg_tilecnn import (
    TILECNN_MAXPOOL,
    WEIGHT_BUFFER_BUDGET,
    VGGTileCNN,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load(open(REPO_ROOT / "configs/quant_config.yaml"))
CHANNEL_PACK = 16


def weight_buffer_cost(module):
    if isinstance(module, nn.Conv2d):
        cin = module.in_channels // module.groups
        kh, kw = module.kernel_size
    else:
        cin, kh, kw = module.in_features, 1, 1
    return math.ceil(cin / CHANNEL_PACK) * kh * kw


@pytest.fixture(scope="module")
def model():
    return VGGTileCNN().eval()


def test_every_maxpool_uses_the_single_legal_geometry(model):
    pools = [m for m in model.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(pools) == 5
    for pool in pools:
        assert pool.kernel_size == TILECNN_MAXPOOL["kernel_size"]
        assert pool.stride == TILECNN_MAXPOOL["stride"]
        assert pool.padding == TILECNN_MAXPOOL["padding"]


def test_every_weight_layer_fits_the_weight_buffer(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            cost = weight_buffer_cost(module)
            assert cost <= WEIGHT_BUFFER_BUDGET, f"{name} costs {cost}"


def test_stock_vgg16_classifier_is_what_forced_the_change():
    """The constraint this variant exists to satisfy is real, not hypothetical."""
    stock = get_model("vgg16_bn", pretrained=False)
    assert weight_buffer_cost(stock.classifier[0]) > WEIGHT_BUFFER_BUDGET


def test_feature_stack_shapes_match_stock_vgg(model):
    """3x3/s2/p1 pooling is shape-identical to 2x2/s2 on even inputs."""
    stock = get_model("vgg16_bn", pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert model.features(x).shape == stock.features(x).shape == (1, 512, 7, 7)


def test_head_produces_logits_through_a_gap_that_follows_a_conv(model):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert model(x).shape == (2, 1000)

    ordered = list(model.classifier)
    conv_index = next(i for i, m in enumerate(ordered) if isinstance(m, nn.Conv2d))
    gap_index = next(i for i, m in enumerate(ordered)
                     if isinstance(m, nn.AdaptiveAvgPool2d))
    between = ordered[conv_index + 1:gap_index]
    assert all(isinstance(m, nn.ReLU) for m in between), \
        "nothing but activations may sit between the conv and the GAP"

    with torch.no_grad():
        feats = model.features(x)
        for module in ordered[:gap_index + 1]:
            feats = module(feats)
    assert feats.shape[1:] == (4096, 1, 1)


def test_pretrained_features_transfer_and_head_is_new():
    """Only the 14.7M-parameter convolution stack carries over."""
    variant = VGGTileCNN()
    stock_features = get_model("vgg16_bn", pretrained=False).features.state_dict()
    # Same keys and shapes means the pretrained load needs no remapping.
    assert variant.features.state_dict().keys() == stock_features.keys()
    for key, tensor in stock_features.items():
        assert variant.features.state_dict()[key].shape == tensor.shape
    backbone_params = sum(p.numel() for p in variant.features.parameters())
    assert 14_000_000 < backbone_params < 15_000_000


def test_freeze_backbone_holds_features_and_their_bn_statistics():
    variant = VGGTileCNN().freeze_backbone(True)
    assert not any(p.requires_grad for p in variant.features.parameters())
    assert all(p.requires_grad for p in variant.classifier.parameters())

    variant.train()
    assert not variant.features.training, "frozen BN must stay in inference mode"
    assert variant.classifier.training

    bn = next(m for m in variant.features.modules() if isinstance(m, nn.BatchNorm2d))
    before = bn.running_mean.clone()
    variant(torch.randn(2, 3, 64, 64))
    assert torch.equal(bn.running_mean, before)

    variant.freeze_backbone(False).train()
    assert variant.features.training
    assert all(p.requires_grad for p in variant.features.parameters())


def test_pooling_ablation_model_keeps_stock_classifier():
    """vgg16_bn_pool3 isolates the pooling cost: pooling swapped, nothing else."""
    ablation = get_model("vgg16_bn_pool3", pretrained=False)
    stock = get_model("vgg16_bn", pretrained=False)
    for pool in (m for m in ablation.modules() if isinstance(m, nn.MaxPool2d)):
        assert (pool.kernel_size, pool.stride, pool.padding) == (3, 2, 1)
    assert str(ablation.classifier) == str(stock.classifier)
    with torch.no_grad():
        assert ablation.eval()(torch.randn(1, 3, 224, 224)).shape == (1, 1000)


def test_linear_relu_is_fused_as_a_post_op_not_dropped():
    """TileCNN lowers `linear` to a 1x1 conv, so it carries the same post-op
    block as conv2d. A ReLU after a Linear must survive the export."""
    from fixquant.export.tilecnn_exporter import _tilecnn_linear

    ifm = torch.full((4, 1, 1), 1, dtype=torch.int8)
    weight = torch.tensor([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=torch.int8)
    bias = torch.zeros(2, dtype=torch.int8)
    kwargs = dict(ifm_frac=0, weight_frac=0, bias_frac=0, out_frac=0)

    plain = _tilecnn_linear(ifm, weight, bias, **kwargs)
    relued = _tilecnn_linear(ifm, weight, bias, post_ops={"relu": True}, **kwargs)
    assert plain[1].item() < 0
    assert relued[1].item() == 0
    assert relued[0].item() == plain[0].item()


@pytest.mark.slow
def test_export_produces_a_legal_tilecnn_package(tmp_path):
    """Full pipeline on random weights: the graph the FPGA would be handed."""
    from fixquant.graph.qat_processor import QatProcessor
    from fixquant.graph.inference_processor import InferProcessor
    from fixquant.emulation.model_introspector import StdModelInspector
    from fixquant.export.tilecnn_exporter import TileCNNGraphExporter

    torch.manual_seed(0)
    proc = QatProcessor(VGGTileCNN(num_classes=10), CONFIG)
    qat = proc.quantize()
    calib = [(torch.randn(2, 3, 224, 224), torch.randint(0, 10, (2,)))]
    proc.calibrate(calib, "cpu")
    proc.freeze()

    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()
    inspector = StdModelInspector(hw, default_input_frac=infer.input_frac or 5)
    inspector.collect_all_shapes(torch.randn(1, 3, 224, 224))

    out_dir = tmp_path / "package"
    TileCNNGraphExporter(inspector, model_name="vgg16_tilecnn",
                         default_input_frac=infer.input_frac or 5).export(str(out_dir))

    graph = json.loads((out_dir / "graph.json").read_text())
    ops = [n["op"] for n in graph["nodes"]]
    assert ops.count("conv2d") == 14      # 13 feature convs + the fc1 head conv
    assert ops.count("maxpool2d") == 5
    assert ops.count("gap2d") == 1
    assert ops.count("linear") == 2

    # The acceptance checker must pass on the emitted graph. It also validates
    # manifest checksums, so write the manifest the export tool would write.
    from tools.export_tilecnn_graph import write_package_manifest

    class _Args:
        model = "vgg16_tilecnn"
    write_package_manifest(out_dir, REPO_ROOT, _Args(), out_dir / "absent.pth.tar",
                           REPO_ROOT / "configs/quant_config.yaml",
                           out_dir / "absent.JPEG")

    check = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/check_tilecnn_legality.py"),
         str(out_dir)],
        capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
