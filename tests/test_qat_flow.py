"""End-to-end structural tests: quantize → calibrate → freeze → hardware model.

These do not train and do not assert accuracy — they assert that the pipeline
wiring (graph passes, qconfig completeness, strict conversion) holds for a
depthwise/ReLU6 block model and for torchvision MobileNetV2.
"""

import pytest
import torch
import yaml
from pathlib import Path

from fixquant.graph.qat_processor import QatProcessor, preflight_check
from fixquant.graph.inference_processor import InferProcessor
from fixquant.quantization.fused_conv_bn import FusedConvBN
from fixquant.quantization.tqt_quantizer import TQTQuantizer

from tests.models import TinyMobileBlockNet, synthetic_loader

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/quant_config.yaml"))


def build_qat(model, loader=None):
    proc = QatProcessor(model, CONFIG)
    qat = proc.quantize()
    if loader is not None:
        proc.calibrate(loader, "cpu")
    return proc, qat


def test_tiny_net_pipeline_to_hardware():
    proc, qat = build_qat(TinyMobileBlockNet(), synthetic_loader())
    proc.freeze()

    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()
    out = hw(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 10)
    assert out.dtype == torch.float32


def test_relu6_bounded_ranges_are_set():
    proc, qat = build_qat(TinyMobileBlockNet())
    modules = dict(qat.named_modules())
    bounded = [m.act_quantizer.bounded_range
               for m in modules.values()
               if isinstance(m, FusedConvBN) and m.act_quantizer.bounded_range]
    # expand + dw convs feed ReLU6 → (0, 6); stem feeds ReLU → (0, None)
    assert (0.0, 6.0) in bounded
    assert (0.0, None) in bounded


def test_qconfig_covers_all_hardware_layers():
    """Strict conversion must succeed — i.e. no layer relies on the removed
    silent frac defaults."""
    proc, qat = build_qat(TinyMobileBlockNet(), synthetic_loader())
    proc.freeze()
    infer = InferProcessor(qat, CONFIG)
    infer.convert_to_std_model()
    qconfig = infer.generate_qconfig()
    for layer, params in qconfig.items():
        if layer == "x":
            continue
        assert params.get("out") is not None, f"{layer} missing output frac"


def test_input_frac_comes_from_quant_stub():
    proc, qat = build_qat(TinyMobileBlockNet(), synthetic_loader())
    proc.freeze()
    infer = InferProcessor(qat, CONFIG)
    infer.convert_to_std_model()
    assert infer.input_frac is not None
    qconfig = infer.generate_qconfig()
    assert qconfig["x"]["out"] == infer.input_frac


def test_depthwise_conv_survives_conversion():
    proc, qat = build_qat(TinyMobileBlockNet(), synthetic_loader())
    proc.freeze()
    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()
    dw = [m for m in hw.modules()
          if type(m).__name__ == "HardwareConv2d" and m.groups > 1]
    assert len(dw) == 1
    assert dw[0].w_int8.shape == (16, 1, 3, 3)


def test_preflight_accepts_supported_rejects_unsupported():
    assert preflight_check(TinyMobileBlockNet()) == []

    class BadNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3)
            self.act = torch.nn.Hardswish()

        def forward(self, x):
            return self.act(self.conv(x))

    issues = preflight_check(BadNet(), raise_on_error=False)
    assert any("Hardswish" in i for i in issues)
    with pytest.raises(ValueError):
        preflight_check(BadNet())


def test_parity_sweep_qat_vs_hardware():
    from fixquant.diagnostics import parity_sweep
    proc, qat = build_qat(TinyMobileBlockNet(), synthetic_loader())
    proc.freeze()
    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()

    rows = parity_sweep(qat, hw, torch.randn(1, 3, 32, 32))
    assert rows, "no layers matched between QAT and hardware model"
    for r in rows:
        assert r["mismatches"] >= 0, r["note"]

    # The first layer sees identical inputs in both models, so its diff is
    # purely fake-quant vs two-step hardware rounding: at most ±1 LSB.
    # Downstream layers see slightly different inputs (propagated ±1s), so
    # only a loose bound applies there.
    first = rows[0]
    assert first["max_abs_diff"] <= 1, f"{first['layer']}: max diff {first['max_abs_diff']}"
    for r in rows:
        assert r["max_abs_diff"] <= 8, f"{r['layer']}: max diff {r['max_abs_diff']} (wiring bug?)"


@pytest.mark.slow
def test_mobilenet_v2_pipeline():
    import torchvision.models as models
    model = models.mobilenet_v2(weights=None)
    assert preflight_check(model) == []

    loader = synthetic_loader(n_batches=1, batch_size=2, size=224, num_classes=1000)
    proc, qat = build_qat(model, loader)
    proc.freeze()

    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()
    out = hw(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000)

    n_dw = sum(1 for m in hw.modules()
               if type(m).__name__ == "HardwareConv2d" and m.groups > 1)
    assert n_dw == 17  # MobileNetV2 has 17 depthwise convs
