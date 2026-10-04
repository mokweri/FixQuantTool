"""Float exporter: a frozen QAT model becomes plain PyTorch that reproduces it."""

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from fixquant.export.float_exporter import (
    FloatExportError, emit_source, quantization_disabled, quantization_record,
    to_float_graph,
)
from fixquant.graph.qat_processor import QatProcessor

from tests.models import TinyMobileBlockNet, synthetic_loader

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/quant_config.yaml"))


@pytest.fixture(scope="module")
def frozen_qat():
    torch.manual_seed(0)
    proc = QatProcessor(TinyMobileBlockNet(), CONFIG)
    proc.quantize()
    proc.calibrate(synthetic_loader(), "cpu")
    proc.freeze()
    return proc.qat_model.eval()


def _load(source: str, tmp_path: Path, class_name: str):
    path = tmp_path / "model.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("exported_tiny", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)()


@pytest.mark.parametrize("weights", ["deployed", "latent"])
def test_export_reproduces_qat_without_activation_quantization(frozen_qat, tmp_path, weights):
    float_gm, counts = to_float_graph(frozen_qat, weights=weights)
    assert not any(type(m).__module__.startswith("fixquant") for m in float_gm.modules())
    assert counts.get("quantizer removed") == 1

    source = emit_source(float_gm, "TinyFloat", "test")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports == ["import torch", "import torch.nn as nn"]
    model = _load(source, tmp_path, "TinyFloat")
    model.load_state_dict(float_gm.state_dict(), strict=True)
    model.eval()

    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad(), quantization_disabled(frozen_qat, weights=weights):
        want = frozen_qat(x)
    with torch.no_grad():
        got = model(x)
    assert torch.equal(got, want)


def test_deployed_weights_are_the_quantized_values(frozen_qat):
    deployed, _ = to_float_graph(frozen_qat, weights="deployed")
    latent, _ = to_float_graph(frozen_qat, weights="latent")
    differ = [name for name, p in deployed.state_dict().items()
              if not torch.equal(p, latent.state_dict()[name])]
    assert differ, "deployed weights should differ from the latent float parameters"


def test_quantization_disabled_restores_the_model(frozen_qat):
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = frozen_qat(x)
        with quantization_disabled(frozen_qat):
            pass
        after = frozen_qat(x)
    assert torch.equal(before, after)


def test_quantization_record_covers_every_conv(frozen_qat):
    record = quantization_record(frozen_qat)
    assert "input" in record
    convs = [k for k, v in record.items() if "weight" in v]
    assert convs and all(isinstance(record[k]["output"], int) for k in convs)


def test_unfrozen_model_is_rejected():
    torch.manual_seed(0)
    proc = QatProcessor(TinyMobileBlockNet(), CONFIG)
    qat = proc.quantize()
    with pytest.raises(FloatExportError):
        to_float_graph(qat.eval())
