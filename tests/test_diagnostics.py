"""Quantizer report diagnostics."""

import math

import torch
import yaml
from pathlib import Path

from fixquant.diagnostics import quantizer_report, weight_sqnr_db
from fixquant.graph.qat_processor import QatProcessor

from tests.models import TinyMobileBlockNet, synthetic_loader

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/quant_config.yaml"))


def test_weight_sqnr_db_zero_tensor():
    """All-zero tensors (FusedConvBN's synthetic zero bias) must not raise."""
    assert weight_sqnr_db(torch.zeros(16), frac=6) == float("-inf")


def test_quantizer_report_on_calibrated_model_with_biasless_convs():
    """quantizer_report must survive bias quantizers whose tensor is all zeros
    (convs created with bias=False get a zero bias Parameter at fusion)."""
    proc = QatProcessor(TinyMobileBlockNet(), CONFIG)
    qat = proc.quantize()
    proc.calibrate(synthetic_loader(), "cpu")

    rows = quantizer_report(qat)
    assert rows
    bias_rows = [r for r in rows if "bias_quantizer" in r["quantizer"]]
    assert any(r["sqnr_db"] == "zero-tensor" for r in bias_rows)
    # every numeric sqnr must be finite
    for r in rows:
        if isinstance(r["sqnr_db"], float):
            assert math.isfinite(r["sqnr_db"])
