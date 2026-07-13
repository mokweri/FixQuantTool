"""TQT calibration, bounded activation ranges, and threshold freezing."""

import torch

from fixquant.quantization.tqt_quantizer import TQTQuantizer


def test_calibration_sets_pow2_threshold():
    q = TQTQuantizer(bitwidth=8, tensor_type='act')
    q.start_calibration()
    for _ in range(5):
        q(torch.empty(4, 16, 8, 8).uniform_(-3.0, 3.0))
    frac = q.finish_calibration()
    # data range ±3 → threshold 4 → frac 5 (or 6 if MSE prefers clipping)
    assert frac in (5, 6)
    log_t = q.log_threshold.item()
    assert log_t == int(log_t), "calibrated threshold must be an exact power of 2"
    assert 8 - 1 - int(log_t) == frac


def test_calibration_multi_batch_beats_single_extreme_batch():
    """Later batches must influence the result (the old flow only used batch 1)."""
    q = TQTQuantizer(bitwidth=8, tensor_type='act')
    q.start_calibration()
    q(torch.empty(2, 8, 8, 8).uniform_(-0.1, 0.1))     # small first batch
    for _ in range(4):
        q(torch.empty(2, 8, 8, 8).uniform_(-7.0, 7.0))  # true range appears later
    frac = q.finish_calibration()
    # a frac fitted to ±0.1 would be ~10; the true ±7 range needs ~4
    assert frac <= 5


def test_bounded_range_relu6():
    q = TQTQuantizer(bitwidth=8, tensor_type='act')
    q.bounded_range = (0.0, 6.0)
    q.start_calibration()
    for _ in range(4):
        q(torch.empty(2, 8, 8, 8).uniform_(-20.0, 20.0))
    frac = q.finish_calibration()
    max_repr = 127 / (2.0 ** frac)
    assert max_repr >= 6.0, "6.0 must be representable for a ReLU6-bounded tensor"
    assert frac == 4


def test_bounded_range_warmup_init():
    q = TQTQuantizer(bitwidth=8, tensor_type='act')
    q.bounded_range = (0.0, 6.0)
    q(torch.empty(2, 4, 4, 4).uniform_(-20.0, 20.0))  # triggers warmup
    assert abs(2.0 ** q.log_threshold.item() - 6.0) < 1e-5


def test_freeze_quant_stops_gradients():
    q = TQTQuantizer(bitwidth=8, tensor_type='act')
    q(torch.randn(2, 4))  # warmup
    q.freeze_quant(True)
    assert not q.log_threshold.requires_grad
    q.freeze_quant(False)
    assert q.log_threshold.requires_grad


def test_enable_quant_passthrough():
    q = TQTQuantizer(bitwidth=8, tensor_type='act')
    x = torch.randn(4, 4) * 3
    q(x)  # warmup + quantize
    q.enable_quant(False)
    assert torch.equal(q(x), x)
    q.enable_quant(True)
    assert not torch.equal(q(x), x)
