"""FusedConvBN training-correctness tests (the June-2026 MobileNet decay bugs)."""

import torch
import torch.nn as nn

from fixquant.quantization.fused_conv_bn import FusedConvBN
from fixquant.quantization.tqt_quantizer import TQTQuantizer


def make_fused(freeze_bn_delay=6000, bias=False):
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=bias)
    bn = nn.BatchNorm2d(8)
    # simulate a pretrained checkpoint state
    with torch.no_grad():
        bn.num_batches_tracked.fill_(736_839)
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_()
    return FusedConvBN.from_float(conv, bn, freeze_bn_delay=freeze_bn_delay)


def disable_quant(module):
    for m in module.modules():
        if isinstance(m, TQTQuantizer):
            m.enable_quant(False)


def test_num_batches_tracked_reset_at_fusion():
    fused = make_fused()
    assert fused.bn_mod.num_batches_tracked.item() == 0


def test_no_freeze_on_first_batch():
    fused = make_fused(freeze_bn_delay=6000)
    fused.train()
    fused(torch.randn(4, 3, 8, 8))
    assert not fused.frozen


def test_freeze_preserves_optimizer_ownership():
    fused = make_fused()
    disable_quant(fused)
    opt = torch.optim.Adam(fused.parameters(), lr=1e-2)
    owned = {id(p) for g in opt.param_groups for p in g["params"]}

    fused.train()
    fused(torch.randn(4, 3, 8, 8)).sum().backward()
    opt.step()

    fused.freeze()
    assert id(fused.conv_mod.weight) in owned, \
        "freeze() must not re-create the weight Parameter"
    assert id(fused.conv_mod.bias) in owned, \
        "freeze() must not re-create the bias Parameter"

    # the weight must keep training after the freeze
    before = fused.conv_mod.weight.detach().clone()
    opt.zero_grad()
    fused(torch.randn(4, 3, 8, 8)).sum().backward()
    opt.step()
    assert not torch.equal(before, fused.conv_mod.weight.detach()), \
        "conv weight stopped training after freeze()"


def test_freeze_matches_eval_folding():
    fused = make_fused()
    disable_quant(fused)
    fused.eval()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        out_unfrozen = fused(x)
        fused.freeze()
        out_frozen = fused(x)
    assert torch.allclose(out_unfrozen, out_frozen, atol=1e-5), \
        "freeze() changed the eval-mode output"


def test_eval_matches_float_conv_bn():
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=True)
    bn = nn.BatchNorm2d(8)
    with torch.no_grad():
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_()
    ref = nn.Sequential(conv, bn).eval()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        expected = ref(x)

    import copy
    fused = FusedConvBN.from_float(copy.deepcopy(conv), copy.deepcopy(bn))
    disable_quant(fused)
    fused.eval()
    with torch.no_grad():
        got = fused(x)
    assert torch.allclose(expected, got, atol=1e-5)


def test_auto_freeze_after_delay():
    fused = make_fused(freeze_bn_delay=3)
    fused.train()
    for _ in range(5):
        fused(torch.randn(2, 3, 8, 8))
    assert fused.frozen


def test_freeze_delay_none_never_freezes():
    fused = make_fused(freeze_bn_delay=None)
    fused.train()
    for _ in range(5):
        fused(torch.randn(2, 3, 8, 8))
    assert not fused.frozen
