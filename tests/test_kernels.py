"""Bit-exactness of the digital-twin modules against the exporter's TileCNN
reference kernels. These two implementations (plus the HLS C code) must agree
exactly — every shift-sign combination, grouped conv, relu/relu6, residual add.
"""

import pytest
import torch

from fixquant.emulation.fxp_emu_modules import (
    HardwareConv2d, HardwareLinear, HardwareGAP, HardwareMaxPool2d,
    HardwareElementwiseAdd, HardwareRelu6,
)
from fixquant.export.tilecnn_exporter import (
    _tilecnn_conv2d, _tilecnn_linear, _tilecnn_gap, _tilecnn_maxpool,
    _tilecnn_residual_add, _relu6_max,
)


def rand_i8(*shape):
    return torch.randint(-128, 128, shape, dtype=torch.int8)


def conv_node(groups=1, stride=1, padding=1, relu=False, relu6=False, kernel=3):
    return {
        "attrs": {"kernel": [kernel, kernel], "stride": [stride, stride],
                  "padding": [padding] * 4, "dilation": [1, 1], "groups": groups},
        "post_ops": {"relu": relu, "relu6": relu6},
    }


FRAC_COMBOS = [
    # (fin, fw, fb, fout) covering shift_out >0, ==0, <0 and bias shifts
    (5, 7, 6, 5),   # shift_out = 7
    (4, 3, 4, 7),   # shift_out = 0
    (2, 2, 7, 6),   # shift_out = -2 (left shift)
    (5, 7, 7, 4),
]


@pytest.mark.parametrize("fin,fw,fb,fout", FRAC_COMBOS)
@pytest.mark.parametrize("groups", [1, 8])
def test_conv2d_twin_matches_reference(fin, fw, fb, fout, groups):
    c = 8
    x = rand_i8(1, c, 6, 6)
    w = rand_i8(c, c // groups, 3, 3)
    b = rand_i8(c)

    twin = HardwareConv2d(w, b, stride=1, padding=1, groups=groups,
                          frac_w=fw, frac_b=fb, frac_din=fin, frac_dout=fout)
    out_twin = twin(x)

    out_ref = _tilecnn_conv2d(x.squeeze(0), w, b, conv_node(groups=groups),
                              fin, fw, fb, fout)
    assert torch.equal(out_twin.squeeze(0), out_ref)


@pytest.mark.parametrize("act", ["relu", "relu6"])
def test_conv2d_activation_post_ops(act):
    fin, fw, fb, fout = 5, 7, 6, 4
    x = rand_i8(1, 4, 6, 6)
    w = rand_i8(4, 4, 3, 3)
    b = rand_i8(4)

    twin = HardwareConv2d(w, b, stride=1, padding=1,
                          frac_w=fw, frac_b=fb, frac_din=fin, frac_dout=fout,
                          relu=(act == "relu"), relu6=(act == "relu6"))
    out_twin = twin(x).squeeze(0)

    node = conv_node(relu=(act == "relu"), relu6=(act == "relu6"))
    out_ref = _tilecnn_conv2d(x.squeeze(0), w, b, node, fin, fw, fb, fout)
    assert torch.equal(out_twin, out_ref)
    assert out_twin.min() >= 0
    if act == "relu6":
        assert out_twin.max() <= _relu6_max(fout)


def test_relu6_never_degrades_to_relu():
    # at fout=4 the int clamp is 96 < 127: relu and relu6 genuinely differ
    fout = 4
    x = torch.full((1, 1, 2, 2), 127, dtype=torch.int8)
    r6 = HardwareRelu6(frac_in=fout)
    assert r6(x).max() == _relu6_max(fout) == 96


@pytest.mark.parametrize("relu6", [False, True])
def test_conv_residual_add_twin_matches_reference(relu6):
    fin, fw, fb, fout, fres = 5, 7, 6, 4, 6
    x = rand_i8(1, 4, 6, 6)
    res = rand_i8(1, 4, 6, 6)
    w = rand_i8(4, 4, 3, 3)
    b = rand_i8(4)

    twin = HardwareConv2d(w, b, stride=1, padding=1,
                          frac_w=fw, frac_b=fb, frac_din=fin, frac_dout=fout,
                          residual_add=True, residual_frac=fres,
                          post_add_relu=not relu6, post_add_relu6=relu6)
    out_twin = twin(x, residual=res).squeeze(0)

    node = conv_node()
    node["post_ops"] = {"residual_add": True}
    conv_out = _tilecnn_conv2d(x.squeeze(0), w, b, node, fin, fw, fb, fout)
    out_ref = _tilecnn_residual_add(conv_out, res.squeeze(0), fout - fres,
                                    relu=not relu6, relu6=relu6, out_frac=fout)
    assert torch.equal(out_twin, out_ref)


@pytest.mark.parametrize("fin,fw,fb,fout", FRAC_COMBOS)
def test_linear_twin_matches_reference(fin, fw, fb, fout):
    x = rand_i8(1, 16)
    w = rand_i8(10, 16)
    b = rand_i8(10)
    twin = HardwareLinear(w, b, frac_w=fw, frac_b=fb, frac_din=fin, frac_dout=fout)
    out_twin = twin(x).reshape(-1)
    out_ref = _tilecnn_linear(x.reshape(-1), w, b, fin, fw, fb, fout).reshape(-1)
    assert torch.equal(out_twin, out_ref)


@pytest.mark.parametrize("fin,fout", [(5, 5), (4, 6), (6, 4)])
def test_gap_twin_matches_reference(fin, fout):
    x = rand_i8(1, 8, 7, 7)
    twin = HardwareGAP(frac_in=fin, frac_out=fout)
    out_twin = twin(x).reshape(-1)
    out_ref = _tilecnn_gap(x.squeeze(0), fin, fout).reshape(-1)
    assert torch.equal(out_twin, out_ref)


@pytest.mark.parametrize("shift", [-1, 0, 1])
def test_maxpool_twin_matches_reference(shift):
    x = rand_i8(1, 4, 8, 8)
    twin = HardwareMaxPool2d(3, 2, 1, post_pool_shift=shift)
    out_twin = twin(x).squeeze(0)
    node = {"attrs": {"kernel": [3, 3], "stride": [2, 2], "padding": [1, 1, 1, 1]}}
    out_ref = _tilecnn_maxpool(x.squeeze(0), node, shift)
    assert torch.equal(out_twin, out_ref)


def test_maxpool_reference_reads_attrs():
    """A 2x2/s2/p0 pool must not silently be computed as 3x3/s2/p1."""
    x = rand_i8(1, 2, 8, 8)
    node = {"attrs": {"kernel": [2, 2], "stride": [2, 2], "padding": [0, 0, 0, 0]}}
    out = _tilecnn_maxpool(x.squeeze(0), node, 0)
    assert out.shape == (2, 4, 4)


def test_elementwise_add_matches_reference_when_main_at_fout():
    """The emu add (two aligned inputs) equals the fused reference add when the
    main branch is already at the output frac."""
    fout, fres = 4, 6
    main = rand_i8(1, 4, 5, 5)
    res = rand_i8(1, 4, 5, 5)
    emu = HardwareElementwiseAdd(frac_in1=fout, frac_in2=fres, frac_out=fout)
    out_emu = emu(main, res).squeeze(0)
    out_ref = _tilecnn_residual_add(main.squeeze(0), res.squeeze(0), fout - fres, relu=False)
    assert torch.equal(out_emu, out_ref)
