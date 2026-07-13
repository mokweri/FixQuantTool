"""Golden regression test for the integer kernel chain.

Pure int8 arithmetic with fixed seeded parameters — the committed reference in
tests/golden/ must reproduce bit-exactly forever, across torch versions and
refactors. If this test fails after a kernel change, the hardware contract
changed: either the change is a bug, or the golden file AND the HLS kernels
must be updated together.

Regenerate (only after a deliberate semantics change):
    FIXQUANT_REGEN_GOLDEN=1 python -m pytest tests/test_golden.py
"""

import os
from pathlib import Path

import numpy as np
import torch

from fixquant.emulation.fxp_emu_modules import (
    HardwareConv2d, HardwareLinear, HardwareGAP, HardwareMaxPool2d,
    HardwareElementwiseAdd,
)

GOLDEN = Path(__file__).resolve().parent / "golden" / "kernel_chain_ref.npy"


def _run_chain() -> np.ndarray:
    g = torch.Generator().manual_seed(1234)

    def rand_i8(*shape):
        return torch.randint(-128, 128, shape, dtype=torch.int8, generator=g)

    x = rand_i8(1, 8, 16, 16)

    stem = HardwareConv2d(rand_i8(16, 8, 3, 3), rand_i8(16), stride=2, padding=1,
                          frac_w=7, frac_b=6, frac_din=5, frac_dout=4, relu=True)
    dw = HardwareConv2d(rand_i8(16, 1, 3, 3), rand_i8(16), stride=1, padding=1,
                        groups=16, frac_w=6, frac_b=6, frac_din=4, frac_dout=4,
                        relu6=True)
    pw = HardwareConv2d(rand_i8(16, 16, 1, 1), rand_i8(16), stride=1, padding=0,
                        frac_w=7, frac_b=5, frac_din=4, frac_dout=5)
    add = HardwareElementwiseAdd(frac_in1=5, frac_in2=4, frac_out=5)
    pool = HardwareMaxPool2d(3, 2, 1, post_pool_shift=-1)
    gap = HardwareGAP(frac_in=4, frac_out=5)
    fc = HardwareLinear(rand_i8(10, 16), rand_i8(10),
                        frac_w=7, frac_b=6, frac_din=5, frac_dout=3)

    y = stem(x)
    z = pw(dw(y))
    y = add(z, y)
    y = pool(y)
    y = gap(y)
    y = fc(y.reshape(1, -1))
    return y.numpy()


def test_kernel_chain_matches_golden():
    result = _run_chain()
    if os.environ.get("FIXQUANT_REGEN_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.save(GOLDEN, result)
        raise AssertionError(f"Golden reference (re)generated at {GOLDEN}; rerun the test.")
    expected = np.load(GOLDEN)
    np.testing.assert_array_equal(result, expected)
