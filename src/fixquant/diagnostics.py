"""Per-layer quantization diagnostics.

Turns "accuracy dropped" into "layer X clipped 30%": reports thresholds and
frac bits per quantizer, weight SQNR/clip rates, per-epoch threshold logging
during QAT, and a per-layer parity sweep between the QAT model and a hardware
(int8) model.
"""

import csv
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fixquant.quantization.tqt_quantizer import TQTQuantizer

__all__ = [
    "quantizer_report",
    "write_report_csv",
    "log_quantizer_state",
    "parity_sweep",
]


def _pow2_quantize(x: torch.Tensor, frac: int, bitwidth: int = 8) -> torch.Tensor:
    scale = 2.0 ** frac
    max_v = (1 << (bitwidth - 1)) - 1
    return torch.clamp(torch.floor(x * scale + 0.5), -max_v - 1, max_v) / scale


def weight_sqnr_db(w: torch.Tensor, frac: int, bitwidth: int = 8) -> float:
    """SQNR (dB) of a tensor quantized at the given fractional position.

    Returns -inf for an all-zero tensor (no signal), e.g. the zero bias
    Parameter that FusedConvBN creates for bias-free convs.
    """
    wq = _pow2_quantize(w, frac, bitwidth)
    p = (w ** 2).mean()
    if p.item() <= 0.0:
        return float("-inf")
    n = ((w - wq) ** 2).mean().clamp_min(1e-20)
    return 10.0 * math.log10((p / n).item())


def clip_rate(w: torch.Tensor, frac: int, bitwidth: int = 8) -> float:
    """Fraction of elements saturated at the given fractional position."""
    scale = 2.0 ** frac
    max_v = (1 << (bitwidth - 1)) - 1
    scaled = torch.floor(w * scale + 0.5)
    return ((scaled > max_v) | (scaled < -max_v - 1)).float().mean().item()


@torch.no_grad()
def quantizer_report(model: nn.Module) -> List[Dict]:
    """One row per TQTQuantizer: name, type, log2 threshold, exported frac.

    For weight/bias quantizers whose owner tensor is reachable, also reports
    SQNR and clip rate at the current frac.
    """
    # Map each quantizer to the tensor it quantizes, where statically known.
    owner_tensors = {}
    for mod_name, mod in model.named_modules():
        w_q = getattr(mod, "weight_quantizer", None)
        b_q = getattr(mod, "bias_quantizer", None)
        if isinstance(w_q, TQTQuantizer):
            w = getattr(mod, "weight", None)
            if w is None and hasattr(mod, "conv_mod"):
                w = mod.conv_mod.weight
            owner_tensors[id(w_q)] = w
        if isinstance(b_q, TQTQuantizer):
            b = getattr(mod, "bias", None)
            if b is None and hasattr(mod, "conv_mod"):
                b = mod.conv_mod.bias
            owner_tensors[id(b_q)] = b

    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, TQTQuantizer):
            continue
        bitwidth, frac = mod.export_quant_info()
        row = {
            "quantizer": name,
            "tensor_type": mod.tensor_type,
            "bitwidth": bitwidth,
            "log2_t": round(float(mod.log_threshold.item()), 4),
            "frac": frac,
            "bounded_range": str(mod.bounded_range) if mod.bounded_range else "",
            "sqnr_db": "",
            "clip_rate": "",
        }
        t = owner_tensors.get(id(mod))
        if t is not None and t.numel() > 0:
            t = t.detach().float()
            sqnr = weight_sqnr_db(t, frac, bitwidth)
            row["sqnr_db"] = round(sqnr, 2) if math.isfinite(sqnr) else "zero-tensor"
            row["clip_rate"] = round(clip_rate(t, frac, bitwidth), 5)
        rows.append(row)
    return rows


def write_report_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def log_quantizer_state(model: nn.Module, epoch: int, path: str) -> None:
    """Append per-epoch threshold/frac state to a CSV (created on first call)."""
    rows = quantizer_report(model)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fieldnames = ["epoch"] + list(rows[0].keys()) if rows else ["epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({"epoch": epoch, **row})


@torch.no_grad()
def parity_sweep(qat_model: nn.Module, hw_model: nn.Module,
                 input_tensor: torch.Tensor,
                 max_report: Optional[int] = None) -> List[Dict]:
    """Compare per-layer int8 activations of the QAT model against a hardware
    (int8) model on the same input. Layers are matched by module_name.

    Returns one row per matched layer: element count, mismatch count, and the
    max absolute int8 difference. Diffs of ±1 are expected (fake-quant vs the
    hardware two-step rounding); anything larger indicates a real mismatch.
    """
    qat_acts: Dict[str, torch.Tensor] = {}
    hw_acts: Dict[str, torch.Tensor] = {}

    def qat_hook(mod_name, frac):
        def hook(module, inp, out):
            scale = 2.0 ** frac
            qat_acts[mod_name] = torch.round(out.detach() * scale).clamp(-128, 127).to(torch.int16)
        return hook

    def hw_hook(mod_name):
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if o.dtype == torch.int8:
                hw_acts[mod_name] = o.detach().to(torch.int16)
        return hook

    handles = []
    for name, mod in qat_model.named_modules():
        mod_name = getattr(mod, "module_name", None)
        q = getattr(mod, "act_quantizer", None) or getattr(mod, "quantizer", None)
        if mod_name and isinstance(q, TQTQuantizer):
            frac = q.export_quant_info()[1]
            handles.append(mod.register_forward_hook(qat_hook(mod_name, frac)))
    for name, mod in hw_model.named_modules():
        mod_name = getattr(mod, "module_name", None)
        if mod_name:
            handles.append(mod.register_forward_hook(hw_hook(mod_name)))

    try:
        qat_model.eval()
        hw_model.eval()
        qat_model(input_tensor)
        hw_model(input_tensor)
    finally:
        for h in handles:
            h.remove()

    rows = []
    for mod_name, q_act in qat_acts.items():
        if mod_name not in hw_acts:
            continue
        h_act = hw_acts[mod_name]
        if q_act.shape != h_act.shape:
            rows.append({"layer": mod_name, "elements": q_act.numel(),
                         "mismatches": -1, "max_abs_diff": -1,
                         "note": f"shape mismatch {tuple(q_act.shape)} vs {tuple(h_act.shape)}"})
            continue
        diff = (q_act - h_act).abs()
        rows.append({
            "layer": mod_name,
            "elements": int(diff.numel()),
            "mismatches": int((diff > 0).sum()),
            "max_abs_diff": int(diff.max()),
            "note": "",
        })
        if max_report is not None and len(rows) >= max_report:
            break
    return rows
