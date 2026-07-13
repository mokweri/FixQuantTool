"""Cross-layer equalization (CLE) and bias correction for per-tensor
power-of-2 quantization (Nagel et al. 2019, arXiv:1906.04721).

Depthwise layers in MobileNet-class models have per-output-channel weight
ranges spanning ~40x after BN folding, which a single per-tensor scale cannot
represent. CLE rescales adjacent conv pairs channel-by-channel so their ranges
equalize — a pure software transform, mathematically exact through ReLU.

Flow (see equalize_model):
    1. fold BN into convs (the model becomes BN-free),
    2. replace ReLU6 with ReLU (CLE scaling is only exact through unbounded
       ReLU; after QAT the learned thresholds bound the range again, and the
       hardware relu + int8 saturation reproduces the same clamp),
    3. iterate pairwise equalization over conv→relu→conv chains.
"""

import copy
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import fuse

logger = logging.getLogger(__name__)

__all__ = ["equalize_model", "cross_layer_equalize", "replace_relu6_with_relu",
           "apply_bias_correction"]


def replace_relu6_with_relu(model: nn.Module) -> nn.Module:
    for name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            if isinstance(child, nn.ReLU6):
                setattr(mod, child_name, nn.ReLU(inplace=getattr(child, "inplace", False)))
    return model


def _find_equalizable_pairs(gm: fx.GraphModule) -> List[Tuple[str, str]]:
    """Find (conv_a, conv_b) module-target pairs connected directly or through
    a single ReLU, where each intermediate node has exactly one consumer."""
    modules = dict(gm.named_modules())
    pairs = []
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(modules.get(node.target), nn.Conv2d):
            continue
        if len(node.users) != 1:
            continue
        (nxt,) = node.users
        if nxt.op == "call_module" and isinstance(modules.get(nxt.target), nn.ReLU):
            if len(nxt.users) != 1:
                continue
            (nxt,) = nxt.users
        if nxt.op == "call_module" and isinstance(modules.get(nxt.target), nn.Conv2d):
            a, b = modules[node.target], modules[nxt.target]
            a_dw = a.groups == a.in_channels and a.groups > 1
            b_dw = b.groups == b.in_channels and b.groups > 1
            # supported wiring: A's out channels feed B's in channels 1:1
            if (a.groups == 1 or a_dw) and (b.groups == 1 or b_dw) \
                    and b.in_channels == a.out_channels:
                pairs.append((node.target, nxt.target))
    return pairs


@torch.no_grad()
def _equalize_pair(a: nn.Conv2d, b: nn.Conv2d) -> None:
    r1 = a.weight.abs().amax(dim=(1, 2, 3))                 # per out-channel of A
    if b.groups == 1:
        r2 = b.weight.abs().amax(dim=(0, 2, 3))             # per in-channel of B
    else:                                                   # depthwise B
        r2 = b.weight.abs().amax(dim=(1, 2, 3))
    s = torch.sqrt(r1 / r2.clamp_min(1e-12))
    s = torch.where((r1 > 1e-12) & (r2 > 1e-12), s, torch.ones_like(s))
    s = s.clamp(1e-4, 1e4)

    a.weight.data.div_(s.view(-1, 1, 1, 1))
    if a.bias is not None:
        a.bias.data.div_(s)
    if b.groups == 1:
        b.weight.data.mul_(s.view(1, -1, 1, 1))
    else:
        b.weight.data.mul_(s.view(-1, 1, 1, 1))


def cross_layer_equalize(gm: fx.GraphModule, iterations: int = 2) -> fx.GraphModule:
    """Equalize all conv→(relu)→conv pairs in a BN-free model, in place."""
    pairs = _find_equalizable_pairs(gm)
    modules = dict(gm.named_modules())
    logger.info("CLE: equalizing %d conv pairs, %d iteration(s)", len(pairs), iterations)
    for _ in range(iterations):
        for ta, tb in pairs:
            _equalize_pair(modules[ta], modules[tb])
    return gm


def equalize_model(model: nn.Module, iterations: int = 2,
                   replace_relu6: bool = True) -> fx.GraphModule:
    """BN-fold + (optionally) ReLU6→ReLU + CLE. Returns a BN-free GraphModule
    ready for QatProcessor (convs carry the folded bias)."""
    model = copy.deepcopy(model).eval()
    gm = fuse(model)  # folds Conv+BN pairs (torch.fx.experimental.optimization)
    if replace_relu6:
        replace_relu6_with_relu(gm)
    return cross_layer_equalize(gm, iterations=iterations)


@torch.no_grad()
def apply_bias_correction(float_model: nn.Module, qat_model: nn.Module,
                          loader, device, n_batches: int = 8) -> int:
    """One-shot empirical bias correction (per-channel output-mean matching).

    Runs the BN-free float reference and the calibrated QAT model on the same
    batches, and adds the per-channel mean difference of each conv/linear
    output to the QAT module's float bias (which is then quantized as usual).
    Layers are matched by name ('.' vs '_' normalized). Returns #corrected.
    """
    def norm(name):
        return name.replace(".", "_")

    float_means, qat_means = {}, {}
    float_counts, qat_counts = {}, {}

    def mean_hook(store, counts, key):
        def hook(module, inp, out):
            dims = [d for d in range(out.dim()) if d != 1]
            m = out.detach().mean(dim=dims)
            store[key] = store.get(key, 0) + m
            counts[key] = counts.get(key, 0) + 1
        return hook

    f_handles, q_handles = [], []
    float_targets = {}
    for name, mod in float_model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            float_targets[norm(name)] = mod
            f_handles.append(mod.register_forward_hook(
                mean_hook(float_means, float_counts, norm(name))))

    qat_targets = {}
    for name, mod in qat_model.named_modules():
        mod_name = getattr(mod, "module_name", None)
        bias = getattr(mod, "bias", None)
        if bias is None and hasattr(mod, "conv_mod"):
            bias = mod.conv_mod.bias
        if mod_name and bias is not None and norm(mod_name) in float_targets:
            qat_targets[norm(mod_name)] = mod
            q_handles.append(mod.register_forward_hook(
                mean_hook(qat_means, qat_counts, norm(mod_name))))

    try:
        float_model.eval().to(device)
        qat_model.eval().to(device)
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            float_model(images)
            qat_model(images)
    finally:
        for h in f_handles + q_handles:
            h.remove()

    corrected = 0
    for key, mod in qat_targets.items():
        if key not in float_means or key not in qat_means:
            continue
        err = float_means[key] / float_counts[key] - qat_means[key] / qat_counts[key]
        bias = getattr(mod, "bias", None)
        if bias is None and hasattr(mod, "conv_mod"):
            bias = mod.conv_mod.bias
        if bias is not None and bias.shape == err.shape:
            bias.data.add_(err.to(bias.device))
            corrected += 1
    logger.info("Bias correction applied to %d layers over %d batches", corrected, n_batches)
    return corrected
