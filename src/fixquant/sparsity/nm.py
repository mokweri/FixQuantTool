"""Fixed N:M weight masks for quantization-aware training.

Patterns (all prune half of each layer's weights):

  filter24      2:4 along input channels, chosen independently per output row
  pair24        2:4 along input channels, one choice shared by two consecutive
                output rows (keeps two weights per DSP packable)
  oc_n16        8:16 along output channels at each (input channel, tap)
  <ic>+perm     filter24 or pair24 after an input-channel permutation

The mask arithmetic is that of the DeepTile gate-5 sensitivity screen, so the
masks of `filter24`, `pair24` and `oc_n16` are identical to the ones it
measured. `+perm` differs on purpose: the screen searched one permutation per
layer, which cannot be built when a tensor feeds several layers (a residual
stream) or joins another at an add. Here one permutation is searched per
*channel class*, the set of tensors that must share a channel order (joined by
adds, or passed through BatchNorm, activations, pooling and depthwise
convolutions), against every pruned layer that reads it. The same
permutation also orders the output rows of the layers that write the class,
so `pair24` pairs rows in that physical order. A permutation is therefore a
real reorder of the network's channels, and the export has to apply it.

Masks are computed on the float model, before BatchNorm folding and
cross-layer equalization. Both scale weights per channel, which keeps zeros
at zero, so the pattern survives into the quantized and deployed weights.
"""
from __future__ import annotations

import operator
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

SCHEMA_VERSION = 1
BASE_PATTERNS = ("filter24", "pair24", "oc_n16")
IC_ROWS = {"filter24": 1, "pair24": 2}   # output rows sharing one 2:4 choice
OC_N, OC_M = 8, 16


# ----------------------------------------------------------------- patterns
def parse_pattern(spec: str) -> Tuple[str, bool]:
    base, sep, flag = spec.partition("+")
    if base not in BASE_PATTERNS or (sep and flag != "perm") or (base == "oc_n16" and sep):
        raise ValueError(f"unknown N:M pattern {spec!r}; expected one of "
                         "filter24, pair24, oc_n16, filter24+perm, pair24+perm")
    return base, bool(sep)


def as4d(w: torch.Tensor) -> torch.Tensor:
    return w if w.dim() == 4 else w[:, :, None, None]


def pad_rows(a: torch.Tensor, mult: int) -> Tuple[torch.Tensor, int]:
    o = a.shape[0]
    p = (-o) % mult
    if p:
        a = torch.cat([a, a.new_zeros((p,) + a.shape[1:])])
    return a, o


def ic_mask(absw: torch.Tensor, rows: int) -> torch.Tensor:
    """2:4 along input channels, one choice shared by `rows` consecutive
    outputs. absw [O,I,kh,kw] in physical order; returns a bool mask."""
    a, o = pad_rows(absw, rows)
    O, I, kh, kw = a.shape
    agg = a.view(O // rows, rows, I // 4, 4, kh, kw).sum(1)
    keep = agg.topk(2, dim=2).indices
    m = torch.zeros_like(agg, dtype=torch.bool).scatter_(2, keep, True)
    return m[:, None].expand(O // rows, rows, I // 4, 4, kh, kw).reshape(O, I, kh, kw)[:o]


def oc_mask(absw: torch.Tensor, n: int = OC_N, m: int = OC_M) -> torch.Tensor:
    """n:m along output channels at each (input channel, tap)."""
    a, o = pad_rows(absw, m)
    O, I, kh, kw = a.shape
    g = a.view(O // m, m, I, kh, kw)
    keep = g.topk(n, dim=1).indices
    mk = torch.zeros_like(g, dtype=torch.bool).scatter_(1, keep, True)
    return mk.reshape(O, I, kh, kw)[:o]


def ic_rows_matrix(absw: torch.Tensor, rows: int) -> torch.Tensor:
    """Magnitudes a 2:4 choice along input channels is made on: [R, I], one
    row per (output-row group, tap)."""
    a, _ = pad_rows(absw, rows)
    O, I, kh, kw = a.shape
    agg = a.view(O // rows, rows, I, kh, kw).sum(1)
    return agg.permute(0, 2, 3, 1).reshape(-1, I)


def search_perm(A: torch.Tensor, iters: int = 300, cand: int = 1024,
                seed: int = 0) -> Optional[torch.Tensor]:
    """Greedy input-channel swap search maximising the magnitude a 2:4 choice
    keeps. A [R, I]; returns the new order (physical channel j holds channel
    perm[j]), or None when there is nothing to permute."""
    R, I = A.shape
    G = I // 4
    if G < 2:
        return None
    dev = A.device
    perm = torch.arange(I, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def retained(gc):                                   # [B,4] -> [B]
        return A[:, gc].topk(2, dim=2).values.sum(dim=(0, 2))

    p2 = perm.view(G, 4)
    cur = retained(p2)
    for _ in range(iters):
        ga = torch.randint(0, G, (cand,), generator=gen)
        gb = torch.randint(0, G, (cand,), generator=gen)
        ok = ga != gb
        ga, gb = ga[ok].to(dev), gb[ok].to(dev)
        sa = torch.randint(0, 4, (len(ga),), generator=gen).to(dev)
        sb = torch.randint(0, 4, (len(ga),), generator=gen).to(dev)
        pa, pb = p2[ga].clone(), p2[gb].clone()
        ar = torch.arange(len(ga), device=dev)
        ca, cb = pa[ar, sa].clone(), pb[ar, sb].clone()
        pa[ar, sa], pb[ar, sb] = cb, ca
        delta = retained(pa) + retained(pb) - cur[ga] - cur[gb]
        touched, accepted = set(), 0
        for k in torch.argsort(delta, descending=True).tolist():
            if delta[k] <= 0:
                break
            i, j = ga[k].item(), gb[k].item()
            if i in touched or j in touched:
                continue
            touched.update((i, j))
            p2[i], p2[j] = pa[k], pb[k]
            accepted += 1
        if accepted == 0:
            break
        cur = retained(p2)
    return perm


# ----------------------------------------------------------------- network
def prunable(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Every non-grouped convolution except one reading the 3-channel image,
    and every linear layer (the screen's set)."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.in_channels > 3:
            out.append((name, m))
        elif isinstance(m, nn.Linear):
            out.append((name, m))
    return out


_PASS_MODULES = (nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.Hardtanh, nn.SiLU, nn.Hardswish,
                 nn.Sigmoid, nn.Dropout, nn.Identity, nn.MaxPool2d, nn.AvgPool2d,
                 nn.AdaptiveAvgPool2d, nn.Flatten)
_PASS_FUNCS = {F.relu, F.relu6, torch.relu, F.hardtanh, F.silu, F.dropout,
               F.adaptive_avg_pool2d, F.avg_pool2d, F.max_pool2d, torch.flatten}
_JOIN_FUNCS = {operator.add, operator.iadd, torch.add}
_PASS_METHODS = {"relu", "relu_", "flatten", "contiguous", "clamp", "clamp_"}


class _UF:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, a: str) -> str:
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _channels(node: fx.Node) -> Optional[int]:
    tm = node.meta.get("tensor_meta")
    return None if tm is None or len(tm.shape) < 2 else int(tm.shape[1])


def channel_classes(model: nn.Module, image_size: int = 224, with_members: bool = False):
    """Trace the float model and group tensors that must share a channel
    order. Returns ({layer: (in_class, out_class)} for every non-grouped
    convolution and linear layer, the set of fixed classes, {class: channels})
    and, with `with_members`, {module: class} for the BatchNorms and depthwise
    convolutions whose channels follow a class."""
    gm = fx.symbolic_trace(model)
    was_training = model.training
    gm.eval()
    dev = next(model.parameters()).device
    ShapeProp(gm).propagate(torch.zeros(1, 3, image_size, image_size, device=dev))
    model.train(was_training)
    mods = dict(gm.named_modules())
    uf = _UF()
    layers: Dict[str, Tuple[str, str]] = {}
    channelwise: List[Tuple[str, str]] = []    # (module, node) whose channels follow the node
    fixed: List[str] = []

    def tensor_args(n):
        return [a for a in n.args if isinstance(a, fx.Node) and _channels(a) is not None]

    for n in gm.graph.nodes:
        uf.find(n.name)
        if n.op == "placeholder":
            fixed.append(n.name)
        elif n.op == "output":
            outs = n.args[0] if isinstance(n.args[0], (tuple, list)) else [n.args[0]]
            fixed.extend(a.name for a in outs if isinstance(a, fx.Node))
        elif n.op == "get_attr":
            raise NotImplementedError(f"channel classes: get_attr node {n.name}")
        elif n.op == "call_module":
            m = mods[n.target]
            src = n.args[0]
            if isinstance(m, nn.Conv2d) and m.groups == 1 or isinstance(m, nn.Linear):
                if isinstance(m, nn.Linear) and src.meta["tensor_meta"].shape[1] != m.in_features:
                    raise NotImplementedError(f"{n.target}: linear input is not [N, C]")
                layers[n.target] = (src.name, n.name)
            elif isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels:
                uf.union(n.name, src.name)            # depthwise keeps the order
                channelwise.append((n.target, n.name))
            elif isinstance(m, nn.Conv2d):
                raise NotImplementedError(f"{n.target}: grouped convolution")
            elif isinstance(m, _PASS_MODULES):
                if isinstance(m, nn.Flatten):
                    _check_flatten(src, n)
                if isinstance(m, nn.BatchNorm2d):
                    channelwise.append((n.target, n.name))
                uf.union(n.name, src.name)
            else:
                raise NotImplementedError(f"channel classes: module {type(m).__name__} ({n.target})")
        elif n.op in ("call_function", "call_method"):
            ok_join = n.op == "call_function" and n.target in _JOIN_FUNCS
            ok_pass = (n.op == "call_function" and n.target in _PASS_FUNCS) or \
                      (n.op == "call_method" and n.target in _PASS_METHODS)
            if not (ok_join or ok_pass):
                raise NotImplementedError(f"channel classes: {n.op} {n.target} ({n.name})")
            ts = tensor_args(n)
            if ok_join:
                if len({_channels(a) for a in ts} | {_channels(n)}) != 1:
                    raise NotImplementedError(f"{n.name}: join of different channel counts")
            elif n.target in (torch.flatten, "flatten"):
                _check_flatten(ts[0], n)
            for a in ts:
                uf.union(n.name, a.name)
    # Resolve every node to its class representative.
    resolved = {k: (uf.find(a), uf.find(b)) for k, (a, b) in layers.items()}
    fixed_classes = {uf.find(f) for f in fixed}
    width = {}
    for n in gm.graph.nodes:
        c = _channels(n)
        if c is not None:
            width.setdefault(uf.find(n.name), c)
    if with_members:
        members = {k: uf.find(v) for k, v in channelwise}
        return resolved, fixed_classes, width, members
    return resolved, fixed_classes, width


def _check_flatten(src: fx.Node, n: fx.Node) -> None:
    shape = src.meta["tensor_meta"].shape
    if len(shape) == 4 and (shape[2] != 1 or shape[3] != 1):
        raise NotImplementedError(f"{n.name}: flatten of a {list(shape)} tensor mixes channels")


# ----------------------------------------------------------------- masks
def _to_physical(a: torch.Tensor, p_in, p_out) -> torch.Tensor:
    if p_out is not None:
        a = a[p_out]
    if p_in is not None:
        a = a[:, p_in]
    return a


def _to_original(m: torch.Tensor, p_in, p_out) -> torch.Tensor:
    if p_in is not None:
        t = torch.empty_like(m)
        t[:, p_in] = m
        m = t
    if p_out is not None:
        t = torch.empty_like(m)
        t[p_out] = m
        m = t
    return m


def pattern_mask(absw4: torch.Tensor, base: str, p_in=None, p_out=None) -> torch.Tensor:
    a = _to_physical(absw4, p_in, p_out)
    m = oc_mask(a) if base == "oc_n16" else ic_mask(a, IC_ROWS[base])
    return _to_original(m, p_in, p_out)


def check_pattern(mask4: torch.Tensor, base: str, p_in=None, p_out=None) -> Optional[str]:
    """None if `mask4` (original order) is the pattern in physical order,
    else a description of the first violation."""
    m = _to_physical(mask4.bool(), p_in, p_out)
    if base == "oc_n16":
        a, _ = pad_rows(m, OC_M)
        O, I, kh, kw = a.shape
        cnt = a.view(O // OC_M, OC_M, I, kh, kw).sum(1)
        if (cnt > OC_N).any():
            return f"more than {OC_N} of {OC_M} output channels kept"
        return None
    rows = IC_ROWS[base]
    a, _ = pad_rows(m, rows)
    O, I, kh, kw = a.shape
    if I % 4:
        return "input channels not a multiple of 4"
    g = a.view(O // rows, rows, I // 4, 4, kh, kw)
    if rows > 1:
        real = torch.ones(O, dtype=torch.bool, device=m.device)
        real[m.shape[0]:] = False
        r = real.view(O // rows, rows)
        full = r.all(1)
        if not (g[full] == g[full][:, :1]).all():
            return f"rows of a {rows}-row group differ"
    if (g.sum(3) > 2).any():
        return "more than 2 of 4 input channels kept"
    return None


def build_nm(model: nn.Module, pattern: str, layers: Optional[Sequence[str]] = None,
             perm_iters: int = 300, perm_seed: int = 0, image_size: int = 224,
             device: Optional[torch.device] = None) -> dict:
    """Masks (and permutations) for `pattern` on the float `model`.

    `layers` selects the pruned layers (default: every prunable layer); the
    others stay dense. Returns a self-describing artifact for `save`."""
    base, perm_on = parse_pattern(pattern)
    candidates = dict(prunable(model))
    names = list(candidates) if layers is None else list(layers)
    unknown = [n for n in names if n not in candidates]
    if unknown:
        raise KeyError(f"not prunable layers: {unknown}")
    bad = [n for n in names if as4d(candidates[n].weight).shape[1] % 4]
    if bad:
        raise ValueError(f"input channels not a multiple of 4: {bad}")
    classes, fixed, width = channel_classes(model, image_size)
    dev = device or next(model.parameters()).device
    absw = {n: as4d(candidates[n].weight.detach().abs().float()).to(dev) for n in names}

    perms: Dict[str, torch.Tensor] = {}
    perm_gain: Dict[str, Tuple[float, float]] = {}
    if perm_on:
        rows = IC_ROWS[base]
        readers: Dict[str, List[str]] = {}
        for n in names:
            readers.setdefault(classes[n][0], []).append(n)
        for c in sorted(readers):
            if c in fixed:
                continue
            A = torch.cat([ic_rows_matrix(absw[n], rows) for n in readers[c]])
            before = A.view(A.shape[0], -1, 4).topk(2, dim=2).values.sum().item()
            p = search_perm(A, iters=perm_iters, seed=perm_seed)
            if p is None:
                continue
            after = A[:, p].view(A.shape[0], -1, 4).topk(2, dim=2).values.sum().item()
            if after > before:
                perms[c] = p.cpu()
                perm_gain[c] = (before / A.sum().item(), after / A.sum().item())

    masks, report = {}, []
    for n in names:
        cin, cout = classes[n]
        p_in = perms.get(cin)
        p_out = perms.get(cout)
        p_in = p_in.to(dev) if p_in is not None else None
        p_out = p_out.to(dev) if p_out is not None else None
        mk = pattern_mask(absw[n], base, p_in, p_out)
        err = check_pattern(mk, base, p_in, p_out)
        if err:
            raise AssertionError(f"{n}: built mask violates {pattern}: {err}")
        w = candidates[n].weight
        masks[n] = mk.view(w.shape).cpu()
        l1 = absw[n].sum().item()
        report.append({"layer": n, "shape": list(w.shape), "in_class": cin, "out_class": cout,
                       "kept_fraction": mk.float().mean().item(),
                       "retained_l1": (absw[n] * mk).sum().item() / l1})
    return {
        "schema_version": SCHEMA_VERSION,
        "pattern": pattern,
        "base": base,
        "layers": names,
        "masks": masks,
        "perms": perms,
        "perm_retained": {c: list(v) for c, v in perm_gain.items()},
        "layer_classes": {n: list(classes[n]) for n in names},
        "class_width": {c: width[c] for c in {x for n in names for x in classes[n]}},
        "perm_iters": perm_iters,
        "perm_seed": perm_seed,
        "report": report,
    }


def save(art: dict, path) -> None:
    torch.save(art, path)


def load(path) -> dict:
    art = torch.load(path, map_location="cpu", weights_only=False)
    if art.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported N:M artifact schema {art.get('schema_version')}")
    return art


@torch.no_grad()
def apply_float(model: nn.Module, art: dict) -> None:
    """Zero the masked weights of the float model in place."""
    mods = dict(model.named_modules())
    for n, mk in art["masks"].items():
        w = mods[n].weight
        w.mul_(mk.to(w.device, w.dtype))


def _weight(module: nn.Module) -> torch.Tensor:
    inner = getattr(module, "conv_mod", None)      # FusedConvBN keeps the conv inside
    return inner.weight if inner is not None else module.weight


class Masker:
    """Keeps the masked weights of a (quantized) model at zero.

    Layers are found by name, which FixQuant's passes keep. Construction
    checks that every masked weight is already zero, so a wrong mapping fails
    here instead of silently training a different network."""

    def __init__(self, model: nn.Module, art: dict):
        mods = dict(model.named_modules())
        self.items = []
        for n, mk in art["masks"].items():
            if n not in mods:
                raise KeyError(f"masked layer {n} not found in the model")
            w = _weight(mods[n])
            if tuple(w.shape) != tuple(mk.shape):
                raise ValueError(f"{n}: mask {tuple(mk.shape)} does not fit weight {tuple(w.shape)}")
            if (w.detach()[~mk.to(w.device)] != 0).any():
                raise ValueError(f"{n}: masked weights are not zero; was the float model pruned?")
            self.items.append((mods[n], mk))

    @torch.no_grad()
    def apply(self) -> None:
        for i, (mod, mk) in enumerate(self.items):
            w = _weight(mod)
            if mk.device != w.device:
                mk = mk.to(w.device)
                self.items[i] = (mod, mk)
            w.mul_(mk.to(w.dtype))

    def attach(self, optimizer: torch.optim.Optimizer):
        """Re-apply the masks after every optimizer step."""
        return optimizer.register_step_post_hook(lambda opt, args, kwargs: self.apply())


@torch.no_grad()
def permute_channels(model: nn.Module, perms: Dict[str, torch.Tensor], image_size: int = 224) -> None:
    """Reorder the float model's channels in place so that every class with a
    permutation is stored in its physical order. The network computes the
    same function; the pruned layers' weights then show the pattern with no
    permutation. This is the reorder an export has to apply."""
    layers, fixed, _, members = channel_classes(model, image_size, with_members=True)
    if set(perms) & fixed:
        raise ValueError("a fixed class (network input or output) cannot be permuted")
    mods = dict(model.named_modules())

    def take(t, p, dim):
        if t is not None:
            t.data = t.data.index_select(dim, p.to(t.device)).contiguous()

    for name, (cin, cout) in layers.items():
        m = mods[name]
        if cout in perms:
            take(m.weight, perms[cout], 0)
            take(m.bias, perms[cout], 0)
        if cin in perms:
            take(m.weight, perms[cin], 1)
    for name, c in members.items():
        if c not in perms:
            continue
        m = mods[name]
        p = perms[c]
        take(m.weight, p, 0)
        take(getattr(m, "bias", None), p, 0)
        if isinstance(m, nn.BatchNorm2d):
            take(m.running_mean, p, 0)
            take(m.running_var, p, 0)


def find_weight(state: Dict[str, torch.Tensor], layer: str) -> torch.Tensor:
    for k in (f"{layer}.conv_mod.weight", f"{layer}.weight"):
        if k in state:
            return state[k]
    raise KeyError(f"no weight for {layer} in the checkpoint")


def verify_state(state: Dict[str, torch.Tensor], art: dict) -> dict:
    """Check a checkpoint's weights against the artifact: every masked weight
    zero and the mask the pattern in physical order. Returns a summary."""
    base = art["base"]
    perms = art["perms"]
    bad, nz, tot = [], 0, 0
    for n in art["layers"]:
        w = find_weight(state, n).detach().cpu()
        mk = art["masks"][n].bool()
        cin, cout = art["layer_classes"][n]
        err = check_pattern(as4d(mk), base, perms.get(cin), perms.get(cout))
        if err:
            bad.append(f"{n}: {err}")
        if (w[~mk] != 0).any():
            bad.append(f"{n}: {int((w[~mk] != 0).sum())} masked weights are nonzero")
        nz += int((w != 0).sum())
        tot += w.numel()
    return {"pattern": art["pattern"], "layers": len(art["layers"]), "ok": not bad,
            "violations": bad, "nonzero_fraction": nz / max(tot, 1)}


# ------------------------------------------------------- training scripts
def prepare(model: nn.Module, pattern: Optional[str] = None, layers_file=None,
            masks_path=None, perm_iters: int = 300, perm_seed: int = 0,
            image_size: int = 224, device=None, save_dir=None) -> Optional[dict]:
    """The N:M step shared by the float and QAT training scripts: choose the
    masks from the float model's weights (`pattern`, optionally restricted to
    the layers in `layers_file`) or take those of an earlier run
    (`masks_path`), zero the masked weights, and write `nm_masks.pt` and
    `nm_report.json` to `save_dir`. Returns the artifact, or None when no
    sparsity is asked for.

    Taking the masks over is what chains a float fine-tune into QAT: choosing
    them again from the fine-tuned weights could pick a different
    permutation, and with it a different network."""
    import json
    import os
    if masks_path:
        art = load(masks_path)
        if pattern and pattern != art["pattern"]:
            raise ValueError(f"--nm_pattern {pattern} differs from the masks' {art['pattern']}")
        if layers_file:
            raise ValueError("--nm_layers cannot be combined with --nm_masks")
    elif pattern:
        layers = None
        if layers_file:
            with open(layers_file) as f:
                spec = json.load(f)
            layers = spec["layers"] if isinstance(spec, dict) else spec
        art = build_nm(model, pattern, layers, perm_iters=perm_iters, perm_seed=perm_seed,
                       image_size=image_size, device=device)
    else:
        return None
    apply_float(model, art)
    if save_dir is not None:
        save(art, os.path.join(save_dir, "nm_masks.pt"))
        with open(os.path.join(save_dir, "nm_report.json"), "w") as f:
            json.dump({k: art[k] for k in ("pattern", "layers", "perm_retained", "perm_iters",
                                           "perm_seed", "report")}, f, indent=1)
    print(f"N:M {art['pattern']}: {len(art['layers'])} layers pruned, "
          f"{len(art['perms'])} channel classes permuted"
          + (f" (masks from {masks_path})" if masks_path else ""))
    return art


def verify_checkpoint(path, art: dict) -> dict:
    """verify_state on a RunManager checkpoint; raises if the check fails."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    check = verify_state(state, art)
    if not check["ok"]:
        raise RuntimeError(f"N:M check failed on {path}: {check['violations'][:5]}")
    print(f"N:M check passed on {path}: nonzero fraction "
          f"{check['nonzero_fraction']:.4f} of the pruned layers")
    return check
