"""Export a QAT model as plain floating-point PyTorch.

The exported model has batch normalization folded into the convolutions
(``FusedConvBN.freeze``) and every quantizer removed. It is written as
standalone source that imports only ``torch`` plus a plain ``state_dict``, so
another toolchain (Vitis AI, ONNX export, ...) can load it without FixQuant.

Two choices of weight values:

* ``deployed`` (default): each convolution's and linear layer's weight and
  bias after its own quantizer, i.e. the values FixQuant deploys, held in
  float. QAT trains the network around these values, so this float model keeps
  the QAT model's accuracy, and because the values lie on power-of-two grids, a
  power-of-two INT8 quantizer can represent them exactly.
* ``latent``: the float parameters QAT updates, before weight quantization.
  For the released models this float model is markedly less accurate than the
  QAT model (MobileNetV2 61.8% against 70.7% top-1 on the evaluated subset), so
  it is kept for analysis rather than deployment.

Conversion is by an explicit whitelist. A FixQuant module the exporter does
not know stops the export rather than being carried over, so a new QAT module
type cannot silently change what is exported.
"""

from __future__ import annotations

import contextlib
import copy
import operator
from typing import Dict, Iterator, List, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from fixquant.quantization.fused_conv_bn import FusedConvBN
from fixquant.quantization.qat_modules import (
    QAdaptiveAvgPool2d,
    QElementwiseAdd,
    QMaxPool2D,
    QuantizedConv2d,
    QuantizedLinear,
    QuantStubC,
)
from fixquant.quantization.tqt_quantizer import FakeQuantizer

FORMAT_VERSION = "fixquant.float_export.v1"

# Standard modules that carry over unchanged, with how to construct each in
# the emitted source.
_PLAIN_MODULES = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                  nn.AvgPool2d, nn.ReLU, nn.ReLU6, nn.Flatten, nn.Dropout,
                  nn.Identity)


class FloatExportError(RuntimeError):
    """The model contains something the float exporter cannot represent."""


WEIGHT_MODES = ("deployed", "latent")


@torch.no_grad()
def _values(weight: torch.Tensor, bias, quantized_by: nn.Module, mode: str):
    """Weight and bias as exported: through the module's quantizers or as is."""
    if mode == "latent":
        return weight, bias
    q_bias = quantized_by.bias_quantizer(bias) if bias is not None else None
    return quantized_by.weight_quantizer(weight), q_bias


def _plain_conv(conv: nn.Conv2d, quantized_by: nn.Module, mode: str) -> nn.Conv2d:
    if conv.padding_mode != "zeros":
        raise FloatExportError(f"unsupported padding mode {conv.padding_mode!r}")
    if conv.bias is None:
        raise FloatExportError("a frozen convolution must carry its folded bias")
    out = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=True)
    weight, bias = _values(conv.weight, conv.bias, quantized_by, mode)
    with torch.no_grad():
        out.weight.copy_(weight)
        out.bias.copy_(bias)
    return out


def _plain_linear(linear: nn.Linear, mode: str) -> nn.Linear:
    out = nn.Linear(linear.in_features, linear.out_features,
                    bias=linear.bias is not None)
    weight, bias = _values(linear.weight, linear.bias, linear, mode)
    with torch.no_grad():
        out.weight.copy_(weight)
        if bias is not None:
            out.bias.copy_(bias)
    return out


def _set_submodule(root: nn.Module, target: str, module: nn.Module) -> None:
    parent_path, _, name = target.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, name, module)


def to_float_graph(qat_model: fx.GraphModule, weights: str = "deployed"
                   ) -> Tuple[fx.GraphModule, Dict[str, int]]:
    """Convert a frozen FixQuant QAT graph into plain PyTorch modules.

    ``weights`` selects the weight values (see the module docstring). Returns
    the converted graph and a count of each conversion applied.
    """
    if weights not in WEIGHT_MODES:
        raise FloatExportError(f"weights must be one of {WEIGHT_MODES}")
    gm = copy.deepcopy(qat_model).eval()
    modules = dict(gm.named_modules())
    counts: Dict[str, int] = {}

    def note(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    for node in list(gm.graph.nodes):
        if node.op == "call_module":
            module = modules[node.target]
            if isinstance(module, FusedConvBN):
                if not module.frozen:
                    raise FloatExportError(
                        f"{node.target}: batch norm is not folded; freeze the model first")
                _set_submodule(gm, node.target, _plain_conv(module.conv_mod, module, weights))
                note("FusedConvBN->Conv2d")
            elif isinstance(module, QuantizedConv2d):
                _set_submodule(gm, node.target, _plain_conv(module, module, weights))
                note("QuantizedConv2d->Conv2d")
            elif isinstance(module, QuantizedLinear):
                _set_submodule(gm, node.target, _plain_linear(module, weights))
                note("QuantizedLinear->Linear")
            elif isinstance(module, QMaxPool2D):
                _set_submodule(gm, node.target, nn.MaxPool2d(
                    module.kernel_size, module.stride, module.padding,
                    module.dilation, module.return_indices, module.ceil_mode))
                note("QMaxPool2D->MaxPool2d")
            elif isinstance(module, QAdaptiveAvgPool2d):
                _set_submodule(gm, node.target, nn.AdaptiveAvgPool2d(module.output_size))
                note("QAdaptiveAvgPool2d->AdaptiveAvgPool2d")
            elif isinstance(module, QElementwiseAdd):
                with gm.graph.inserting_after(node):
                    add = gm.graph.call_function(operator.add, args=node.args)
                node.replace_all_uses_with(add)
                old_name = node.name
                gm.graph.erase_node(node)
                add.name = old_name  # keep the node name the quantization record uses
                note("QElementwiseAdd->add")
            elif isinstance(module, (QuantStubC, FakeQuantizer)):
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                note("quantizer removed")
            elif type(module) in _PLAIN_MODULES:
                note(f"{type(module).__name__} kept")
            else:
                raise FloatExportError(
                    f"{node.target}: no float conversion for {type(module).__qualname__}")
        elif node.op == "call_function":
            origin = getattr(node.target, "__module__", "") or ""
            if origin.startswith("fixquant"):
                raise FloatExportError(f"{node.name}: FixQuant function {node.target}")
    gm.graph.lint()
    gm.recompile()
    gm.delete_all_unused_submodules()

    for name, module in gm.named_modules():
        if type(module).__module__.startswith("fixquant"):
            raise FloatExportError(f"{name}: {type(module).__qualname__} survived conversion")
    return gm, counts


@contextlib.contextmanager
def quantization_disabled(qat_model: nn.Module, weights: str = "deployed"
                          ) -> Iterator[nn.Module]:
    """Run a QAT model as the float export of the given weight mode computes it.

    Activation quantizers become identities (``TQTQuantizer.forward`` does not
    consult its ``quant_enabled`` buffer, so this replaces the method for the
    duration), and the residual add's rounding of its inputs onto the output
    grid is switched off. With ``deployed`` weights the weight and bias
    quantizers stay active; with ``latent`` weights they are disabled too.
    """
    saved: List[Tuple[nn.Module, str, object]] = []
    for module in qat_model.modules():
        if isinstance(module, FakeQuantizer) and (
                weights == "latent" or getattr(module, "tensor_type", "act") == "act"):
            saved.append((module, "forward", module.__dict__.get("forward")))
            module.forward = lambda x: x
        if isinstance(module, QElementwiseAdd):
            saved.append((module, "align_inputs", module.align_inputs))
            module.align_inputs = False
    try:
        yield qat_model
    finally:
        for module, attr, value in reversed(saved):
            if attr == "forward":
                if value is None:
                    del module.forward
                else:
                    module.forward = value
            else:
                setattr(module, attr, value)


def _module_constructor(module: nn.Module) -> str:
    """Source that constructs a module of the same configuration."""
    if isinstance(module, nn.Conv2d):
        return (f"nn.Conv2d({module.in_channels}, {module.out_channels}, "
                f"kernel_size={tuple(module.kernel_size)}, stride={tuple(module.stride)}, "
                f"padding={tuple(module.padding)}, dilation={tuple(module.dilation)}, "
                f"groups={module.groups}, bias={module.bias is not None})")
    if isinstance(module, nn.Linear):
        return (f"nn.Linear({module.in_features}, {module.out_features}, "
                f"bias={module.bias is not None})")
    if isinstance(module, nn.MaxPool2d):
        return (f"nn.MaxPool2d(kernel_size={module.kernel_size!r}, stride={module.stride!r}, "
                f"padding={module.padding!r}, dilation={module.dilation!r}, "
                f"ceil_mode={module.ceil_mode!r})")
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return f"nn.AdaptiveAvgPool2d({module.output_size!r})"
    if isinstance(module, nn.AvgPool2d):
        return (f"nn.AvgPool2d(kernel_size={module.kernel_size!r}, stride={module.stride!r}, "
                f"padding={module.padding!r}, ceil_mode={module.ceil_mode!r}, "
                f"count_include_pad={module.count_include_pad!r})")
    if isinstance(module, nn.ReLU6):
        return "nn.ReLU6()"
    if isinstance(module, nn.ReLU):
        return "nn.ReLU()"
    if isinstance(module, nn.Flatten):
        return f"nn.Flatten({module.start_dim}, {module.end_dim})"
    if isinstance(module, nn.Dropout):
        return f"nn.Dropout(p={module.p!r})"
    if isinstance(module, nn.Identity):
        return "nn.Identity()"
    raise FloatExportError(f"cannot emit {type(module).__qualname__}")


def emit_source(gm: fx.GraphModule, class_name: str, header: str) -> str:
    """Standalone Python source for the converted graph.

    The module tree is rebuilt with the same dotted names, so ``gm.state_dict()``
    loads into it unchanged, and ``forward`` is the graph's own generated code.
    """
    leaves = [(name, module) for name, module in gm.named_modules()
              if name and not any(True for _ in module.children())]
    lines = [f'"""{header}"""', "",
             "import torch", "import torch.nn as nn", "", "",
             "def _attach(root, path, module):",
             '    *parents, name = path.split(".")',
             "    node = root",
             "    for part in parents:",
             "        if not hasattr(node, part):",
             "            setattr(node, part, nn.Module())",
             "        node = getattr(node, part)",
             "    setattr(node, name, module)", "", "",
             f"class {class_name}(nn.Module):",
             "    def __init__(self):",
             "        super().__init__()"]
    for name, module in leaves:
        lines.append(f"        _attach(self, {name!r}, {_module_constructor(module)})")
    lines.append("")
    forward = gm.code.strip("\n").splitlines()
    lines.extend("    " + line if line else "" for line in forward)
    return "\n".join(lines) + "\n"


def quantization_record(qat_model: fx.GraphModule) -> Dict[str, Dict[str, object]]:
    """FixQuant's fractional bits for every quantized tensor, by graph node name.

    Each value is the number of fractional bits of a signed 8-bit,
    power-of-two-scaled tensor (real value = integer * 2**-frac). Node names
    are those of the exported model: convolutions and pools keep their module
    names, and residual adds keep the QAT node's name.
    """
    modules = dict(qat_model.named_modules())
    record: Dict[str, Dict[str, object]] = {}

    def frac(quantizer) -> int:
        return int(quantizer.export_quant_info()[1])

    for node in qat_model.graph.nodes:
        if node.op != "call_module":
            continue
        module = modules[node.target]
        if isinstance(module, (FusedConvBN, QuantizedConv2d, QuantizedLinear)):
            record[node.target] = {"kind": type(module).__name__, "bits": 8,
                                   "weight": frac(module.weight_quantizer),
                                   "bias": frac(module.bias_quantizer),
                                   "output": frac(module.act_quantizer)}
        elif isinstance(module, (QMaxPool2D, QAdaptiveAvgPool2d)):
            record[node.target] = {"kind": type(module).__name__, "bits": 8,
                                   "output": frac(module.quantizer)}
        elif isinstance(module, QElementwiseAdd):
            record[node.name] = {"kind": "add", "bits": 8,
                                 "output": frac(module.quantizer)}
        elif isinstance(module, QuantStubC):
            record["input"] = {"kind": "input", "bits": 8, "output": frac(module.quantizer)}
    return record
