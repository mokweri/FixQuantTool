import logging
from typing import Dict, List, Optional, Tuple, Iterable, Any

import torch
import torch.nn as nn
import torch.fx as fx

from fixquant.quantization.fix_ops import to_int_tensor


class StdModelInspector:
    """
    Introspects a standard (inference) model produced by InferProcessor.convert_to_std_model().
    - Builds a graph view with predecessors/successors (handles branches/residual adds).
    - Exposes per-layer quantization params (frac_w, frac_b, frac_out, frac_in list).
    - Provides utilities to capture activations (input/output) via hooks and record shapes.
    - Provides utilities to save quantized activations and parameters to files.
    - Dumps a comprehensive graph summary (types, shapes, quant params, edges).
    """

    def __init__(self, std_model: fx.GraphModule, default_input_frac: int = 5, logger: Optional[logging.Logger] = None):
        self.model: fx.GraphModule = std_model
        self.default_input_frac = int(default_input_frac)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Graph structures
        self.predecessors: Dict[str, List[str]] = {}
        self.successors: Dict[str, List[str]] = {}
        self.node_types: Dict[str, str] = {}

        # Activation capture
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        # For multi-input nodes, store list[Tensor]; for single-input, a list of len 1
        self.activations_input: Dict[str, List[torch.Tensor]] = {}
        self.activations_output: Dict[str, torch.Tensor] = {}

        # Shapes
        self.input_shapes: Dict[str, List[Tuple[int, ...]]] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}

        self._analyze_graph()

    # ---------------------- Graph Analysis ----------------------
    def _analyze_graph(self):
        self.predecessors.clear()
        self.successors.clear()
        self.node_types.clear()

        modules = dict(self.model.named_modules())

        for node in self.model.graph.nodes:
            if node.op == "call_module":
                target_name = node.target
                if target_name not in modules:
                    # Some nodes like Flatten may be inserted with node.name; fallback to name
                    target_name = node.name
                self.node_types[target_name] = type(modules.get(target_name, nn.Module())).__name__
                preds: List[str] = []
                for arg in node.args:
                    if hasattr(arg, "op"):
                        if arg.op == "call_module":
                            preds.append(arg.target)
                        elif arg.op == "call_function" and hasattr(arg, "args"):
                            prev = arg.args[0] if arg.args else None
                            if prev is not None and hasattr(prev, "op") and prev.op == "call_module":
                                preds.append(prev.target)
                        elif arg.op == "placeholder":
                            pass
                self.predecessors[target_name] = preds
                for p in preds:
                    self.successors.setdefault(p, []).append(target_name)
                self.successors.setdefault(target_name, [])

        # Ensure all referenced nodes exist in dicts
        for name, mod in self.model.named_modules():
            if name == "":  # root
                continue
            self.predecessors.setdefault(name, [])
            self.successors.setdefault(name, [])
            self.node_types.setdefault(name, type(mod).__name__)

        self.logger.debug("Graph analysis complete. Nodes: %d", len(self.predecessors))

    def list_layers(self, types: Tuple[type, ...] = (nn.Conv2d, nn.Linear)) -> List[str]:
        return [name for name, mod in self.model.named_modules() if isinstance(mod, types)]

    def get_predecessors(self, name: str) -> List[str]:
        return list(self.predecessors.get(name, []))

    def get_successors(self, name: str) -> List[str]:
        return list(self.successors.get(name, []))

    def get_module(self, name: str) -> nn.Module:
        return self.model.get_submodule(name)

    # ---------------------- Quantization Info ----------------------
    def get_quant_params(self, name: str) -> Dict[str, Optional[int] | List[int]]:
        mod = self.get_module(name)
        frac_w = int(getattr(mod, "frac_weight", 0)) if hasattr(mod, "frac_weight") else None
        frac_b = int(getattr(mod, "frac_bias", 0)) if hasattr(mod, "frac_bias") else None
        frac_out = int(getattr(mod, "frac_act", 0)) if hasattr(mod, "frac_act") else None

        preds = self.get_predecessors(name)
        frac_in_list: List[int] = []
        for p in preds:
            pmod = self.get_module(p)
            if hasattr(pmod, "frac_act"):
                try:
                    frac_in_list.append(int(getattr(pmod, "frac_act")))
                except Exception:
                    pass
        if not frac_in_list:
            frac_in_list = [self.default_input_frac]

        return {
            "frac_w": frac_w,
            "frac_b": frac_b,
            "frac_out": frac_out,
            "frac_in": frac_in_list,
        }

    # ---------------------- Activation Hooks/Shapes ----------------------
    def register_activation_hooks(
        self,
        target_names: Iterable[str],
        capture_input: bool = True,
        capture_output: bool = True,
        clear_existing: bool = True,
    ) -> None:
        if clear_existing:
            self.remove_hooks()
            self.activations_input.clear()
            self.activations_output.clear()

        for name in target_names:
            mod = self.get_module(name)

            def _hook(m, inp, out, _name=name):
                if capture_input:
                    ins: List[torch.Tensor] = []
                    if isinstance(inp, (list, tuple)):
                        for x in inp:
                            if torch.is_tensor(x):
                                ins.append(x.detach().cpu())
                    elif torch.is_tensor(inp):
                        ins.append(inp.detach().cpu())
                    if ins:
                        self.activations_input[_name] = ins
                        self.input_shapes[_name] = [tuple(x.shape) for x in ins]
                if capture_output and out is not None and torch.is_tensor(out):
                    self.activations_output[_name] = out.detach().cpu()
                    self.output_shapes[_name] = tuple(out.shape)

            handle = mod.register_forward_hook(_hook)
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    @torch.no_grad()
    def run_and_capture(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(input_tensor)

    def collect_all_shapes(self, example_input: torch.Tensor) -> None:
        """Register hooks for all leaf modules and run once to record shapes."""
        names = [n for n, _ in self.model.named_modules() if n != ""]
        self.register_activation_hooks(names, capture_input=True, capture_output=True, clear_existing=True)
        self.run_and_capture(example_input)
        self.remove_hooks()

    # ---------------------- Save Utilities ----------------------
    def save_activation(
        self,
        name: str,
        filepath: str,
        which: str = "input",  # "input" or "output"
        n_bits: int = 8,
        n_frac: Optional[int] = None,
        which_input_idx: int = 0,
    ) -> None:
        if which not in ("input", "output"):
            raise ValueError("which must be 'input' or 'output'")
        tensor = None
        if which == "input":
            inputs = self.activations_input.get(name)
            if not inputs:
                raise RuntimeError(f"No captured input activation for layer '{name}'. Did you register hooks and run?")
            if which_input_idx >= len(inputs):
                raise IndexError(f"Layer '{name}' has only {len(inputs)} inputs; requested index {which_input_idx}")
            tensor = inputs[which_input_idx]
            if n_frac is None:
                n_frac = self.get_quant_params(name)["frac_in"][min(which_input_idx, len(self.get_quant_params(name)["frac_in"]) - 1)]
        else:
            tensor = self.activations_output.get(name)
            if tensor is None:
                raise RuntimeError(f"No captured output activation for layer '{name}'. Did you register hooks and run?")
            if n_frac is None:
                q = self.get_quant_params(name)
                n_frac = q["frac_out"] if q["frac_out"] is not None else self.default_input_frac

        if tensor.dim() > 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        int_tensor = to_int_tensor(tensor, signed=True, n_bits=n_bits, n_frac=int(n_frac))
        int_tensor.numpy().astype("int8").tofile(filepath)
        self.logger.info("Saved %s activation to '%s' (n_bits=%d, n_frac=%d)", which, filepath, n_bits, n_frac)

    def save_layer_params(
        self,
        name: str,
        weight_file: str,
        bias_file: str,
        target_weight_shape: Optional[Tuple[int, ...]] = None,
        n_bits_out: int = 8,
    ) -> Tuple[Optional[int], Optional[int]]:
        mod = self.get_module(name)
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"Layer '{name}' is not Conv2d/Linear: {type(mod).__name__}")

        frac_w = int(getattr(mod, "frac_weight", 0)) if hasattr(mod, "frac_weight") else None
        frac_b = int(getattr(mod, "frac_bias", 0)) if hasattr(mod, "frac_bias") else None

        # Weights
        if hasattr(mod, "weight") and mod.weight is not None:
            w = mod.weight.data.detach().clone()
            if target_weight_shape is not None:
                if len(target_weight_shape) != w.ndim:
                    raise ValueError(
                        f"Target weight rank {len(target_weight_shape)} != tensor rank {w.ndim}"
                    )
                slices = tuple(slice(0, min(s, w.shape[i])) for i, s in enumerate(target_weight_shape))
                w = w[slices]
            if frac_w is None:
                self.logger.warning("Layer '%s' missing frac_weight; using 0 for quantization.", name)
                frac_w = 0
            w_q = to_int_tensor(w, signed=True, n_bits=n_bits_out, n_frac=int(frac_w))
            w_q.cpu().numpy().astype("int8").tofile(weight_file)
            self.logger.info("Saved weights to '%s'", weight_file)
        else:
            open(weight_file, "wb").close()
            self.logger.info("Layer '%s' has no weights; created empty '%s'", name, weight_file)

        # Bias
        if hasattr(mod, "bias") and mod.bias is not None:
            b = mod.bias.data.detach().clone()
            if target_weight_shape is not None and isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, "weight"):
                out_len = min(mod.weight.shape[0], target_weight_shape[0])
                b = b[:out_len]
            if frac_b is None:
                self.logger.warning("Layer '%s' missing frac_bias; using 0 for quantization.", name)
                frac_b = 0
            b_q = to_int_tensor(b, signed=True, n_bits=n_bits_out, n_frac=int(frac_b))
            b_q.cpu().numpy().astype("int8").tofile(bias_file)
            self.logger.info("Saved bias to '%s'", bias_file)
        else:
            open(bias_file, "wb").close()
            self.logger.info("Layer '%s' has no bias; created empty '%s'", name, bias_file)

        return frac_w, frac_b

    # ---------------------- Info/Dump ----------------------
    def get_layer_info(self, name: str) -> Dict[str, Any]:
        mod = self.get_module(name)
        q = self.get_quant_params(name)
        info: Dict[str, Any] = {
            "name": name,
            "type": type(mod).__name__,
            "predecessors": self.get_predecessors(name),
            "successors": self.get_successors(name),
            "quant": q,
            "weight_shape": tuple(mod.weight.shape) if hasattr(mod, "weight") and mod.weight is not None else None,
            "bias_shape": tuple(mod.bias.shape) if hasattr(mod, "bias") and mod.bias is not None else None,
            "input_shapes": self.input_shapes.get(name),
            "output_shape": self.output_shapes.get(name),
        }
        return info

    def dump_graph_text(self, file: Optional[str] = None) -> None:
        lines: List[str] = []
        lines.append("Model Graph Summary:\n")
        ordered = self.topological_order()
        for name in ordered:
            info = self.get_layer_info(name)
            line = (
                f"- {info['name']} [{info['type']}]\n"
                f"  preds: {info['predecessors']}  succs: {info['successors']}\n"
                f"  quant: in={info['quant']['frac_in']} w={info['quant']['frac_w']} b={info['quant']['frac_b']} out={info['quant']['frac_out']}\n"
                f"  shapes: in={info['input_shapes']} out={info['output_shape']} weight={info['weight_shape']} bias={info['bias_shape']}\n"
            )
            lines.append(line)
        text = "\n".join(lines)
        if file:
            with open(file, "w") as f:
                f.write(text)
            self.logger.info("Wrote graph text summary to '%s'", file)
        else:
            print(text)

    def dump_graph_json(self, file: str) -> None:
        import json
        ordered = self.topological_order()
        nodes = [self.get_layer_info(n) for n in ordered]
        edges = []
        for n in ordered:
            for s in self.get_successors(n):
                edges.append({"from": n, "to": s})
        payload = {"nodes": nodes, "edges": edges}
        with open(file, "w") as f:
            json.dump(payload, f, indent=2)
        self.logger.info("Wrote graph JSON to '%s'", file)

    # ---------------------- Utility ----------------------
    def topological_order(self, filter_types: Optional[Tuple[type, ...]] = None) -> List[str]:
        indeg: Dict[str, int] = {n: len(self.predecessors.get(n, [])) for n in self.predecessors}
        queue: List[str] = [n for n, d in indeg.items() if d == 0]
        order: List[str] = []
        visited: set = set()
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            if filter_types is None or isinstance(self.get_module(n), filter_types):
                order.append(n)
            for s in self.successors.get(n, []):
                indeg[s] = max(0, indeg.get(s, 0) - 1)
                if indeg[s] == 0:
                    queue.append(s)
        # Fallback: include any stragglers to avoid missing modules
        for n in self.predecessors:
            if n not in order:
                order.append(n)
        return order

