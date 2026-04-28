import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.graph.inference_processor import AddWithMetadata


GAP_SCALE_FRAC_BITS = 16


def _clean_shape(shape_tuple, is_linear=False):
    s = list(shape_tuple)
    if len(s) > 1 and s[0] == 1:
        s = s[1:]
    if is_linear and len(s) == 1:
        s = [s[0], 1, 1]
    return s


def _as_i8(values: torch.Tensor) -> torch.Tensor:
    return torch.clamp(values, -128, 127).to(torch.int8)


def _signed_shift(values: torch.Tensor, shift: int) -> torch.Tensor:
    values = values.to(torch.int64)
    if shift > 0:
        return values << shift
    if shift < 0:
        s = -shift
        round_bias = (1 << (s - 1)) + (values >> 63)
        return (values + round_bias) >> s
    return values


def _bias_shift(values: torch.Tensor, shift: int) -> torch.Tensor:
    values = values.to(torch.int64)
    if shift >= 0:
        return values << shift
    return values >> (-shift)


def _load_i8_tensor(export_path: Path, spec: Dict[str, Any]) -> torch.Tensor:
    data = np.fromfile(str(export_path / spec["file"]), dtype=np.int8)
    return torch.from_numpy(data.copy()).to(torch.int8).reshape(spec["shape"])


def _write_i8_tensor(export_path: Path, spec: Dict[str, Any], tensor: torch.Tensor) -> None:
    out = tensor.detach().cpu().contiguous().numpy().astype(np.int8)
    out.tofile(str(export_path / spec["file"]))


def _tilecnn_conv2d(ifm: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                    node: Dict[str, Any], ifm_frac: int, weight_frac: int,
                    bias_frac: int, out_frac: int) -> torch.Tensor:
    attrs = node.get("attrs", {})
    stride = tuple(attrs.get("stride", [1, 1]))
    padding_attr = attrs.get("padding", [0, 0, 0, 0])
    padding = (padding_attr[0], padding_attr[2])
    dilation = tuple(attrs.get("dilation", [1, 1]))
    groups = int(attrs.get("groups", 1))
    if groups != 1:
        raise ValueError("TileCNN reference supports groups=1 only")

    x = ifm.reshape(1, *ifm.shape).to(torch.float64)
    w = weight.to(torch.float64)
    y = F.conv2d(x, w, bias=None, stride=stride, padding=padding,
                 dilation=dilation, groups=groups).to(torch.int64)

    shift_out = ifm_frac + weight_frac - out_frac
    if shift_out > 0:
        s1 = (y >> (shift_out - 1)) + 1
    elif shift_out == 0:
        s1 = (y << 1) + 1
    else:
        s1 = (y << ((-shift_out) + 1)) + 1

    bias_adj = _bias_shift(bias, out_frac - bias_frac + 1).view(1, -1, 1, 1)
    out = (s1 + bias_adj + 1) >> 1
    out = _as_i8(out)
    if node.get("post_ops", {}).get("relu", False) and not node.get("post_ops", {}).get("residual_add", False):
        out = torch.clamp_min(out, 0)
    return out.squeeze(0)


def _tilecnn_linear(ifm: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                    ifm_frac: int, weight_frac: int, bias_frac: int,
                    out_frac: int) -> torch.Tensor:
    x = ifm.reshape(1, -1).to(torch.float64)
    w = weight.to(torch.float64)
    y = torch.matmul(x, w.t()).to(torch.int64)

    shift_out = ifm_frac + weight_frac - out_frac
    if shift_out > 0:
        s1 = (y >> (shift_out - 1)) + 1
    elif shift_out == 0:
        s1 = (y << 1) + 1
    else:
        s1 = (y << ((-shift_out) + 1)) + 1

    bias_adj = _bias_shift(bias, out_frac - bias_frac + 1).view(1, -1)
    out = (s1 + bias_adj + 1) >> 1
    return _as_i8(out).reshape(weight.shape[0], 1, 1)


def _tilecnn_residual_add(main: torch.Tensor, residual: torch.Tensor,
                          residual_shift: int, relu: bool) -> torch.Tensor:
    summed = main.to(torch.int64) + _signed_shift(residual, residual_shift)
    out = _as_i8(summed)
    if relu:
        out = torch.clamp_min(out, 0)
    return out


def _tilecnn_maxpool(ifm: torch.Tensor, post_pool_shift: int) -> torch.Tensor:
    y = F.max_pool2d(ifm.reshape(1, *ifm.shape).to(torch.float32),
                     kernel_size=3, stride=2, padding=1).to(torch.int64)
    return _as_i8(_signed_shift(y, post_pool_shift)).squeeze(0)


def _tilecnn_gap(ifm: torch.Tensor, in_frac: int, out_frac: int) -> torch.Tensor:
    channels, height, width = ifm.shape
    num_elems = height * width
    total_shift = GAP_SCALE_FRAC_BITS + (out_frac - in_frac)
    if total_shift < 0:
        raise ValueError("TileCNN GAP reference cannot use negative total shift")
    gap_mul = ((1 << total_shift) + num_elems // 2) // num_elems
    sums = ifm.to(torch.int64).reshape(channels, -1).sum(dim=1)
    scaled = (sums * gap_mul + (1 << (GAP_SCALE_FRAC_BITS - 1))) >> GAP_SCALE_FRAC_BITS
    return _as_i8(scaled).reshape(channels, 1, 1)


def _write_tilecnn_bitexact_references(export_path: Path, graph_json: Dict[str, Any],
                                       logger: logging.Logger) -> None:
    tensors = graph_json["tensors"]
    values: Dict[str, torch.Tensor] = {}

    for tensor_id, spec in tensors.items():
        if spec.get("kind") in ("input", "param") and "file" in spec:
            values[tensor_id] = _load_i8_tensor(export_path, spec)

    for node in graph_json["nodes"]:
        op = node["op"]
        inputs = node.get("inputs", {})
        outputs = node.get("outputs", {})

        if op == "conv2d":
            ifm_id = inputs["ifm"]
            weight_id = inputs["weight"]
            bias_id = inputs["bias"]
            ofm_id = outputs["ofm"]
            ifm = values[ifm_id]
            weight = values[weight_id]
            bias = values[bias_id]
            out = _tilecnn_conv2d(
                ifm, weight, bias, node,
                tensors[ifm_id]["frac"], tensors[weight_id]["frac"],
                tensors[bias_id]["frac"], tensors[ofm_id]["frac"])
            if node.get("post_ops", {}).get("residual_add", False):
                residual_id = inputs["residual"]
                residual_shift = tensors[ofm_id]["frac"] - tensors[residual_id]["frac"]
                residual_shift = int(node.get("post_ops", {}).get("residual_shift", residual_shift))
                out = _tilecnn_residual_add(
                    out, values[residual_id], residual_shift,
                    bool(node.get("post_ops", {}).get("post_add_relu", False)))
            values[ofm_id] = out

        elif op == "linear":
            ifm_id = inputs["ifm"]
            weight_id = inputs["weight"]
            bias_id = inputs["bias"]
            ofm_id = outputs["ofm"]
            values[ofm_id] = _tilecnn_linear(
                values[ifm_id], values[weight_id], values[bias_id],
                tensors[ifm_id]["frac"], tensors[weight_id]["frac"],
                tensors[bias_id]["frac"], tensors[ofm_id]["frac"])

        elif op == "maxpool2d":
            ifm_id = inputs["ifm"]
            ofm_id = outputs["ofm"]
            values[ofm_id] = _tilecnn_maxpool(
                values[ifm_id], tensors[ofm_id]["frac"] - tensors[ifm_id]["frac"])

        elif op == "gap2d":
            ifm_id = inputs["ifm"]
            ofm_id = outputs["ofm"]
            values[ofm_id] = _tilecnn_gap(values[ifm_id], tensors[ifm_id]["frac"], tensors[ofm_id]["frac"])

        else:
            raise ValueError(f"Unsupported TileCNN reference op: {op}")

    for output_id, ref_id in graph_json["graph"].get("references", {}).items():
        if output_id not in values:
            raise ValueError(f"Cannot generate reference for missing graph output '{output_id}'")
        _write_i8_tensor(export_path, tensors[ref_id], values[output_id])
        logger.info("Rewrote TileCNN bit-exact reference '%s'", tensors[ref_id]["file"])


class TileCNNGraphExporter:
    """
    Exports a converted QAT standard model to the TileCNN graph handoff specification format.
    Can export the full graph or a designated sub-graph.
    """

    def __init__(
            self,
            inspector: StdModelInspector,
            model_name: str = "resnet18",
            producer_version: str = "0.1.0",
            default_input_frac: int = 5,
            logger: Optional[logging.Logger] = None
    ):
        self.inspector = inspector
        self.model_name = model_name
        self.producer_version = producer_version
        self.default_input_frac = default_input_frac
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def export(self, export_dir: str, subgraph_nodes: Optional[List[str]] = None):
        """
        Exports the graph. If subgraph_nodes is provided, exports only the nodes in the list,
        automatically resolving boundary inputs and references.
        """
        export_path = Path(export_dir)
        inputs_dir = export_path / "inputs"
        params_dir = export_path / "params"
        refs_dir = export_path / "refs"

        inputs_dir.mkdir(parents=True, exist_ok=True)
        params_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)

        ordered = self.inspector.topological_order()
        
        if subgraph_nodes is not None:
            # Keep nodes that are in the subgraph
            ordered = [n for n in ordered if n in subgraph_nodes]
            if not ordered:
                self.logger.error("No valid nodes found for subgraph export.")
                return

            # Validation: Check for skipped nodes creating disconnected components
            adj = {n: [] for n in ordered}
            for n in ordered:
                for p in self.inspector.get_predecessors(n):
                    if p in ordered:
                        adj[n].append(p)
                        adj[p].append(n)
            
            visited = set()
            components = 0
            for n in ordered:
                if n not in visited:
                    components += 1
                    q = [n]
                    visited.add(n)
                    while q:
                        curr = q.pop(0)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
            
            if components > 1:
                self.logger.warning(
                    f"Subgraph validation: Found {components} disconnected components in the specified nodes! "
                    "You likely skipped an intermediate node (like a ReLU or Add) which breaks the graph's contiguity. "
                    "The missing node will force the graph to split into disconnected segments, treating broken links as external inputs/outputs."
                )

        tensors = {}
        built_nodes = {}
        tensor_map = {}
        node_to_fused = {}
        
        graph_inputs = []

        # Find and register inputs (edges coming into the subgraph from outside)
        for name in ordered:
            preds = self.inspector.get_predecessors(name)
            if not preds:
                # Global input
                if "input" not in tensors:
                    input_shape = _clean_shape(self.inspector.input_shapes[name][0])
                    tensors["input"] = {
                        "kind": "input",
                        "dtype": "int8",
                        "layout": "CHW",
                        "shape": input_shape,
                        "frac": self.default_input_frac,
                        "file": "inputs/input_0.int8.bin"
                    }
                    self.inspector.save_activation(name, str(inputs_dir / "input_0.int8.bin"), which="input", n_frac=self.default_input_frac)
                    graph_inputs.append("input")
            else:
                for i, p in enumerate(preds):
                    if p not in ordered:
                        # External input to the subgraph
                        ext_in_name = f"ext_in_{p.replace('.', '_')}" if p else "input"
                        if ext_in_name not in tensors:
                            shape = _clean_shape(self.inspector.input_shapes[name][i])
                            q_params = self.inspector.get_quant_params(name)
                            in_frac = q_params["frac_in"][i] if i < len(q_params["frac_in"]) else self.default_input_frac
                            tensors[ext_in_name] = {
                                "kind": "input",
                                "dtype": "int8",
                                "layout": "CHW",
                                "shape": shape,
                                "frac": in_frac,
                                "file": f"inputs/{ext_in_name}.int8.bin"
                            }
                            self.inspector.save_activation(name, str(inputs_dir / f"{ext_in_name}.int8.bin"), which="input", which_input_idx=i, n_frac=in_frac)
                            graph_inputs.append(ext_in_name)
                        # map the external predecessor to this input tensor name
                        tensor_map[p] = ext_in_name

        # Iterate through nodes and lower them to Spec nodes
        for name in ordered:
            mod = self.inspector.get_module(name)
            preds = self.inspector.get_predecessors(name)

            if isinstance(mod, nn.Conv2d) or type(mod).__name__ in (
                    "FxP_QConv2D", "HLSConv2d", "TileCNNConv2d"):
                node_id = name.replace(".", "_")
                # TileCNNConv2d with fused residual add receives (main, residual) as two preds
                is_tilecnn = type(mod).__name__ == "TileCNNConv2d"
                is_emu     = type(mod).__name__ in ("FxP_QConv2D", "HLSConv2d")

                # Pick the main IFM (first pred)
                ifm = tensor_map[preds[0]] if preds else "input"

                w_tensor = mod.w_int8 if is_emu or is_tilecnn else mod.weight
                k = (w_tensor.shape[2], w_tensor.shape[3])

                node = {
                    "id": node_id,
                    "op": "conv2d",
                    "inputs": {
                        "ifm": ifm,
                        "weight": f"{node_id}_w",
                        "bias": f"{node_id}_b"
                    },
                    "outputs": {
                        "ofm": f"{node_id}_out"
                    },
                    "attrs": {
                        "kernel": [k[0], k[1]],
                        "stride": list(mod.stride) if isinstance(mod.stride, (list, tuple)) else [mod.stride, mod.stride],
                        "padding": [mod.padding[0], mod.padding[0], mod.padding[1], mod.padding[1]] if isinstance(mod.padding, (list, tuple)) else [mod.padding]*4,
                        "dilation": list(mod.dilation) if isinstance(mod.dilation, (list, tuple)) else [mod.dilation, mod.dilation],
                        "groups": mod.groups
                    },
                    "post_ops": {}
                }

                # For TileCNNConv2d with fused residual add, wire up post_ops immediately
                # (the Add node is gone from the graph — fusion already happened)
                if is_tilecnn and getattr(mod, "residual_add", False):
                    # preds[1] is the residual branch
                    if len(preds) >= 2:
                        residual_tensor = tensor_map.get(preds[1], preds[1])
                        node["post_ops"]["residual_add"] = True
                        node["inputs"]["residual"] = residual_tensor
                    if getattr(mod, "post_add_relu", False):
                        node["post_ops"]["post_add_relu"] = True

                built_nodes[node_id] = node
                node_to_fused[name] = node_id

                if is_emu:
                    frac_out = mod.qconfig[mod.module_name]["frac_out"]
                elif is_tilecnn:
                    frac_out = mod.fout  # already set to add's frac_out for fused nodes
                else:
                    frac_out = int(mod.frac_act.item()) if hasattr(mod, "frac_act") else self.default_input_frac

                tensors[f"{node_id}_out"] = {
                    "kind": "activation",
                    "dtype": "int8",
                    "layout": "CHW",
                    "shape": _clean_shape(self.inspector.output_shapes[name]),
                    "frac": frac_out
                }
                tensor_map[name] = f"{node_id}_out"

                weight_file = params_dir / f"{node_id}.weight.int8.bin"
                bias_file   = params_dir / f"{node_id}.bias.int8.bin"
                frac_w, frac_b = self.inspector.save_layer_params(name, str(weight_file), str(bias_file))

                has_bias = (hasattr(mod, "b_int8") and mod.b_int8 is not None) or \
                           (hasattr(mod, "bias") and mod.bias is not None)
                if not has_bias:
                    np.zeros(w_tensor.shape[0], dtype=np.int8).tofile(str(bias_file))
                    frac_b = 0

                w_shape = list(w_tensor.shape)
                if hasattr(mod, "b_int8") and mod.b_int8 is not None:
                    b_shape = list(mod.b_int8.shape)
                elif hasattr(mod, "bias") and mod.bias is not None:
                    b_shape = list(mod.bias.shape)
                else:
                    b_shape = [w_shape[0]]

                tensors[f"{node_id}_w"] = {
                    "kind": "param",
                    "dtype": "int8",
                    "layout": "OIHW",
                    "shape": w_shape,
                    "frac": frac_w if frac_w is not None else 0,
                    "file": f"params/{node_id}.weight.int8.bin"
                }
                tensors[f"{node_id}_b"] = {
                    "kind": "param",
                    "dtype": "int8",
                    "layout": "O",
                    "shape": b_shape,
                    "frac": frac_b if frac_b is not None else 0,
                    "file": f"params/{node_id}.bias.int8.bin"
                }

            elif isinstance(mod, nn.Linear) or type(mod).__name__ in (
                    "FxP_QLinear", "HLSLinear", "TileCNNLinear"):
                node_id = name.replace(".", "_")
                ifm = tensor_map[preds[0]] if preds else "input"
                
                is_emu     = type(mod).__name__ in ("FxP_QLinear", "HLSLinear")
                is_tilecnn = type(mod).__name__ == "TileCNNLinear"

                node = {
                    "id": node_id,
                    "op": "linear",
                    "inputs": {
                        "ifm": ifm,
                        "weight": f"{node_id}_w",
                        "bias": f"{node_id}_b"
                    },
                    "outputs": {
                        "ofm": f"{node_id}_out"
                    }
                }
                built_nodes[node_id] = node
                node_to_fused[name] = node_id

                if is_emu:
                    frac_out = mod.qconfig[mod.module_name]["frac_out"]
                elif is_tilecnn:
                    frac_out = mod.fout
                else:
                    frac_out = int(mod.frac_act.item()) if hasattr(mod, "frac_act") else self.default_input_frac
                    
                tensors[f"{node_id}_out"] = {
                    "kind": "activation",
                    "dtype": "int8",
                    "layout": "CHW",
                    "shape": _clean_shape(self.inspector.output_shapes[name], is_linear=True),
                    "frac": frac_out
                }
                tensor_map[name] = f"{node_id}_out"

                weight_file = params_dir / f"{node_id}.weight.int8.bin"
                bias_file = params_dir / f"{node_id}.bias.int8.bin"
                frac_w, frac_b = self.inspector.save_layer_params(name, str(weight_file), str(bias_file))

                has_bias = (hasattr(mod, "bias") and mod.bias is not None) or (hasattr(mod, "b_int8") and mod.b_int8 is not None)
                if not has_bias:
                    np.zeros(mod.weight.shape[0] if hasattr(mod, "weight") else mod.w_int8.shape[0], dtype=np.int8).tofile(str(bias_file))
                    frac_b = 0

                w_shape = list(mod.weight.shape) if hasattr(mod, "weight") else list(mod.w_int8.shape)
                b_shape = list(mod.bias.shape) if hasattr(mod, "bias") and mod.bias is not None else list(mod.b_int8.shape) if hasattr(mod, "b_int8") and mod.b_int8 is not None else [w_shape[0]]

                tensors[f"{node_id}_w"] = {
                    "kind": "param",
                    "dtype": "int8",
                    "layout": "OI",
                    "shape": w_shape,
                    "frac": frac_w if frac_w is not None else 0,
                    "file": f"params/{node_id}.weight.int8.bin"
                }
                tensors[f"{node_id}_b"] = {
                    "kind": "param",
                    "dtype": "int8",
                    "layout": "O",
                    "shape": b_shape,
                    "frac": frac_b if frac_b is not None else 0,
                    "file": f"params/{node_id}.bias.int8.bin"
                }

            elif isinstance(mod, nn.MaxPool2d) or type(mod).__name__ in (
                    "FxP_QMaxPool2D", "HLSMaxPool2D", "TileCNNMaxPool"):
                node_id = name.replace(".", "_")
                ifm = tensor_map[preds[0]] if preds else "input"
                
                k = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
                s = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
                p = mod.padding if isinstance(mod.padding, tuple) else (mod.padding, mod.padding)
                
                node = {
                    "id": node_id,
                    "op": "maxpool2d",
                    "inputs": {
                        "ifm": ifm
                    },
                    "outputs": {
                        "ofm": f"{node_id}_out"
                    },
                    "attrs": {
                        "kernel": list(k),
                        "stride": list(s),
                        "padding": [p[0], p[0], p[1], p[1]]
                    }
                }
                built_nodes[node_id] = node
                node_to_fused[name] = node_id

                is_emu     = type(mod).__name__ in ("FxP_QMaxPool2D", "HLSMaxPool2D")
                is_tilecnn = type(mod).__name__ == "TileCNNMaxPool"
                if is_emu:
                    frac_out = mod.qconfig[mod.module_name]["frac_out"]
                elif is_tilecnn:
                    # TileCNNMaxPool has no fout attr; derive from qconfig or fallback
                    q_mp = self.inspector.get_quant_params(name)
                    fin_mp = q_mp["frac_in"][0] if q_mp["frac_in"] else self.default_input_frac
                    frac_out = q_mp.get("frac_out") or (fin_mp + getattr(mod, "post_pool_shift", 0))
                else:
                    frac_out = int(mod.frac_act.item()) if hasattr(mod, "frac_act") else self.default_input_frac

                    
                tensors[f"{node_id}_out"] = {
                    "kind": "activation",
                    "dtype": "int8",
                    "layout": "CHW",
                    "shape": _clean_shape(self.inspector.output_shapes[name]),
                    "frac": frac_out
                }
                tensor_map[name] = f"{node_id}_out"

            elif isinstance(mod, nn.AdaptiveAvgPool2d) or type(mod).__name__ in (
                    "FxP_QAdaptiveAvgPool2d", "HLSAdaptiveAvgPool2d", "TileCNNGAP"):
                node_id = name.replace(".", "_")
                ifm = tensor_map[preds[0]] if preds else "input"
                
                node = {
                    "id": node_id,
                    "op": "gap2d",
                    "inputs": {
                        "ifm": ifm
                    },
                    "outputs": {
                        "ofm": f"{node_id}_out"
                    }
                }
                built_nodes[node_id] = node
                node_to_fused[name] = node_id

                is_emu     = type(mod).__name__ in ("FxP_QAdaptiveAvgPool2d", "HLSAdaptiveAvgPool2d")
                is_tilecnn = type(mod).__name__ == "TileCNNGAP"
                if is_emu:
                    frac_out = mod.qconfig[mod.module_name]["frac_out"]
                elif is_tilecnn:
                    frac_out = mod.fout
                else:
                    frac_out = int(mod.frac_act.item()) if hasattr(mod, "frac_act") else self.default_input_frac
                    
                tensors[f"{node_id}_out"] = {
                    "kind": "activation",
                    "dtype": "int8",
                    "layout": "CHW",
                    "shape": _clean_shape(self.inspector.output_shapes[name]),
                    "frac": frac_out
                }
                tensor_map[name] = f"{node_id}_out"

            elif isinstance(mod, nn.ReLU) or type(mod).__name__ in ("HLSRelu",):
                # Standalone ReLU nodes in both emu and tilecnn models —
                # fuse them into their predecessor's post_ops
                pred = preds[0] if preds else None
                if pred is None:
                    continue
                fused_id = node_to_fused.get(pred)
                if fused_id and fused_id in built_nodes:
                    if built_nodes[fused_id]["post_ops"].get("residual_add"):
                        built_nodes[fused_id]["post_ops"]["post_add_relu"] = True
                    else:
                        built_nodes[fused_id]["post_ops"]["relu"] = True
                    tensor_map[name] = tensor_map[pred]
                    node_to_fused[name] = fused_id
                else:
                    tensor_map[name] = tensor_map.get(pred, "input")
                    node_to_fused[name] = fused_id

            elif isinstance(mod, AddWithMetadata) or type(mod).__name__ in (
                    "AddWithMetadata", "FxP_QElementwiseAdd", "HLSElementwiseAdd"):
                # AddWithMetadata only exists in the emu model — in tilecnn it is fused
                # into the preceding TileCNNConv2d and the node is erased from the graph.
                if len(preds) < 2:
                    # Orphan / erased node slipped through — skip silently
                    self.logger.warning(
                        f"AddWithMetadata '{name}' has {len(preds)} predecessor(s); expected 2. Skipping.")
                    if preds:
                        tensor_map[name] = tensor_map.get(preds[0], "input")
                    continue

                pred1, pred2 = preds[0], preds[1]
                topo = self.inspector.topological_order()
                idx1 = topo.index(pred1) if pred1 in topo else -1
                idx2 = topo.index(pred2) if pred2 in topo else -1

                if idx1 > idx2:
                    main_pred, res_pred = pred1, pred2
                else:
                    main_pred, res_pred = pred2, pred1

                fused_id = node_to_fused.get(main_pred)
                if fused_id and fused_id in built_nodes:
                    built_nodes[fused_id]["post_ops"]["residual_add"] = True
                    built_nodes[fused_id]["inputs"]["residual"] = tensor_map.get(res_pred, res_pred)

                    is_emu = type(mod).__name__ in ("FxP_QElementwiseAdd", "HLSElementwiseAdd")
                    if is_emu:
                        ofm_name = built_nodes[fused_id]["outputs"]["ofm"]
                        tensors[ofm_name]["frac"] = mod.qconfig[mod.module_name]["frac_out"]
                    elif hasattr(mod, "frac_act"):
                        ofm_name = built_nodes[fused_id]["outputs"]["ofm"]
                        tensors[ofm_name]["frac"] = int(mod.frac_act.item())

                tensor_map[name] = tensor_map.get(main_pred, "input")
                node_to_fused[name] = fused_id

            elif isinstance(mod, nn.Flatten):
                tensor_map[name] = tensor_map[preds[0]] if preds else "input"
                node_to_fused[name] = node_to_fused.get(preds[0]) if preds else None
            
            elif type(mod).__name__ == "QuantStubC" or type(mod).__name__ == "Dropout":
                if preds:
                    tensor_map[name] = tensor_map[preds[0]]
                    node_to_fused[name] = node_to_fused.get(preds[0])

            elif type(mod).__name__ in ("InputQuantizer",):
                # InputQuantizer is injected by convert_to_emu_model at the graph entry.
                # It consumes the raw float input (a graph placeholder, not in tensor_map)
                # and emits the first quantized activation.  Map it to the global "input"
                # tensor so that the next Conv can find its IFM in tensor_map.
                tensor_map[name] = "input"
                node_to_fused[name] = None

            elif type(mod).__name__ in ("OutputDequantizer",):
                # OutputDequantizer is injected at the graph exit.  It doesn't create a
                # new hardware tensor — it is purely a software de-scaling stub.
                # Pass through the predecessor's tensor so reference collection works.
                if preds:
                    tensor_map[name] = tensor_map.get(preds[0], "input")
                    node_to_fused[name] = node_to_fused.get(preds[0])

            else:
                self.logger.warning(f"Unhandled module type {type(mod).__name__} for {name}")
                if preds:
                    tensor_map[name] = tensor_map.get(preds[0], "input")
                    node_to_fused[name] = node_to_fused.get(preds[0])


        # Find and register references (outputs of the subgraph)
        graph_outputs = []
        references = {}
        for name in ordered:
            succs = self.inspector.get_successors(name)
            # It's an output if it has no successors inside the subgraph
            if not any(s in ordered for s in succs):
                out_tensor = tensor_map[name]
                if out_tensor and out_tensor not in graph_outputs:
                    graph_outputs.append(out_tensor)
                    ref_file_path = refs_dir / f"{out_tensor}_ref.int8.bin"
                    self.inspector.save_activation(name, str(ref_file_path), which="output")

                    tensors[f"{out_tensor}_ref"] = {
                        "kind": "reference",
                        "dtype": "int8",
                        "layout": tensors[out_tensor]["layout"],
                        "shape": tensors[out_tensor]["shape"],
                        "frac": tensors[out_tensor]["frac"],
                        "file": f"refs/{out_tensor}_ref.int8.bin"
                    }
                    references[out_tensor] = f"{out_tensor}_ref"

        # Construct Graph JSON
        graph_json = {
            "schema": "tilecnn.graph.v1",
            "model": {
                "name": self.model_name,
                "producer": "fixed_point_quantizer",
                "producer_version": self.producer_version,
                "source_framework": "pytorch"
            },
            "target": {
                "bitwidth": 8,
                "signed": True,
                "activation_layout": "CHW",
                "conv_weight_layout": "OIHW",
                "linear_weight_layout": "OI",
                "bias_dtype": "int8"
            },
            "graph": {
                "inputs": list(set(graph_inputs)),
                "outputs": list(set(graph_outputs)),
                "references": references
            },
            "tensors": tensors,
            "nodes": list(built_nodes.values())
        }

        graph_path = export_path / "graph.json"
        with open(graph_path, "w") as f:
            json.dump(graph_json, f, indent=2)

        _write_tilecnn_bitexact_references(export_path, graph_json, self.logger)

        self.logger.info(f"TileCNN graph successfully exported to {export_path}")
