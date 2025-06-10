import torch
import torch.fx as fx
import torch.nn as nn
import copy
import onnx
import logging
from typing import Optional
import numpy as np

from quantization.fix_ops import fake_quantize_tensor, to_int_tensor
from quantization.fusedConvBn import FusedConvBN
from quantization.qat_modules import (QElementwiseAdd, QuantStubC, QMaxPool2D,
                                      QAdaptiveAvgPool2d, QuantizedLinear, QuantizedConv2d)


class StandardModel(nn.Module):
    """New model composed of standard PyTorch modules"""

    def __init__(self):
        super(StandardModel, self).__init__()
        self.layers = nn.ModuleDict()

    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x


class AddWithMetadata(nn.Module):
    def __init__(self):
        super(AddWithMetadata, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


# ----------------------------------------------------------------------
def _get_src_param(mod: nn.Module, name: str) -> Optional[torch.Tensor]:
    """
    Return the tensor `weight` / `bias` from either
        • plain module (mod.weight)
        • fused module (mod.conv_mod.weight)
    """
    if hasattr(mod, name):                         # plain layer
        t = getattr(mod, name)
    elif hasattr(mod, "conv_mod") and hasattr(mod.conv_mod, name):
        t = getattr(mod.conv_mod, name)            # fused layer
    else:
        t = None
    return t.detach() if isinstance(t, torch.Tensor) else None


def _quantize_det(t: torch.Tensor, n_bits: int, n_frac: int, signed: bool):
    """Deterministic fake-quant wrapper (assumes fake_quantize_tensor is pure)."""
    return fake_quantize_tensor(t, signed=signed, n_bits=n_bits, n_frac=n_frac)


@torch.no_grad()
def _copy_and_quantize_params(
        src: nn.Module, dst: nn.Module
) -> None:
    """
    Copy weight/bias from `src` to `dst`, quantising them with the
    frac bits stored in `src.export_quant_info()`.
    Works for both “plain” and fused Conv-BN containers.
    """

    # ---- 1. Weight ----------------------------------------------------
    w_src = _get_src_param(src, "weight")
    if w_src is None:
        print(f"[copy-quant]  No weight found in {src.__class__.__name__}")
    else:
        w_fake= _quantize_det(
                    w_src,
                    n_bits=8, signed=True,
                    n_frac=int(src.export_quant_info()[0]))
        dst.weight.copy_(w_fake.to(dst.weight.dtype))

    # ---- 2. Bias  (only if both sides have it) ------------------------
    if hasattr(dst, "bias") and dst.bias is not None:
        b_src = _get_src_param(src, "bias")
        if b_src is not None:
            b_fake = _quantize_det(b_src, n_bits=8, signed=True, n_frac=int(src.export_quant_info()[1]))
            dst.bias.copy_(b_fake.to(dst.bias.dtype))


def _register_quant_metadata(new_layer: nn.Module, quant_info: list) -> None:
    """Registers quantization metadata as buffers in the new layer."""
    new_layer.register_buffer("bitwidth", torch.tensor(8))
    new_layer.register_buffer("frac_weight", torch.tensor(quant_info[0]))
    new_layer.register_buffer("frac_bias", torch.tensor(quant_info[1]))
    new_layer.register_buffer("frac_act", torch.tensor(quant_info[2]))


def _handle_fused_conv_bn(target_module: FusedConvBN,) -> nn.Conv2d:
    """Handles FusedConvBN modules."""

    conv_layer = nn.Conv2d(
        in_channels=target_module.conv_mod.in_channels,
        out_channels=target_module.conv_mod.out_channels,
        kernel_size=target_module.conv_mod.kernel_size,
        stride=target_module.conv_mod.stride,
        padding=target_module.conv_mod.padding,
        dilation=target_module.conv_mod.dilation,
        groups=target_module.conv_mod.groups,
        bias=target_module.conv_mod.bias is not None,
    )
    _copy_and_quantize_params(target_module, conv_layer)
    _register_quant_metadata(conv_layer, target_module.export_quant_info())
    return conv_layer


def _handle_conv2d(target_module: nn.Conv2d) -> nn.Conv2d:
    """Handles Conv2d modules."""

    conv_layer = nn.Conv2d(
        in_channels=target_module.in_channels,
        out_channels=target_module.out_channels,
        kernel_size=target_module.kernel_size,
        stride=target_module.stride,
        padding=target_module.padding,
        dilation=target_module.dilation,
        groups=target_module.groups,
        bias=target_module.bias is not None,
    )
    _copy_and_quantize_params(target_module, conv_layer)
    _register_quant_metadata(conv_layer, target_module.export_quant_info())
    return conv_layer


def _handle_linear(target_module: nn.Linear) -> nn.Linear:
    """Handles Linear modules."""

    linear_layer = nn.Linear(
        in_features=target_module.in_features,
        out_features=target_module.out_features,
        bias=target_module.bias is not None,
    )
    _copy_and_quantize_params(target_module, linear_layer)
    _register_quant_metadata(linear_layer, target_module.export_quant_info())
    return linear_layer


def _handle_maxpool2d(target_module: nn.MaxPool2d) -> nn.MaxPool2d:
    """Handles MaxPool2d modules."""

    maxpool_layer = nn.MaxPool2d(
        kernel_size=target_module.kernel_size,
        stride=target_module.stride,
        padding=target_module.padding,
        dilation=target_module.dilation,
        return_indices=target_module.return_indices,
        ceil_mode=target_module.ceil_mode,
    )
    maxpool_layer.register_buffer(
        "frac_act", torch.tensor(target_module.export_quant_info())
    )
    return maxpool_layer


def _handle_adaptiveavgpool2d(target_module: nn.AdaptiveAvgPool2d,) -> nn.AdaptiveAvgPool2d:
    """Handles AdaptiveAvgPool2d modules."""

    adaptive_avgpool_layer = nn.AdaptiveAvgPool2d(
        output_size=target_module.output_size
    )
    adaptive_avgpool_layer.register_buffer(
        "frac_act", torch.tensor(target_module.export_quant_info())
    )
    return adaptive_avgpool_layer


def _handle_Relu(target_module: nn.ReLU) -> nn.ReLU:
    """Handles ReLU modules."""
    relu_layer = copy.deepcopy(target_module)
    return relu_layer


def _handle_QuantStub(target_module):
    """Handles ReLU modules."""
    stub = copy.deepcopy(target_module)
    return stub


def _handle_Dropout(target_module: nn.Dropout) -> nn.Dropout:
    """Handles ReLU modules."""
    stub = copy.deepcopy(target_module)
    return stub


def _handle_qelementwiseadd(target_module: QElementwiseAdd,) -> AddWithMetadata:
    """Handles QElementwiseAdd modules."""

    add_layer = AddWithMetadata()
    add_layer.register_buffer(
        "frac_act", torch.tensor(target_module.export_quant_info())
    )
    return add_layer


def _get_new_module(target_module, node):
    """Creates the appropriate new module based on the target_module type."""

    module_handlers = {
        FusedConvBN: _handle_fused_conv_bn,
        QuantizedConv2d: _handle_conv2d,
        QuantizedLinear: _handle_linear,
        QMaxPool2D: _handle_maxpool2d,
        QAdaptiveAvgPool2d: _handle_adaptiveavgpool2d,
        QElementwiseAdd: _handle_qelementwiseadd,
        nn.ReLU: _handle_Relu,
        nn.Flatten: nn.Flatten,  # No special handling needed
        QuantStubC: _handle_QuantStub,
        nn.Dropout: _handle_Dropout,
    }
    module_type = type(target_module)
    handler = module_handlers.get(module_type)
    if handler:
        return handler(target_module)
    else:
        print(f"Warning: Module type {module_type} not handled.")
        return None  # Or raise an exception if you prefer


class InferProcessor:
    def __init__(self, model: nn.Module, config: dict):
        if isinstance(model, fx.GraphModule):
            fx_model = copy.deepcopy(model)
        else:
            model = copy.deepcopy(model)
            fx_model = fx.symbolic_trace(model)
        self.fx_model = fx_model

        self.std_model: Optional[fx.GraphModule] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def convert_to_std_model(self) -> fx.GraphModule:
        """
        Transforms a QAT-trained model into an inference-ready model.

        Args:
            model (fx.GraphModule, optional): The QAT-trained model (fx.GraphModule).
                If None, uses the internal qat_model. Defaults to None.

        Returns:
            fx.GraphModule: The converted inference model.
        """

        inference_model = StandardModel()
        new_graph = fx.Graph()
        node_map = {}

        modules = dict(self.fx_model.named_modules())
        logger = logging.getLogger(self.__class__.__name__)  # Get the class logger

        for node in self.fx_model.graph.nodes:
            if node.op == "placeholder":
                new_node = new_graph.placeholder(node.name)

            elif node.op == "call_module":
                target_module = modules[node.target]
                new_module = _get_new_module(target_module, node)

                if new_module:
                    if isinstance(target_module, FusedConvBN):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, QuantizedConv2d):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, QuantizedLinear):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, QMaxPool2D):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, QAdaptiveAvgPool2d):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, nn.ReLU):
                        module_name = node.target.replace('.', '_')  # Fix the invalid naming issue
                        inference_model.layers[module_name] = new_module
                        if node.args[0] in node_map:
                            new_node = new_graph.call_module(module_name, args=(node_map[node.args[0]],))
                        else:
                            raise RuntimeError(
                                f"Node '{node.target}' is trying to use '{node.args[0]}' before it exists in the graph!")
                    elif isinstance(target_module, nn.AdaptiveAvgPool2d):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, QElementwiseAdd):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name,
                                                         args=(node_map[node.args[0]], node_map[node.args[1]]))
                    else:
                        if node.name == "QuantStub":
                            print('QuantStub skipped')

                    # logger.debug(f"Handled module: {target_module.module_name} ({type(target_module).__name__})")
                else:
                    logger.warning(
                        f"Skipping unsupported module: {node.target} ({type(target_module).__name__})"
                    )
                    continue  # Skip this node if not handled
                node_map[node] = new_node

            elif node.op == "call_function":
                if node.target == torch.flatten:
                    inference_model.layers[node.name] = nn.Flatten()
                    new_node = new_graph.call_module(
                        node.name, args=(node_map[node.args[0]],)
                    )
                elif node.target == torch.add:
                    add_layer = AddWithMetadata()
                    add_layer.register_buffer(
                        "frac_act", torch.tensor(target_module.export_quant_info())
                    )
                    inference_model.layers[target_module.module_name] = add_layer
                    new_node = new_graph.call_module(
                        target_module.module_name,
                        args=(node_map[node.args[0]], node_map[node.args[1]]),
                    )
                    node_map[node] = new_node
                else:
                    logger.warning(f"Skipping unsupported function: {node.target}")
                    continue

            elif node.op == "output":
                new_node = new_graph.output(node_map[node.args[0]])

            node_map[node] = new_node  # Keep track of mapped nodes

        # Recompile graph
        new_graph.lint()
        new_gm = fx.GraphModule(inference_model.layers, new_graph)
        self.std_model = new_gm
        return new_gm

    def export_onnx_with_layer_metadata(self, save_path):
        # Export the model to ONNX
        assert self.std_model is not None

        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            self.std_model,
            dummy_input,
            save_path,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )

        # Load the exported ONNX model
        onnx_model = onnx.load(save_path)

        # Create a dictionary to store quantization parameters based on PyTorch module names
        quant_params_by_torch_name = {}
        for name, module in self.std_model.named_modules():
            if hasattr(module, "bitwidth") or hasattr(module, "frac_w") or hasattr(module, "frac_act"):
                if name == 'fc':
                    quant_params_by_torch_name[name] = {
                        "bitwidth": getattr(module, "bitwidth", None),
                        "frac_B": getattr(module, "frac_weight", None),
                        "frac_C": getattr(module, "frac_bias", None),
                        "frac_out": getattr(module, "frac_act", None)
                    }
                else:
                    quant_params_by_torch_name[name] = {
                        "bitwidth": getattr(module, "bitwidth", None),
                        "frac_W": getattr(module, "frac_weight", None),
                        "frac_B": getattr(module, "frac_bias", None),
                        "frac_out": getattr(module, "frac_act", None)
                    }

        # Match PyTorch layers to ONNX nodes**
        matched_layers = {}
        for torch_layer_name in quant_params_by_torch_name.keys():
            for onnx_node_name in [node.name for node in onnx_model.graph.node]:
                if torch_layer_name in onnx_node_name:  # Check if ONNX node contains the PyTorch module name
                    matched_layers[onnx_node_name] = quant_params_by_torch_name[torch_layer_name]
                    break  # Stop after the first match

        # Attach quantization parameters to ONNX nodes**
        for node in onnx_model.graph.node:
            if node.name in matched_layers:
                params = matched_layers[node.name]
                for key, value in params.items():
                    if value is not None:
                        attr = onnx.helper.make_attribute(key, int(value))  # Convert value to int
                        node.attribute.append(attr)

        # Save the updated ONNX model
        onnx.save(onnx_model, save_path)

    def generate_qconfig(self):
        # Generate qconfig

        qconfig = {}
        for name, module in self.std_model.named_modules():
            if hasattr(module, "bitwidth") or hasattr(module, "frac_w") or hasattr(module, "frac_act"):
                frac_in = []
                for node in self.std_model.graph.nodes:
                    if node.target == name and node.op == "call_module":
                        for arg in node.args:
                            input_node = next((n for n in self.std_model.graph.nodes if n.name == arg.name), None)
                            while input_node:
                                if input_node.target in dict(self.std_model.named_modules()):
                                    prev_module = dict(self.std_model.named_modules())[input_node.target]
                                    if hasattr(prev_module, "frac_act"):
                                        frac_in.append(int(prev_module.frac_act))
                                        break
                                    else:
                                        input_node = next(
                                            (n for n in self.std_model.graph.nodes if n.name == input_node.args[0].name), None)
                                else:
                                    break
                        break

                qconfig[name] = {
                    "bitwidth": int(module.bitwidth) if hasattr(module, "bitwidth") else None,
                    "frac_in": frac_in,
                    "frac_w": int(module.frac_weight) if hasattr(module, "frac_weight") else None,
                    "frac_b": int(module.frac_bias) if hasattr(module, "frac_bias") else None,
                    "frac_out": int(module.frac_act) if hasattr(module, "frac_act") else None
                }
        # Append frac of the first layer which is 5 for imagenet
        # @TODO correctly get frac_n of in for a given dataset
        first_layer_name = next(
            (name for name, module in self.std_model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))),
            next(iter(self.std_model.named_modules()))[0])
        qconfig[first_layer_name] = {
            "bitwidth": 8,
            "frac_in": [5],
            "frac_w": int(dict(self.std_model.named_modules())[first_layer_name].frac_weight),
            "frac_b": int(dict(self.std_model.named_modules())[first_layer_name].frac_bias),
            "frac_out": int(dict(self.std_model.named_modules())[first_layer_name].frac_act)
        }

        self.logger.info("Successfully Generated qconfig:")

        return self._standardize_qconfig(qconfig)

    def _standardize_qconfig(self, qconfig):
        """
        Convert a qconfig dictionary into the standard format.

        Args:
            qconfig (dict): The original qconfig dictionary. Example format:
                            {
                                'conv1': {'bitwidth': 8, 'frac_in': [], 'frac_w': 9, 'frac_b': 6, 'frac_out': 5},
                                'maxpool': {'bitwidth': None, 'frac_in': [5], 'frac_w': None, 'frac_b': None, 'frac_out': 5},
                                ...
                            }

        Returns:
            dict: The standardized dictionary in the required format.
                  Example:
                  {
                      "x": {"out": 5},
                      "conv1": {"weight": 8, "bias": 6, "in": 5, "out": 5},
                      ...
                  }
        """
        standardized = {}

        # Extracting "x" settings from the input frac of the first layer
        first_layer = next(iter(qconfig))
        first_frac_in = qconfig.get(first_layer, {}).get("frac_in", [])
        standardized["x"] = {"out": first_frac_in[0] if first_frac_in else None}

        for layer, params in qconfig.items():
            layer_config = {}
            if "frac_w" in params and params["frac_w"] is not None:
                layer_config["weight"] = params["frac_w"]
            if "frac_b" in params and params["frac_b"] is not None:
                layer_config["bias"] = params["frac_b"]
            if "frac_in" in params and params["frac_in"]:
                layer_config["in"] = params["frac_in"]
            if "frac_out" in params and params["frac_out"] is not None:
                layer_config["out"] = params["frac_out"]

            if layer_config:  # Only add layers with valid configuration
                standardized[layer] = layer_config

        return standardized

    def export_weights_to_file(self, layer_order, output_filename="weights.data", n_bits_out=8,
                               pad_last_layer_dim_to=1024):
        """
        Extracts weights and biases, quantizes them, optionally pads the last layer,
        rearranges them to HWCM format, and saves to a binary file.
        """
        self.logger.info(f"Starting extraction of quantized parameters to '{output_filename}' (n_bits={n_bits_out}).")
        if pad_last_layer_dim_to:
            self.logger.info(f"Padding output dimension of the last Conv/Linear layer to {pad_last_layer_dim_to}.")

        all_quantized_numpy_arrays = []

        # Determine the last layer for potential padding
        last_conv_linear_layer_name = None
        for name in reversed(layer_order):
            if isinstance(self.std_model.get_submodule(name), (nn.Conv2d, nn.Linear)):
                last_conv_linear_layer_name = name
                break

        if last_conv_linear_layer_name:
            self.logger.info(
                f"Identified last Conv/Linear layer for potential padding: '{last_conv_linear_layer_name}'")
        else:
            self.logger.warning("No Conv/Linear layer found in the specified layer order. Padding will not be applied.")

        for name in layer_order:
            module = self.std_model.get_submodule(name)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.logger.debug(f"Processing layer: {name} ({type(module).__name__})")
                is_last_layer_to_pad = (name == last_conv_linear_layer_name and pad_last_layer_dim_to is not None)

                # Process Weights
                if hasattr(module, 'weight') and module.weight is not None:
                    if not hasattr(module, 'frac_weight'):
                        self.logger.warning(f"Layer '{name}' has weights but no 'frac_weight'. Skipping weights.")
                    else:
                        frac_w = int(module.frac_weight)
                        int_weights_tensor = to_int_tensor(
                            module.weight.data.detach().clone(),
                            signed=True, n_bits=n_bits_out, n_frac=frac_w
                        )
                        if int_weights_tensor is not None:
                            # Transpose weights to HWCM format
                            if isinstance(module, nn.Conv2d):
                                int_weights_tensor = int_weights_tensor.permute(2, 3, 1, 0)  # NCHW to HWCM

                            # Apply Padding if this is the last layer and padding is enabled
                            if is_last_layer_to_pad:
                                current_dim0_size = int_weights_tensor.shape[0]
                                if current_dim0_size < pad_last_layer_dim_to:
                                    if isinstance(module, nn.Linear):  # Weights shape (out_features, in_features)
                                        pad_rows = pad_last_layer_dim_to - current_dim0_size
                                        padding = torch.zeros((pad_rows, int_weights_tensor.shape[1]),
                                                              dtype=int_weights_tensor.dtype,
                                                              device=int_weights_tensor.device)
                                        int_weights_tensor = torch.cat((int_weights_tensor, padding), dim=0)
                                        self.logger.info(
                                            f"  Padded weights of Linear layer '{name}' (dim 0) from {current_dim0_size} to {pad_last_layer_dim_to}.")
                                    elif isinstance(module,
                                                    nn.Conv2d):  # Weights shape (out_channels, in_channels, kH, kW)
                                        pad_channels = pad_last_layer_dim_to - current_dim0_size
                                        padding_shape = (pad_channels,) + int_weights_tensor.shape[1:]
                                        padding = torch.zeros(padding_shape,
                                                              dtype=int_weights_tensor.dtype,
                                                              device=int_weights_tensor.device)
                                        int_weights_tensor = torch.cat((int_weights_tensor, padding), dim=0)
                                        self.logger.info(
                                            f"  Padded weights of Conv2d layer '{name}' (dim 0: out_channels) from {current_dim0_size} to {pad_last_layer_dim_to}.")
                                elif current_dim0_size > pad_last_layer_dim_to:
                                    self.logger.warning(
                                        f"  Weights of last layer '{name}' dim 0 ({current_dim0_size}) is already > target pad size ({pad_last_layer_dim_to}). No padding done.")

                            numpy_weights = int_weights_tensor.cpu().numpy().flatten()
                            all_quantized_numpy_arrays.append(numpy_weights)
                            self.logger.debug(
                                f"  Quantized weights for '{name}' added (elements: {numpy_weights.size})")

                # Process Biases
                if hasattr(module, 'bias') and module.bias is not None:
                    if not hasattr(module, 'frac_bias'):
                        self.logger.warning(f"Layer '{name}' has bias but no 'frac_bias'. Skipping bias.")
                    else:
                        frac_b = int(module.frac_bias)
                        int_bias_tensor = to_int_tensor(
                            module.bias.data.detach().clone(),
                            signed=True, n_bits=n_bits_out, n_frac=frac_b
                        )
                        if int_bias_tensor is not None:
                            # Apply Padding if this is the last layer and padding is enabled
                            if is_last_layer_to_pad:
                                current_bias_size = int_bias_tensor.shape[0]
                                if current_bias_size < pad_last_layer_dim_to:
                                    pad_elements = pad_last_layer_dim_to - current_bias_size
                                    padding = torch.zeros(pad_elements,
                                                          dtype=int_bias_tensor.dtype,
                                                          device=int_bias_tensor.device)
                                    int_bias_tensor = torch.cat((int_bias_tensor, padding), dim=0)
                                    self.logger.info(
                                        f"  Padded bias of layer '{name}' from {current_bias_size} to {pad_last_layer_dim_to} elements.")
                                elif current_bias_size > pad_last_layer_dim_to:
                                    self.logger.warning(
                                        f"  Bias of last layer '{name}' ({current_bias_size}) is already > target pad size ({pad_last_layer_dim_to}). No padding done.")

                            numpy_bias = int_bias_tensor.cpu().numpy().flatten()
                            all_quantized_numpy_arrays.append(numpy_bias)
                            self.logger.debug(f"  Quantized bias for '{name}' added (elements: {numpy_bias.size})")

        if not all_quantized_numpy_arrays:
            self.logger.warning("No parameters collected. Output file will be empty or not created.")
            try:
                with open(output_filename, 'wb') as f:
                    pass
                self.logger.info(f"Created empty file '{output_filename}'.")
            except IOError as e:
                self.logger.error(f"Failed to create empty file: {e}")
            return

        try:
            final_numpy_array = np.concatenate(all_quantized_numpy_arrays)
        except ValueError as e:
            self.logger.error(f"NumPy concatenation failed: {e}")
            for i, arr in enumerate(all_quantized_numpy_arrays):
                self.logger.error(f"Arr {i}: shape={arr.shape}, dtype={arr.dtype}")
            raise

        self.logger.info(
            f"Concatenated parameters. Total elements: {final_numpy_array.size}, dtype: {final_numpy_array.dtype}")

        try:
            final_numpy_array.tofile(output_filename)
            self.logger.info(f"Successfully wrote {final_numpy_array.size} params to '{output_filename}'.")
        except IOError as e:
            self.logger.error(f"Failed to write to file '{output_filename}': {e}")
            raise

    def extract_and_subset_layer_parameters(
            self,
            layer_name: str,
            output_filename: str,
            target_weight_shape: tuple = None,  # e.g., (16, original_C_in, 32, 32) or (16, original_C_in)
            n_bits_out: int = 8
    ):
        """
         Extracts weights and (if present) bias from a specific layer,
         optionally takes a subset, quantizes them, saves to file, and
         returns the original floating-point (potentially subsetted) tensors.

         Args:
             layer_name (str): The name of the layer.
             output_filename (str): Path to save the quantized parameters.
             target_weight_shape (tuple, optional): Desired shape for the subset of weights.
             n_bits_out (int): Number of bits for output quantized integers.

         Returns:
             tuple[torch.Tensor | None, torch.Tensor | None]:
                 A tuple containing:
                 - The original floating-point (potentially subsetted) weight tensor. None if no weights.
                 - The original floating-point (potentially subsetted) bias tensor. None if no bias.
         """
        self.logger.info(f"Extracting parameters from layer '{layer_name}' to '{output_filename}'.")
        if target_weight_shape:
            self.logger.info(f"  Target subset weight shape: {target_weight_shape}")

        module_to_extract = dict(self.std_model.named_modules()).get(layer_name)

        if module_to_extract is None:
            self.logger.error(f"Layer '{layer_name}' not found.")
            raise ValueError(f"Layer '{layer_name}' not found.")
        if not isinstance(module_to_extract, (nn.Conv2d, nn.Linear)):
            self.logger.error(f"Layer '{layer_name}' is {type(module_to_extract).__name__}, not Conv2d/Linear.")
            raise TypeError(f"Layer '{layer_name}' is not a Conv2d or Linear layer.")

        quantized_params_for_file = []
        return_fp_weight: torch.Tensor | None = None
        return_fp_bias: torch.Tensor | None = None

        # --- Process Weights ---
        if hasattr(module_to_extract, 'weight') and module_to_extract.weight is not None:
            original_fp_weights = module_to_extract.weight.data.detach().clone()
            self.logger.debug(f"  Original FP weight shape: {original_fp_weights.shape}")

            # Determine the final FP weights (full or subset) to be returned and quantized
            current_fp_weights_to_process = original_fp_weights
            if target_weight_shape:
                if len(target_weight_shape) != original_fp_weights.ndim:
                    msg = (f"Target weight shape {target_weight_shape} (rank {len(target_weight_shape)}) "
                           f"mismatches original rank {original_fp_weights.ndim}.")
                    self.logger.error(msg)
                    raise ValueError(msg)

                slicing_indices = []
                valid_subset = True
                for i, dim_size in enumerate(target_weight_shape):
                    if not (0 < dim_size <= original_fp_weights.shape[i]):  # dim_size must be positive
                        msg = (f"Target dim {i} size {dim_size} is invalid or exceeds "
                               f"original size {original_fp_weights.shape[i]}.")
                        self.logger.error(msg)
                        valid_subset = False
                        break
                    slicing_indices.append(slice(0, dim_size))

                if not valid_subset:
                    raise ValueError("Invalid target_weight_shape.")

                current_fp_weights_to_process = original_fp_weights[tuple(slicing_indices)]
                self.logger.info(f"  Subsetted FP weights to shape: {current_fp_weights_to_process.shape}")

            return_fp_weight = current_fp_weights_to_process  # This is the float tensor to return

            # Now quantize these (potentially subsetted) FP weights
            frac_w = int(getattr(module_to_extract, 'frac_weight', 0))  # Default frac to 0 if not found
            if not hasattr(module_to_extract, 'frac_weight'):
                self.logger.warning(f"Layer '{layer_name}' weights missing 'frac_weight', using default 0.")

            quantized_weights = to_int_tensor(
                return_fp_weight, signed=True, n_bits=n_bits_out, n_frac=frac_w
            )
            if quantized_weights is not None:
                quantized_params_for_file.append(quantized_weights.cpu().numpy().flatten())
                self.logger.debug(f"  Quantized weights added (elements: {quantized_params_for_file[-1].size})")
            else:
                self.logger.error(f"Quantization of weights for '{layer_name}' resulted in None.")
        else:
            self.logger.warning(f"Layer '{layer_name}' has no 'weight' or it's None.")

        # --- Process Biases ---
        if hasattr(module_to_extract, 'bias') and module_to_extract.bias is not None:
            original_fp_bias = module_to_extract.bias.data.detach().clone()
            self.logger.debug(f"  Original FP bias shape: {original_fp_bias.shape}")

            current_fp_bias_to_process = original_fp_bias
            # Bias subsetting is dependent on the first dimension of the processed weights
            if return_fp_weight is not None and target_weight_shape is not None:  # Implying weights were subsetted
                target_bias_len_from_weights = return_fp_weight.shape[0]
                if target_bias_len_from_weights < original_fp_bias.shape[0]:
                    if original_fp_bias.ndim == 1:
                        current_fp_bias_to_process = original_fp_bias[:target_bias_len_from_weights]
                        self.logger.info(f"  Subsetted FP bias to length: {current_fp_bias_to_process.shape[0]}")
                    else:
                        self.logger.warning("  Bias is not 1D. Auto-subsetting based on weights not applied robustly.")
                elif target_bias_len_from_weights > original_fp_bias.shape[0]:
                    self.logger.warning(f"  Target bias length from weights ({target_bias_len_from_weights}) "
                                        f"> original bias length ({original_fp_bias.shape[0]}). Using original bias length.")

            return_fp_bias = current_fp_bias_to_process  # This is the float tensor to return

            # Now quantize these (potentially subsetted) FP biases
            frac_b = int(getattr(module_to_extract, 'frac_bias', 0))  # Default frac to 0 if not found
            if not hasattr(module_to_extract, 'frac_bias'):
                self.logger.warning(f"Layer '{layer_name}' bias missing 'frac_bias', using default 0.")

            quantized_bias = to_int_tensor(
                return_fp_bias, signed=True, n_bits=n_bits_out, n_frac=frac_b
            )
            if quantized_bias is not None:
                quantized_params_for_file.append(quantized_bias.cpu().numpy().flatten())
                self.logger.debug(f"  Quantized bias added (elements: {quantized_params_for_file[-1].size})")
            else:
                self.logger.error(f"Quantization of bias for '{layer_name}' resulted in None.")
        elif isinstance(module_to_extract, (nn.Conv2d, nn.Linear)):
            self.logger.debug(f"  Layer '{layer_name}' has no 'bias' or it's None.")

        # --- Save to File ---
        if not quantized_params_for_file:
            self.logger.warning(f"No parameters (weights or bias) processed for file output from '{layer_name}'.")
            try:
                with open(output_filename, 'wb') as f:
                    pass  # Create empty file
                self.logger.info(f"Created empty file '{output_filename}'.")
            except IOError as e:
                self.logger.error(f"Failed to create empty file: {e}")
        else:
            try:
                final_numpy_array = np.concatenate(quantized_params_for_file)
                self.logger.info(
                    f"Concatenated quantized params for '{layer_name}'. Total elements for file: {final_numpy_array.size}")
                final_numpy_array.tofile(output_filename)
                self.logger.info(
                    f"Successfully wrote {final_numpy_array.size} quantized params from '{layer_name}' to '{output_filename}'.")
            except ValueError as e:
                self.logger.error(f"NumPy concatenation failed for '{layer_name}': {e}")
                raise
            except IOError as e:
                self.logger.error(f"Failed to write to file '{output_filename}': {e}")
                raise

        return return_fp_weight, return_fp_bias, frac_w, frac_b