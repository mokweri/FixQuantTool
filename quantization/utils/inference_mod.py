import torch
import torch.fx as fx
import torch.nn as nn
import copy
import onnx
import logging
from typing import Dict, Any

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


def _copy_and_quantize_params(
    target_module: nn.Module, new_layer: nn.Module
) -> None:
    """Copies weights/bias from target_module to new_layer and quantizes them."""
    if hasattr(target_module, "weight") and target_module.weight is not None:

        new_layer.weight.data.copy_(
            fake_quantize_tensor(
                target_module.weight.data,
                signed=True,
                n_bits=8,
                n_frac=int(target_module.export_quant_info()[0]),
            )
        )

    if hasattr(target_module, "bias") and target_module.bias is not None:
        new_layer.bias.data.copy_(
            fake_quantize_tensor(
                target_module.bias.data,
                signed=True,
                n_bits=8,
                n_frac=int(target_module.export_quant_info()[1]),
            )
        )

def _copy_and_quantize_fused_params(
    target_module: nn.Module, new_layer: nn.Module
) -> None:
    """Copies weights/bias from target_module to new_layer and quantizes them."""
    """This function is used for fusedConvBN layers."""
    if hasattr(target_module, "weight") and target_module.weight is not None:

        new_layer.weight.data.copy_(
            fake_quantize_tensor(
                target_module.conv_mod.weight.data,
                signed=True,
                n_bits=8,
                n_frac=int(target_module.export_quant_info()[0]),
            )
        )

    if hasattr(target_module, "bias") and target_module.bias is not None:
        new_layer.bias.data.copy_(
            fake_quantize_tensor(
                target_module.conv_mod.bias.data,
                signed=True,
                n_bits=8,
                n_frac=int(target_module.export_quant_info()[1]),
            )
        )


def _register_quant_metadata(new_layer: nn.Module, quant_info: list) -> None:
    """Registers quantization metadata as buffers in the new layer."""
    new_layer.register_buffer("bitwidth", torch.tensor(8))
    new_layer.register_buffer("frac_weight", torch.tensor(quant_info[0]))
    new_layer.register_buffer("frac_bias", torch.tensor(quant_info[1]))
    new_layer.register_buffer("frac_act", torch.tensor(quant_info[2]))


def _handle_fused_conv_bn(
    target_module: FusedConvBN,
) -> nn.Conv2d:
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
    _copy_and_quantize_fused_params(target_module, conv_layer)
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


def _handle_adaptiveavgpool2d(
    target_module: nn.AdaptiveAvgPool2d,
) -> nn.AdaptiveAvgPool2d:
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

def _handle_qelementwiseadd(
    target_module: QElementwiseAdd,
) -> AddWithMetadata:
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


    def convert_to_inference_model(self) -> fx.GraphModule:
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
                    elif isinstance(target_module, nn.Conv2d):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, nn.Linear):
                        inference_model.layers[target_module.module_name] = new_module
                        new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                    elif isinstance(target_module, nn.MaxPool2d):
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
        return new_gm


def export_onnx(model: torch.nn.Module, filename: str, dynamic_axes: bool = False):
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export.
        filename: The name of the output ONNX file.
        dynamic_axes: Whether to use dynamic axes for input/output.
    """
    dummy_input = torch.randn(
        1, 3, 224, 224
    )  # Example input, adjust as needed
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        opset_version=11,  # Choose an appropriate opset version
        export_params=True,  # Include model parameters
        do_constant_folding=True,  # Optimize constants
        input_names=["input"],  # Name the input tensor
        output_names=["output"],  # Name the output tensor
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        if dynamic_axes
        else None,  # Set dynamic axes if needed
    )
    print(f"Model exported to ONNX as '{filename}'")


def export_onnx_with_layer_metadata(model: torch.nn.Module, filename: str):
    """
    Exports a PyTorch model to ONNX with custom operator and metadata
    (This is a placeholder; actual metadata embedding in ONNX is complex)

    Args:
        model: The PyTorch model.
        filename: The output ONNX filename.
    """

    export_onnx(model, filename)  # For now, just do a standard export
    print(
        "Warning: export_onnx_with_layer_metadata is a placeholder. "
        "ONNX metadata embedding requires further implementation."
    )
