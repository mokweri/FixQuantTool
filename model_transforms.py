from typing import Dict, Any, Tuple
import torch.fx as fx
import torch
import torch.nn as nn
import copy
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
from torch.fx.immutable_collections import immutable_list
from torch.nn.utils.fusion import fuse_conv_bn_weights
from quantization.qmodules import QuantizedLinear, QuantizedConv2d, QMaxPool2D, QAdaptiveAvgPool2d, QElementwiseAdd
from quantization.conv_fused import QuantizedConvBatchNorm2d
from quantization.qmodules import QuantizedConv2d
from quantization.fix_ops import FixedPointQuantizer, QuantStubF


def create_quantizable_model(model, qconfig):
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    assert qconfig, 'qconfig must be provided for the model quantizer configuration'

    modules_patterns = {
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d)
    }

    # Fuse conv with BN according to matching pattern
    for pattern in modules_patterns:
        for node in fx_model.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # conv that has multiple consumers is not fused
                    print('Replacing Conv2d with QuantizedConv2d')
                    conv = modules[node.args[0].target]
                    newConv = QuantizedConv2d.from_float(conv, qconfig)
                    newConv.module_name = str(node.args[0]).strip()
                    replace_node_module(node.args[0], modules, newConv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
                    continue
                # Fuse
                print('Fusing Conv2d with BatchNorm2d --> QuantizedConvBatchNorm2d')
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_cnv_bn = QuantizedConvBatchNorm2d.from_float(conv, bn, qconfig)
                fused_cnv_bn.module_name = str(node.args[0]).strip()
                replace_node_module(node.args[0], modules, fused_cnv_bn)
                node.replace_all_uses_with(node.args[0])
                fx_model.graph.erase_node(node)
                fx_model.graph.lint()

    fx_model.recompile()
    fx_model.delete_all_unused_submodules()

    modules = dict(fx_model.named_modules())
    # check and replace other modules
    for node in fx_model.graph.nodes:
        if node.target == 'x':
            with fx_model.graph.inserting_after(node):
                quant_stub = fx_model.graph.call_function(QuantStubF, args=(node,))
                node.replace_all_uses_with(quant_stub)
                quant_stub.name = str(node.name).strip()
                node.name = "q_input"
                quant_stub.replace_input_with(quant_stub, node)

        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, nn.Conv2d):
                if type(target_module).__name__ == 'QuantizedConvBatchNorm2d':
                    # to skip the fused Conv2d, QuantizedConvBatchNorm2d that already was replaced
                    continue
                print('Replacing a Conv2d layer with QuantizedConv2d')
                newConv = QuantizedConv2d.from_float(target_module, qconfig)
                newConv.module_name = str(node.name).strip()
                replace_node_module(node, modules, newConv)
            elif isinstance(target_module, nn.Linear):
                print('Replacing a linear layer with QLinear')
                newQlinear = QuantizedLinear.from_float(target_module, qconfig)
                newQlinear.module_name = str(node.name).strip()
                replace_node_module(node, modules, newQlinear)
            elif isinstance(target_module, nn.MaxPool2d):
                print('Replacing nn.MaxPool2d layer with QMaxPool')
                newMaxPool = QMaxPool2D.from_float(target_module, qconfig)
                newMaxPool.module_name = str(node.name).strip()
                replace_node_module(node, modules, newMaxPool)
            elif isinstance(target_module, nn.AdaptiveAvgPool2d):
                print('Replacing nn.AdaptiveAvgPool2d layer with QAdaptiveAvgPool2d')
                newPool = QAdaptiveAvgPool2d.from_float(target_module, qconfig)
                newPool.module_name = str(node.name).strip()
                replace_node_module(node, modules, newPool)
        elif node.op == "call_function":
            if node.target.__name__ == "add":
                with fx_model.graph.inserting_after(node):
                    print('Replacing function add with QAdd')
                    QAdd = QElementwiseAdd(qconfig)
                    QAdd.module_name = str(node.name).strip()
                    QAdd_node = fx_model.graph.call_function(QAdd.forward, args=(node.args[0], node.args[1]))
                    QAdd_node.name = str(node.name).strip()
                    node.replace_all_uses_with(QAdd_node)
                fx_model.graph.erase_node(node)

        fx_model.graph.lint()
    fx_model.recompile()
    return fx_model


def create_quantizable_model1(model):
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    modules_patterns = {
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d)
    }

    # Fuse conv with BN according to matching pattern
    for pattern in modules_patterns:
        for node in fx_model.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # conv that has multiple consumers is not fused
                    conv = modules[node.args[0].target]
                    quantizer = FixedPointQuantizer(bitwidth=8)
                    newConv = QuantizedConv2d.from_float(conv, quantizer)
                    # model.add_submodule(node.target, newConv)
                    replace_node_module(node.args[0], modules, newConv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
                    continue
                # Fuse
                print('Fusing Conv2d with BatchNorm2d --> QuantizedConvBatchNorm2d')
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                quantizer = FixedPointQuantizer(bitwidth=8)
                fused_cnv_bn = QuantizedConvBatchNorm2d.from_float(conv, bn, quantizer)
                replace_node_module(node.args[0], modules, fused_cnv_bn)
                node.replace_all_uses_with(node.args[0])
                fx_model.graph.erase_node(node)
                fx_model.graph.lint()

    fx_model.recompile()
    fx_model.delete_all_unused_submodules()

    modules = dict(fx_model.named_modules())
    # check and replace other modules
    for node in fx_model.graph.nodes:
        if node.target == 'x':
            with fx_model.graph.inserting_after(node):
                quant_stub = fx_model.graph.call_function(QuantStubF, args=(node,))
                node.replace_all_uses_with(quant_stub)
                quant_stub.name = node.name
                node.name = "q_input"
                quant_stub.replace_input_with(quant_stub, node)

        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, nn.Conv2d):
                if type(target_module).__name__ == 'QuantizedConvBatchNorm2d':
                    # to skip the fused Conv2d, QuantizedConvBatchNorm2d that already was replaced
                    continue
                print('Replacing a Conv2d layer with QuantizedConv2d')
                quantizer = FixedPointQuantizer(bitwidth=8)
                newConv = QuantizedConv2d.from_float(target_module, quantizer)
                # @TODO newConv.module_name()
                replace_node_module(node, modules, newConv)
            elif isinstance(target_module, nn.Linear):
                print('Replacing a linear layer with QLinear')
                quantizer = FixedPointQuantizer(bitwidth=8)
                newQlinear = QuantizedLinear.from_float(target_module, quantizer)
                replace_node_module(node, modules, newQlinear)
                # node.replace_all_uses_with(node)
                # fx_model.graph.erase_node(node.args[0])
            elif isinstance(target_module, nn.MaxPool2d):
                print('Replacing nn.MaxPool2d layer with QMaxPool')
                quantizer = FixedPointQuantizer(bitwidth=8)
                newMaxPool = QMaxPool2D.from_float(target_module, quantizer)
                replace_node_module(node, modules, newMaxPool)
            elif isinstance(target_module, nn.AdaptiveAvgPool2d):
                print('Replacing nn.AdaptiveAvgPool2d layer with QAdaptiveAvgPool2d')
                quantizer = FixedPointQuantizer(bitwidth=8)
                newMaxPool = QAdaptiveAvgPool2d.from_float(target_module, quantizer)
                replace_node_module(node, modules, newMaxPool)
        elif node.op == "call_function":
            if node.target.__name__ == "add":
                with fx_model.graph.inserting_after(node):
                    print('Replacing function add with QAdd')
                    quantizer = FixedPointQuantizer(bitwidth=8)
                    QAdd = QElementwiseAdd(quantizer)
                    QAdd_node = fx_model.graph.call_function(QAdd.forward, args=(node.args[0], node.args[1]))
                    QAdd_node.name = node.name
                    node.replace_all_uses_with(QAdd_node)
                fx_model.graph.erase_node(node)

        fx_model.graph.lint()
    fx_model.recompile()
    return fx_model


def fuse_conv_bn(m: torch.nn.Module):
    model = copy.deepcopy(m)
    gm: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())

    modules_patterns = {
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d)
    }

    for pattern in modules_patterns:
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # conv that has multiple consumers is ignored
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]

                bn_running_mean = bn.running_mean.data
                bn_running_var = bn.running_var.data
                bn_weight = bn.weight.data
                bn_bias = bn.bias.data
                bn_eps = bn.eps

                fused_conv = copy.deepcopy(conv)
                fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
                    fused_conv.weight,
                    fused_conv.bias,
                    bn_running_mean,
                    bn_running_var,
                    bn_eps,
                    bn_weight,
                    bn_bias,
                )

                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                gm.graph.lint()
    gm.recompile()
    gm.delete_all_unused_submodules()
    return gm


def create_qconfig(model, valid_loader=None, bitwidth=8):
    model = fuse_conv_bn(model)

    qconfig = {}
    node_input = {}
    modules = dict(model.named_modules())

    # Initialize the qconfig
    for node in model.graph.nodes:
        qconfig[node.name] = dict()
        node_input[node.name] = dict()

    # Go over the model - perform PTQ
    for node in model.graph.nodes:
        if node.target == 'x':
            quantizer = FixedPointQuantizer(bitwidth)
            images, classes = next(iter(valid_loader))
            q_image = quantizer.get_weight_quantizer('out')(images)
            qconfig[node.name][node.next.name] = int(quantizer.get_frac_out.item())
            quantizer = None
            for user in node.users:
                node_input[node.name][user.name] = q_image

        elif node.op == 'call_module':
            if hasattr(modules[node.target], "weight"):
                if isinstance(modules[node.target], nn.Conv2d) or isinstance(modules[node.target], nn.Linear):
                    print("Generating QConfig::Layer is a conv - Conv2d or Linear")
                    quantizer = FixedPointQuantizer(bitwidth)
                    q_weight = quantizer.get_weight_quantizer('weight')(modules[node.target].weight.data)
                    qconfig[node.name]["weight"] = int(quantizer.get_frac_w.item())
                    if modules[node.target].bias !=None:
                        q_bias = quantizer.get_weight_quantizer('bias')(modules[node.target].bias.data)
                        qconfig[node.name]["bias"] = int(quantizer.get_frac_b.item())
                    else:
                        q_bias = None
                        qconfig[node.name]["bias"] = int(8)
                    # forward
                    with torch.no_grad():
                        modules[node.target].weight.copy_(q_weight)
                        if modules[node.target].bias != None:
                            modules[node.target].bias.copy_(q_bias)

                    act_i = modules[node.target](node_input[node.args[0].name][node.name])
                    q_act = quantizer.get_weight_quantizer('out')(act_i)
                    for ar in node.args:
                        qconfig[node.name][ar.name] = qconfig[ar.name][node.name]
                    for no in node.users:
                        qconfig[node.name][no.name] = int(quantizer.get_frac_out.item())
                    quantizer = None
                    # prepare input for next layer
                    for user in node.users:
                        node_input[node.name][user.name] = q_act
                else:
                    print("Generating QConfig::Layer has has weight but not a conv - eg bn")
                    # forward
                    quantizer = FixedPointQuantizer(bitwidth)
                    act_i = modules[node.target](node_input[node.args[0].name][node.name])
                    q_act = quantizer.get_weight_quantizer('out')(act_i)
                    for ar in node.args:
                        qconfig[node.name][ar.name] = qconfig[ar.name][node.name]
                    for no in node.users:
                        qconfig[node.name][no.name] = int(quantizer.get_frac_out.item())
                    quantizer = None
                    # prepare input for next layer
                    for user in node.users:
                        node_input[node.name][user.name] = q_act
            else:
                print("Generating QConfig::Layer has no weight - eg maxpool")
                for ar in node.args:
                    if type(ar) != int and type(ar) != tuple and type(ar) != immutable_list:
                        qconfig[node.name][ar.name] = qconfig[ar.name][node.name]
                quantizer = FixedPointQuantizer(bitwidth)
                act_i = modules[node.target](node_input[node.args[0].name][node.name])
                q_act = quantizer.get_weight_quantizer('out')(act_i)
                for no in node.users:
                    qconfig[node.name][no.name] = int(quantizer.get_frac_out.item())
                quantizer = None
                # prepare input for next layer
                for user in node.users:
                    node_input[node.name][user.name] = q_act
        elif node.op == 'call_function':
            if node.target.__name__ == "add":
                print("Generating QConfig::Layer is an add function")
                act_i = node.target(node_input[node.args[0].name][node.name], node_input[node.args[1].name][node.name])
            elif node.target.__name__ == "flatten":
                print("Generating QConfig::Layer is a flatten function")
                act_i = node.target(act_i, start_dim=1)
            else:
                print("Generating QConfig::Layer is a call function")
                act_i = node.target(act_i)
            quantizer = FixedPointQuantizer(bitwidth)
            q_act = quantizer.get_weight_quantizer('out')(act_i)
            for ar in node.args:
                if type(ar) != int and type(ar) != tuple and type(ar) != immutable_list:
                    qconfig[node.name][ar.name] = qconfig[ar.name][node.name]
            for no in node.users:
                if "flatten" in node.name:
                    qconfig[node.name][no.name] = qconfig[node.prev.name][node.name]
                else:
                    qconfig[node.name][no.name] = int(quantizer.get_frac_out.item())
            quantizer = None
            # prepare input for next layer
            for user in node.users:
                node_input[node.name][user.name] = q_act
        else:
            print("Generating QConfig::Layer is not a conv mod, not a call functiom --")
            for ar in node.args:
                if type(ar) != int and type(ar) != tuple and type(ar) != immutable_list:
                    qconfig[node.name][ar.name] = qconfig[ar.name][node.name]
            for no in node.users:
                qconfig[node.name][no.name] = qconfig[node.prev.name][node.name]
    print("Done!! QConfig Generated!!.....")
    return qconfig


def standardize_qconfig(qconfig):
    standardized_qconfig = {}

    # Iterate through each module's qconfig (outer key)
    for module_name, inner_dict in qconfig.items():
        # Separate weights and biases from other items in the dict
        standardized_inner_dict = {}
        weights_biases = {}
        other_items = {}

        # Step 1: Separate weight, bias, and other items
        for key, value in inner_dict.items():
            if key in ["weight", "bias"]:
                weights_biases[key] = value
            else:
                other_items[key] = value

        # Special case for modules with 'add' in the name
        if "add" in module_name:
            # Ensure there are exactly three items for "add" modules
            other_keys = list(other_items.keys())
            if len(other_keys) >= 2:  # Need at least 2 inputs for add operation
                standardized_inner_dict["in1"] = other_items[other_keys[0]]
                standardized_inner_dict["in2"] = other_items[other_keys[1]]
                # If there's a third item, treat it as "out"
                if len(other_keys) > 2:
                    standardized_inner_dict["out"] = other_items[other_keys[2]]
                else:  # Use a default value for "out" if only two items are available
                    standardized_inner_dict["out"] = other_items[other_keys[1]]
        else:
            # Handle standard modules (same as the previous approach)
            other_keys = list(other_items.keys())

            if len(other_keys) == 1:  # If there's only one key, it's the "out"
                standardized_inner_dict["out"] = other_items[other_keys[0]]
            elif len(other_keys) >= 2:
                # The first key is considered the "in"
                standardized_inner_dict["in"] = other_items[other_keys[0]]

                # Check if the remaining keys have the same value
                remaining_items = {key: other_items[key] for key in other_keys[1:]}
                unique_values = set(remaining_items.values())

                if len(unique_values) == 1:  # If all remaining values are the same, group them as "out"
                    standardized_inner_dict["out"] = unique_values.pop()
                else:  # If not, add each one individually as a separate entry
                    for key, value in remaining_items.items():
                        standardized_inner_dict[key] = value

        # Step 2: Create the final ordered dictionary with weight, bias, in, and out
        ordered_inner_dict = {}

        # Add weight and bias if they exist, in the specified order
        if "weight" in weights_biases:
            ordered_inner_dict["weight"] = weights_biases["weight"]
        if "bias" in weights_biases:
            ordered_inner_dict["bias"] = weights_biases["bias"]

        # Add "in" and "out" or "in1", "in2", and "out" entries, ensuring they are added after weight and bias
        for key in ["in1", "in2", "in", "out"]:
            if key in standardized_inner_dict:
                ordered_inner_dict[key] = standardized_inner_dict[key]

        # Update the standardized qconfig with the new inner dict in the specified order
        standardized_qconfig[module_name] = ordered_inner_dict

    return standardized_qconfig

