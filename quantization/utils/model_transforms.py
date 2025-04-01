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
from quantization.fix_ops import FixedPointQuantizer, QuantStubF, QuantStubI, QuantStubE
from quantization.FxP_modules2 import *
from quantization.utils.inference_model import generate_qconfig, AddWithMetadata


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

    print('=' * 50)
    print('> Preparing Quantized Model...'.center(50))
    print('=' * 50)

    # Fuse conv with BN according to matching pattern
    for pattern in modules_patterns:
        for node in fx_model.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # conv that has multiple consumers is not fused
                    print('{}: Replacing Conv2d with QuantizedConv2d'.format(str(node.args[0])))
                    conv = modules[node.args[0].target]
                    newConv = QuantizedConv2d.from_float(conv, qconfig)
                    newConv.module_name = str(node.args[0]).strip()
                    newConv.quantize_module()
                    replace_node_module(node.args[0], modules, newConv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
                    continue
                # Fuse
                print('{}: Fusing Conv2d with BatchNorm2d --> QuantizedConvBatchNorm2d'.format(str(node.args[0])))
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_cnv_bn = QuantizedConvBatchNorm2d.from_float(conv, bn, qconfig)
                fused_cnv_bn.module_name = str(node.args[0]).strip()
                fused_cnv_bn.quantize_module()
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
                print('{}: Replacing a Conv2d layer with QuantizedConv2d'.format(str(node.name)))
                newConv = QuantizedConv2d.from_float(target_module, qconfig)
                newConv.module_name = str(node.name).strip()
                newConv.quantize_module()
                replace_node_module(node, modules, newConv)
            elif isinstance(target_module, nn.Linear):
                print('{}: Replacing a linear layer with QLinear'.format(str(node.name)))
                newQlinear = QuantizedLinear.from_float(target_module, qconfig)
                newQlinear.module_name = str(node.name).strip()
                newQlinear.quantize_module()
                replace_node_module(node, modules, newQlinear)
            elif isinstance(target_module, nn.MaxPool2d):
                print('{}: Replacing nn.MaxPool2d layer with QMaxPool'.format(str(node.name)))
                newMaxPool = QMaxPool2D.from_float(target_module, qconfig)
                newMaxPool.module_name = str(node.name).strip()
                replace_node_module(node, modules, newMaxPool)
            elif isinstance(target_module, nn.AdaptiveAvgPool2d):
                print('{}: Replacing nn.AdaptiveAvgPool2d layer with QAdaptiveAvgPool2d'.format(str(node.name)))
                newPool = QAdaptiveAvgPool2d.from_float(target_module, qconfig)
                newPool.module_name = str(node.name).strip()
                replace_node_module(node, modules, newPool)
        elif node.op == "call_function":
            if node.target.__name__ == "add":
                with fx_model.graph.inserting_after(node):
                    print('{}: Replacing function add with QAdd'.format(str(node.name)))
                    QAdd = QElementwiseAdd(qconfig)
                    QAdd.module_name = str(node.name).strip()
                    QAdd_node = fx_model.graph.call_function(QAdd.forward, args=(node.args[0], node.args[1]))
                    QAdd_node.name = str(node.name).strip()
                    node.replace_all_uses_with(QAdd_node)
                fx_model.graph.erase_node(node)

        fx_model.graph.lint()
    fx_model.recompile()
    return fx_model


def create_deployable_model(model, qconfig):
    # Check if model is already a GraphModule
    if isinstance(model, fx.GraphModule):
        fx_model = copy.deepcopy(model)
    else:
        model = copy.deepcopy(model)
        fx_model = fx.symbolic_trace(model)

    print('=' * 50)
    print('> Preparing Emulation Model...'.center(50))
    print('=' * 50)

    # Step 1: Fuse the weights of the QuantizedConvBatchNorm2d module
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, nn.Conv2d):
                if type(target_module).__name__ == 'QuantizedConvBatchNorm2d':
                    # fuse the weights
                    fusedConv = target_module.to_fusedQConv2d()
                    fusedConv.quantize_module()
                    replace_node_module(node, modules, fusedConv)
    fx_model.graph.lint()
    fx_model.recompile()

    # Step 2: Switch modules to the emulation modules
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, QuantizedConv2d):
                print(f'{node.name}: Replacing QuantizedConv2d with FxP_QConv2D')
                emuConv = FxP_QConv2D.from_float(target_module)
                emuConv.module_name = str(node.name).strip()
                emuConv.quantize_module()
                replace_node_module(node, modules, emuConv)

            if isinstance(target_module, nn.Linear):
                print(f'{node.name}: Replacing QuantizedLinear with FxP_QLinear')
                emuConv = FxP_QLinear.from_float(target_module)
                emuConv.module_name = str(node.name).strip()
                emuConv.quantize_module()
                replace_node_module(node, modules, emuConv)

            if isinstance(target_module, nn.MaxPool2d):
                print(f'{node.name}: Replacing QMaxPool2D with FxP_QMaxPool2D')
                emuMax = FxP_QMaxPool2D.from_float(target_module)
                emuMax.module_name = str(node.name).strip()
                replace_node_module(node, modules, emuMax)

            if isinstance(target_module, nn.AdaptiveAvgPool2d):
                print(f'{node.name}: Replacing QAdaptiveAvgPool2d with FxP_QAdaptiveAvgPool2d')
                emuAdptAvgP = FxP_QAdaptiveAvgPool2d.from_float(target_module)
                emuAdptAvgP.module_name = str(node.name).strip()
                replace_node_module(node, modules, emuAdptAvgP)
        #
        #     if isinstance(target_module, nn.AvgPool2d):
        #         emuAvgP = FxP_QAvgPool2d.from_float(target_module)
        #         emuAvgP.module_name = str(node.name).strip()
        #         replace_node_module(node, modules, emuAvgP)
        #
        elif node.op == "call_function":
            if hasattr(node.target, '__self__') and isinstance(node.target.__self__, QElementwiseAdd):
                print(f'{node.name}: Replacing QElementwiseAdd with FxP_QElementwiseAdd')
                with fx_model.graph.inserting_after(node):
                    QAdd = FxP_QElementwiseAdd(qconfig)
                    QAdd.module_name = str(node.name).strip()
                    QAdd_node = fx_model.graph.call_function(QAdd.forward, args=(node.args[0], node.args[1]))
                    QAdd_node.name = str(node.name).strip()
                    node.replace_all_uses_with(QAdd_node)
                fx_model.graph.erase_node(node)
            elif node.target == QuantStubF:
                with fx_model.graph.inserting_after(node):
                    print(f'{node.name}: Replacing QuantStubF with QuantStubI')
                    quant_stub_i = fx_model.graph.call_function(QuantStubI, args=(node.args[0], qconfig['x']['out']))
                    node.replace_all_uses_with(quant_stub_i)
                    fx_model.graph.erase_node(node)
        fx_model.graph.lint()
    fx_model.recompile()
    add_stub_at_end(fx_model, QuantStubE, qconfig)

    return fx_model


def create_emulation_model(model):
    # Check if model is already a GraphModule
    if isinstance(model, fx.GraphModule):
        fx_model = copy.deepcopy(model)
    else:
        model = copy.deepcopy(model)
        fx_model = fx.symbolic_trace(model)

    # Step 1: Extract quantization parameters
    quant_params = generate_qconfig(fx_model)
    quant_params['conv1']['frac_in'].append(5)

    # Step 2: Switch modules to the emulation modules
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, nn.Conv2d):
                print(f'{node.name}: Replacing nn.Conv2D with FxP_QConv2D')
                emuConv = FxP_QConv2D.from_float(target_module, quant_params)
                emuConv.module_name = str(node.name).strip()
                # emuConv.quantize_module()
                replace_node_module(node, modules, emuConv)

            if isinstance(target_module, nn.Linear):
                print(f'{node.name}: Replacing nn.Linear with FxP_QLinear')
                emuConv = FxP_QLinear.from_float(target_module, quant_params)
                emuConv.module_name = str(node.name).strip()
                # emuConv.quantize_module()
                replace_node_module(node, modules, emuConv)

            if isinstance(target_module, nn.MaxPool2d):
                print(f'{node.name}: Replacing nn.MaxPool2D with FxP_QMaxPool2D')
                emuMax = FxP_QMaxPool2D.from_float(target_module, quant_params)
                emuMax.module_name = str(node.name).strip()
                replace_node_module(node, modules, emuMax)

            if isinstance(target_module, nn.AdaptiveAvgPool2d):
                print(f'{node.name}: Replacing nn.AdaptiveAvgPool2d with FxP_QAdaptiveAvgPool2d')
                emuAdptAvgP = FxP_QAdaptiveAvgPool2d.from_float(target_module, quant_params)
                emuAdptAvgP.module_name = str(node.name).strip()
                replace_node_module(node, modules, emuAdptAvgP)

            if isinstance(target_module, AddWithMetadata):
                print(f'{node.name}: Replacing AddWithMetadata with FxP_QElementwiseAdd')
                emuAdd = FxP_QElementwiseAdd(quant_params)
                emuAdd.module_name = str(node.name).strip()
                replace_node_module(node, modules, emuAdd)

        fx_model.graph.lint()
    fx_model.recompile()

    return fx_model


def add_stub_at_end(fx_model: fx.GraphModule, stub_fn, qconfig):
    last_node = None
    output_node = None

    for node in fx_model.graph.nodes:
        if node.op == 'output':
            output_node = node
        else:
            last_node = node

    # Insert the new stub function after the last operation node (before output)
    if last_node is not None:
        with fx_model.graph.inserting_after(last_node):
            quant_stub_node = fx_model.graph.call_function(stub_fn, args=(last_node, qconfig['output']['out']))
            quant_stub_node.name = 'qfloat_output'

        # Replace the input to the output node with the new quant_stub_node
        if output_node is not None:
            output_node.replace_input_with(output_node.args[0], quant_stub_node)

    # Recompile the graph after modification
    fx_model.graph.lint()
    fx_model.recompile()


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
