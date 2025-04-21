import torch.fx as fx
import torch
import torch.nn as nn
import copy
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
from torch.fx.immutable_collections import immutable_list
from torch.nn.utils.fusion import fuse_conv_bn_weights
from quantization.qat_modules import QuantizedLinear, QuantizedConv2d, QMaxPool2D, QAdaptiveAvgPool2d, QElementwiseAdd
from quantization.qat_modules import QuantizedConv2d
from quantization.fix_ops import FixedPointQuantizer, QuantStubF, QuantStubI, QuantStubE
from quantization.FxP_modules2 import *


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

