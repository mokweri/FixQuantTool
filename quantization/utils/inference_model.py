import torch.fx as fx
import torch.nn as nn
import copy

from quantization.fusedConvBn import FusedConvBN
from quantization.qmodules import QElementwiseAdd

def make_inference_model(model):
    # Check if model is already a GraphModule
    if isinstance(model, fx.GraphModule):
        fx_model = copy.deepcopy(model)
    else:
        model = copy.deepcopy(model)
        fx_model = fx.symbolic_trace(model)

    # Declare a new model - with standard pytorch modules

    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, FusedConvBN):
                print('Fused conv --' + target_module.module_name)
                print(target_module.export_quant_info())
            elif isinstance(target_module, nn.Conv2d):
                print( 'Normal conv -- '+ target_module.module_name)
            elif isinstance(target_module, nn.Linear):
                print('Linear conv -- ' + target_module.module_name)
            elif isinstance(target_module, nn.MaxPool2d):
                print('MaxPool conv -- ' + target_module.module_name)
            elif isinstance(target_module, nn.AdaptiveAvgPool2d):
                print('Avg pool conv -- ' + target_module.module_name)
            elif isinstance(target_module, QElementwiseAdd):
                print('Addd')
