import copy 
import torch
import torch.nn as nn 
import torch.fx as fx
from typing import Type, Dict, Any, Tuple, Iterable
import copy
import torch.fx as fx
import torch
import torch.nn as nn


__all__ = ["fuse_conv_bn","replace_node_module","create_fused_model"]

def fuse_conv_bn(conv, bn):
    # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
    assert conv.bias is None

    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)   
    return conv 
def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name
def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)
    
def create_fused_model(model):
    model_fused = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue
            # print("name={}".format(node.name) ,"args = {}".format(node.args[0].target))
            conv = modules[node.args[0].target]            
            bn = modules[node.target]
            # if  conv.bias is None :
            #     fused_conv = fuse_conv_bn(conv, bn)
            #     replace_node_module(node.args[0], modules, fused_conv)
            #     node.replace_all_uses_with(node.args[0])
            #     fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model,torch.fx.GraphModule(model_fused, fx_model.graph)   


if __name__ == "__main__": 
    import torchvision.models as models
    resnet18 = models.resnet18() 
    create_fused_model(resnet18)
    