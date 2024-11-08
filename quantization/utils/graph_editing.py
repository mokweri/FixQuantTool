import torch.fx as fx
import torch.nn as nn
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
from torch.fx.immutable_collections import immutable_list

from quantization.fusedConvBn import  FusedConvBN
from quantization.qmodules import (
    QuantizedConv2d,
    QuantizedLinear,
    QMaxPool2D,
    QAdaptiveAvgPool2d,
    QAvgPool2d
)

# for testing
import torchvision.models as models


def module_of_node(gm : fx.GraphModule, node : fx.Node):
    assert node.op == "call_module", "module_of_node can only be called on 'call_module' nodes!"
    return gm.get_submodule(node.target)


def module_of_previous_node(gm: fx.GraphModule, node: fx.Node):
    """
    Retrieve the module associated with the node immediately before the specified node.
    """
    assert node.op == "call_module", "module_of_previous_node can only be called on 'call_module' nodes!"

    previous_node = None
    for n in gm.graph.nodes:
        if n == node:
            break
        if n.op == "call_module":
            previous_node = n

    assert previous_node is not None, "No previous 'call_module' node found in the graph!"
    return gm.get_submodule(previous_node.target)


def fuse_convbn(mod, verbose=False):
    gm: fx.GraphModule = fx.symbolic_trace(mod)
    modules = dict(gm.named_modules())

    modules_patterns = {
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d)
    }
    if verbose:
        print('=' * 50)
        print('> Fusing Conv and Bn modules...'.center(50))
        print('=' * 50)

    for pattern in modules_patterns:
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:
                    # conv that has multiple consumers is not fused
                    print('NOOOO')
                else:
                    previous_node = node.args[0]
                    bn = module_of_node(gm, node)
                    conv = module_of_previous_node(gm, node)
                    assert isinstance(bn, nn.BatchNorm2d)
                    assert isinstance(conv, nn.Conv2d)
                    fusemod = FusedConvBN.from_float(conv, bn)
                    replace_node_module(previous_node, modules, fusemod)
                    node.replace_all_uses_with(previous_node)
                    gm.graph.erase_node(node)
                gm.graph.lint()
    gm.recompile()
    gm.delete_all_unused_submodules()
    return gm











if __name__ == '__main__':
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    model = fuse_convbn(model)
    print(model)
