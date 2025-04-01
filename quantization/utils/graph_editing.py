import torch.fx as fx
import torch.nn as nn
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
from torch.fx.immutable_collections import immutable_list
# from torch.utils.tensorboard.summary import image

from quantization.fusedConvBn import FusedConvBN
from quantization.qmodules import (
    QuantizedConv2d,
    QuantizedLinear,
    QMaxPool2D,
    QAdaptiveAvgPool2d,
    QAvgPool2d,
    QElementwiseAdd,
    QuantStubF,
    QuantStubC
)

# for testing
import torchvision.models as models
from data_providers.imagenet import ImagenetDataProvider


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
                    if verbose:
                        print('{}: Fusing Conv2d with BatchNorm2d --> QFusedConvBN'
                              .format(str(node.args[0])))
                    previous_node = node.args[0]    # node with conv
                    bn = module_of_node(gm, node)
                    conv = module_of_previous_node(gm, node)
                    assert isinstance(bn, nn.BatchNorm2d)
                    assert isinstance(conv, nn.Conv2d)
                    fusemod = FusedConvBN.from_float(conv, bn)
                    fusemod.module_name = str(previous_node).strip()
                    replace_node_module(previous_node, modules, fusemod)
                    node.replace_all_uses_with(previous_node)
                    gm.graph.erase_node(node)
                    gm.graph.lint()
    gm.recompile()
    gm.delete_all_unused_submodules()
    return gm


def create_quantized_model(mod, verbose=False):
    # fuse the conv bn modules
    gm = fuse_convbn(mod, verbose=verbose)

    gm.add_submodule("quant_stub", QuantStubC(bitwidth=8, tensor_type='act'))

    modules = dict(gm.named_modules())
    # Replace other standard modules
    for node in gm.graph.nodes:
        modules = dict(gm.named_modules())

        if node.target == 'x':
            with gm.graph.inserting_after(node):
                quant_stub = gm.graph.call_module("quant_stub", args=(node,))
                quant_stub.name = "QuantStub"
                node.replace_all_uses_with(quant_stub)
                quant_stub.replace_input_with(quant_stub, node)

        elif node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, FusedConvBN):
                # print(target_module.module_name)
                pass
            elif isinstance(target_module, nn.Conv2d):
                if verbose:
                    print('{}: Replacing a Conv2d layer with QuantizedConv2d'
                          .format(str(node.name)))
                newConv = QuantizedConv2d.from_float(target_module)
                newConv.module_name = str(node.name).strip()
                replace_node_module(node, modules, newConv)
            elif isinstance(target_module, nn.Linear):
                if verbose:
                    print('{}: Replacing a Linear layer with QuantizedLinear'
                          .format(str(node.name)))
                newQlinear = QuantizedLinear.from_float(target_module)
                newQlinear.module_name = str(node.name).strip()
                replace_node_module(node, modules, newQlinear)
            elif isinstance(target_module, nn.MaxPool2d):
                if verbose:
                    print('{}: Replacing a MaxPool2d layer with QMaxPool2d'. format(str(node.name)))
                newMaxPool = QMaxPool2D.from_float(target_module)
                newMaxPool.module_name = str(node.name).strip()
                replace_node_module(node, modules, newMaxPool)
            elif isinstance(target_module, nn.AdaptiveAvgPool2d):
                if verbose:
                    print('{}: Replacing a AdaptiveAvgPool2d layer with'.format(str(node.name)))
                newPool = QAdaptiveAvgPool2d.from_float(target_module)
                newPool.module_name = str(node.name).strip()
                replace_node_module(node, modules, newPool)
        elif node.op == "call_function":
            if node.target.__name__ == "add":
                with gm.graph.inserting_after(node):
                    if verbose:
                        print('{}: Replacing function add with QAdd'.format(str(node.name)))

                    QAdd = QElementwiseAdd()
                    QAdd.module_name = str(node.name).strip()
                    QAdd_mod = str(node.name).strip()
                    gm.add_submodule(QAdd_mod, QAdd)
                    QAdd_node = gm.graph.call_module(QAdd_mod, args=(node.args[0], node.args[1]))
                    QAdd_node.name = str(node.name).strip()
                    node.replace_all_uses_with(QAdd_node)

                gm.graph.erase_node(node)

        gm.graph.lint()
    gm.recompile()
    return gm


def freeze(model):
    gm = model
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, FusedConvBN):
                target_module.freeze()


def calibrate(model, calib_loader):
    model.eval()
    model.cuda()
    for iteration, (input, target) in enumerate(calib_loader):
        input = input.cuda()
        output = model(input)


def create_compact_model(mod, verbose=False):
    """
    Transforms a given model into a compact model by replacing FusedConvBN layers defined in the
    graph with the compact quantized variant (a QuantizedConv2d
    layer) if applicable.

    Args:
        mod: fx.GraphModule or torch.nn.Module
            The input model to be transformed. If not already a GraphModule, the
            function will encapsulate the model in an fx.GraphModule first.
        verbose: bool, optional
            Indicates whether to print out debugging and transformation information
            during the process. Defaults to False.

    Returns:
        fx.GraphModule:
            The modified compact model with applicable layers replaced by their
            quantized counterparts.
    """
    # Check if model is already a GraphModule
    if isinstance(mod, fx.GraphModule):
        gm = mod
    else:
        gm = fx.GraphModule(mod)

    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, FusedConvBN):
                if verbose:
                    print('{}: Replacing a FusedConvBN layer with QuantizedConv2d'
                          .format(str(node.name)))
                QConv = target_module.to_qconv()
                replace_node_module(node, modules, QConv)

        gm.graph.lint()
    gm.recompile()
    return gm



if __name__ == '__main__':

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = fuse_convbn(model)

    print(model.state_dict())
