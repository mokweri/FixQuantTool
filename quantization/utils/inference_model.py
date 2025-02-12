import torch
import torch.fx as fx
import torch.nn as nn
import copy

from quantization.fusedConvBn import FusedConvBN
from quantization.qmodules import QElementwiseAdd


class StandardModel(nn.Module):
    """New model composed from standard PyTorch modules"""
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


def make_inference_model(model):
    # Check if model is already a GraphModule
    if isinstance(model, fx.GraphModule):
        fx_model = copy.deepcopy(model)
    else:
        model = copy.deepcopy(model)
        fx_model = fx.symbolic_trace(model)

    # Declare a new model - with standard pytorch modules
    inference_model = StandardModel()
    new_graph = fx.Graph() # new graph
    node_map = {}  # Map original nodes to new nodes

    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op == "placeholder":  # Input node
            new_node = new_graph.placeholder(node.name)
            print('input node --')
        if node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, FusedConvBN):
                print('Fused conv --' + target_module.module_name)
                print('node args:', node.args)
                conv_layer = nn.Conv2d(
                    in_channels=target_module.conv_mod.in_channels, out_channels=target_module.conv_mod.out_channels,
                    kernel_size=target_module.conv_mod.kernel_size, stride=target_module.conv_mod.stride,
                    padding=target_module.conv_mod.padding, dilation=target_module.conv_mod.dilation,
                    groups=target_module.conv_mod.groups, bias=target_module.conv_mod.bias is not None)
                # Copy trained weights
                conv_layer.weight.data.copy_(target_module.conv_mod.weight.data)
                if target_module.conv_mod.bias is not None:
                    conv_layer.bias.data.copy_(target_module.conv_mod.bias.data)

                # **Register fixed-point quantization parameters as buffers**
                conv_layer.register_buffer("bitwidth", torch.tensor(8))
                conv_layer.register_buffer("frac_weight", torch.tensor(target_module.export_quant_info()[0]))
                conv_layer.register_buffer("frac_bias", torch.tensor(target_module.export_quant_info()[1]))
                conv_layer.register_buffer("frac_act", torch.tensor(target_module.export_quant_info()[2]))

                # Register layers
                inference_model.layers[target_module.module_name] = conv_layer
                # Add to new graph
                new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))

                print(target_module.export_quant_info())
                print('-' * 50)
            elif isinstance(target_module, nn.Conv2d):
                print( 'Normal conv -- '+ target_module.module_name)
                print('node args:', node.args)
                conv_layer = nn.Conv2d(
                    in_channels=target_module.in_channels, out_channels=target_module.out_channels,
                    kernel_size=target_module.kernel_size, stride=target_module.stride,
                    padding=target_module.padding, dilation=target_module.dilation,
                    groups=target_module.groups, bias=target_module.bias is not None)
                # Copy trained weights
                conv_layer.weight.data.copy_(target_module.weight.data)
                if target_module.bias is not None:
                    conv_layer.bias.data.copy_(target_module.bias.data)

                # Register fixed-point quantization parameters as buffers**
                conv_layer.register_buffer("bitwidth", torch.tensor(8))
                conv_layer.register_buffer("frac_weight", torch.tensor(target_module.export_quant_info()[0]))
                conv_layer.register_buffer("frac_bias", torch.tensor(target_module.export_quant_info()[1]))
                conv_layer.register_buffer("frac_act", torch.tensor(target_module.export_quant_info()[2]))

                # Register layers
                inference_model.layers[target_module.module_name] = conv_layer
                # Add to new graph
                new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))

                print(target_module.export_quant_info())
                print('-' * 50)

            elif isinstance(target_module, nn.Linear):
                print('Linear conv -- ' + target_module.module_name)
                print('node args:', node.args)
                linear_layer = nn.Linear(in_features=target_module.in_features, out_features=target_module.out_features,
                                         bias=target_module.bias is not None)

                # Copy trained weights
                linear_layer.weight.data.copy_(target_module.weight.data)
                if target_module.bias is not None:
                    linear_layer.bias.data.copy_(target_module.bias.data)

                # **Register fixed-point quantization parameters as buffers**
                linear_layer.register_buffer("bitwidth", torch.tensor(8))
                linear_layer.register_buffer("frac_weight", torch.tensor(target_module.export_quant_info()[0]))
                linear_layer.register_buffer("frac_bias", torch.tensor(target_module.export_quant_info()[1]))
                linear_layer.register_buffer("frac_act", torch.tensor(target_module.export_quant_info()[2]))

                # Register layer
                inference_model.layers[target_module.module_name] = linear_layer
                # Add to new graph
                new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))

                print(target_module.export_quant_info())
                print('-' * 50)
            elif isinstance(target_module, nn.MaxPool2d):
                print('MaxPool -- ' + target_module.module_name)
                print('node args:', node.args)
                maxPool_layer = nn.MaxPool2d(kernel_size=target_module.kernel_size, stride=target_module.stride,
                                             padding=target_module.padding, dilation=target_module.dilation,
                                             return_indices=target_module.return_indices,
                                             ceil_mode=target_module.ceil_mode)
                maxPool_layer.register_buffer("frac_act", torch.tensor(target_module.export_quant_info()))

                # Register layers
                inference_model.layers[target_module.module_name] = maxPool_layer
                # Add to new graph
                new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))

            elif isinstance(target_module, nn.ReLU):
                print('ReLU -- ')
                print('node args:', node.args)
                new_layer = copy.deepcopy(target_module)
                module_name = node.target.replace('.', '_')  # Fix the invalid naming issue
                inference_model.layers[module_name] = new_layer
                if node.args[0] in node_map:
                    new_node = new_graph.call_module(module_name, args=(node_map[node.args[0]],))
                else:
                    raise RuntimeError(
                        f"Node '{node.target}' is trying to use '{node.args[0]}' before it exists in the graph!")
                print('-' * 50)
            elif isinstance(target_module, nn.AdaptiveAvgPool2d):
                print('Avg pool -- ' + target_module.module_name)
                print('node args:', node.args)
                adaptive_avgpool_layer = nn.AdaptiveAvgPool2d(output_size=target_module.output_size)
                adaptive_avgpool_layer.register_buffer("frac_act", torch.tensor(target_module.export_quant_info()))
                # Register layers
                inference_model.layers[target_module.module_name] = adaptive_avgpool_layer
                # Add to new graph
                new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]],))
                print('-' * 50)
            elif isinstance(target_module, QElementwiseAdd):
                print('Addd')
                print('node args:', node.args)

                #-- just add function without metadata
                # new_node = new_graph.call_function(torch.add, args=(node_map[node.args[0]], node_map[node.args[1]]))
                # node_map[node] = new_node

                #--add as a module with metadata included
                add_layer = AddWithMetadata()
                add_layer.register_buffer("frac_act", torch.tensor(target_module.export_quant_info()))
                inference_model.layers[target_module.module_name] = add_layer
                new_node = new_graph.call_module(target_module.module_name, args=(node_map[node.args[0]], node_map[node.args[1]]))
                node_map[node] = new_node

            else:
                    if node.name == "QuantStub":
                        print('QuantStub skipped')
                    else:
                        print('Warning: Some module exists but not handled')
                    print(node.name)
                    print('-' * 50)
        elif node.op == "call_function":
            if node.target == torch.flatten:
                print('Flatten --')
                print('node args:', node.args)
                new_node = new_graph.call_function(torch.flatten, args=(node_map[node.args[0]],))
            else:
                print('Warning: Some function exists but not handled')

            print('-' * 50)
        elif node.op == "output":  # Output node
            new_node = new_graph.output(node_map[node.args[0]])
            print('Output --')

        node_map[node] = new_node  # Keep track of mapped nodes

    # Recompile graph
    new_graph.lint()
    new_gm = fx.GraphModule(inference_model.layers, new_graph)
    # ------------------------------------------------

    torch.save(new_gm.state_dict(), "fx_inference_model.pt")

    #--to onnx
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape according to your model input
    torch.onnx.export(
        new_gm,
        dummy_input,
        "fx_inference_model.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(new_gm)

    #@TODO Add metadata to the onnx model
