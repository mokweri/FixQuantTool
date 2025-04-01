import torch
import torch.fx as fx
import torch.nn as nn
import copy
import onnx

from quantization.fix_ops import fake_quantize_tensor, to_int_tensor
from quantization.fusedConvBn import FusedConvBN
from quantization.qmodules import QElementwiseAdd


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


def convert_to_inference_model(model):
    """
    Transforms a given neural network model into an inference-ready model by converting it into a quantizable format.
    This involves converting the model into a symbolic graph representation, creating a new graph with standard layers,
    and mapping these modules back into an inference model structure.

    The function replicates required layers, registers quantization metadata to manage fixed-point quantization parameters,
    and adds modules with associated quantization information to a standardized model representation.
    It uses PyTorch's symbolic tracing capabilities and facilitates export to ONNX formats with metadata.

    Arguments:
        model: The trained QAT-Aware network model.

    Returns:
        fx.GraphModule: The converted inference model represented as a new GraphModule with standard layers.

    Raises:
        RuntimeError: If a node in the graph attempts to use another node that has not been created
        or registered during processing.

    Notes:
        1. The function handles different types of modules, such as convolutional layers, linear layers, pooling layers,
           activation functions, and element-wise operations. Unsupported modules or functions are logged as warnings.
        2. Quantization metadata is registered to relevant layers to support fixed-point quantization.
        3. Final exported ONNX models contain quantization metadata.
        4. The input model is deep-copied to ensure the original structure is preserved.
    """

    # Check if model is already a GraphModule
    if isinstance(model, fx.GraphModule):
        fx_model = copy.deepcopy(model)
    else:
        model = copy.deepcopy(model)
        fx_model = fx.symbolic_trace(model)

    # Declare a new model - with standard pytorch modules
    inference_model = StandardModel()
    new_graph = fx.Graph()  # new graph
    node_map = {}  # Map original nodes to new nodes

    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op == "placeholder":  # Input node
            print('input node --')
            new_node = new_graph.placeholder(node.name)

        elif node.op == "call_module":
            target_module = modules[node.target]
            if isinstance(target_module, FusedConvBN):
                print('Fused conv --' + target_module.module_name)
                print('node args:', node.args)
                conv_layer = nn.Conv2d(
                    in_channels=target_module.conv_mod.in_channels,
                    out_channels=target_module.conv_mod.out_channels,
                    kernel_size=target_module.conv_mod.kernel_size, stride=target_module.conv_mod.stride,
                    padding=target_module.conv_mod.padding, dilation=target_module.conv_mod.dilation,
                    groups=target_module.conv_mod.groups, bias=target_module.conv_mod.bias is not None)
                # Copy trained weights
                # conv_layer.weight.data.copy_(target_module.conv_mod.weight.data)
                conv_layer.weight.data.copy_(fake_quantize_tensor(target_module.conv_mod.weight.data,
                                                                  signed=True, n_bits=8,
                                                                  n_frac=int(target_module.export_quant_info()[0])))
                if target_module.conv_mod.bias is not None:
                    # conv_layer.bias.data.copy_(target_module.conv_mod.bias.data)
                    conv_layer.bias.data.copy_(fake_quantize_tensor(target_module.conv_mod.bias.data,
                                                                    signed=True, n_bits=8,
                                                                    n_frac=int(target_module.export_quant_info()[1])))

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
                print('Normal conv -- ' + target_module.module_name)
                print('node args:', node.args)
                conv_layer = nn.Conv2d(
                    in_channels=target_module.in_channels, out_channels=target_module.out_channels,
                    kernel_size=target_module.kernel_size, stride=target_module.stride,
                    padding=target_module.padding, dilation=target_module.dilation,
                    groups=target_module.groups, bias=target_module.bias is not None)
                # Copy trained weights
                # conv_layer.weight.data.copy_(target_module.weight.data)
                conv_layer.weight.data.copy_(fake_quantize_tensor(target_module.weight.data,
                                                                  signed=True, n_bits=8,
                                                                  n_frac=int(target_module.export_quant_info()[0])))

                if target_module.bias is not None:
                    # conv_layer.bias.data.copy_(target_module.bias.data)
                    conv_layer.bias.data.copy_(fake_quantize_tensor(target_module.bias.data,
                                                                    signed=True, n_bits=8,
                                                                    n_frac=int(target_module.export_quant_info()[1])))

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
                linear_layer = nn.Linear(in_features=target_module.in_features,
                                         out_features=target_module.out_features,
                                         bias=target_module.bias is not None)
                print(target_module.in_features, target_module.out_features)
                # Copy trained weights
                # linear_layer.weight.data.copy_(target_module.weight.data)
                linear_layer.weight.data.copy_(fake_quantize_tensor(target_module.weight.data,
                                                                signed=True, n_bits=8,
                                                                n_frac=int(target_module.export_quant_info()[0])))
                if target_module.bias is not None:
                    # linear_layer.bias.data.copy_(target_module.bias.data)
                    linear_layer.bias.data.copy_(fake_quantize_tensor(target_module.bias.data,
                                                                        signed=True, n_bits=8,
                                                                        n_frac=int(
                                                                            target_module.export_quant_info()[1])))

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

                # -- just add function without metadata
                # new_node = new_graph.call_function(torch.add, args=(node_map[node.args[0]], node_map[node.args[1]]))
                # node_map[node] = new_node

                # --add as a module with metadata included
                add_layer = AddWithMetadata()
                # add_layer.register_buffer("frac_in1", torch.tensor(node.args[0].export_quant_info()))
                # add_layer.register_buffer("frac_in2", torch.tensor(target_module.export_quant_info()))
                add_layer.register_buffer("frac_act", torch.tensor(target_module.export_quant_info()))
                inference_model.layers[target_module.module_name] = add_layer
                new_node = new_graph.call_module(target_module.module_name,
                                                 args=(node_map[node.args[0]], node_map[node.args[1]]))
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
                print('Flatten --', node.name)
                print('node args:', node.args)
                # new_node = new_graph.call_function(torch.flatten, args=(node_map[node.args[0]],))

                inference_model.layers[node.name] = nn.Flatten()
                new_node = new_graph.call_module(node.name, args=(node_map[node.args[0]],))
                node_map[node] = new_node
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

    export_onnx(new_gm, "inf_resnet18.onnx", True)
    export_onnx_with_layer_metadata(new_gm, "resnet18.onnx")

    return new_gm


def generate_qconfig(model):
    # Generate qconfig
    qconfig = {}
    for name, module in model.named_modules():
        if hasattr(module, "bitwidth") or hasattr(module, "frac_w") or hasattr(module, "frac_act"):
            frac_in = []
            for node in model.graph.nodes:
                if node.target == name and node.op == "call_module":
                    for arg in node.args:
                        input_node = next((n for n in model.graph.nodes if n.name == arg.name), None)
                        while input_node:
                            if input_node.target in dict(model.named_modules()):
                                prev_module = dict(model.named_modules())[input_node.target]
                                if hasattr(prev_module, "frac_act"):
                                    frac_in.append(int(prev_module.frac_act))
                                    break
                                else:
                                    input_node = next(
                                        (n for n in model.graph.nodes if n.name == input_node.args[0].name), None)
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
        (name for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))),
        next(iter(model.named_modules()))[0])
    qconfig[first_layer_name] = {
        "bitwidth": 8,
        "frac_in": [5],
        "frac_w": 8,
        "frac_b": 8,
        "frac_out": 8
        }

    print("Successfully Generated qconfig:")
    return qconfig


def standardize_qconfig(qconfig):
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


def export_onnx(model, save_path, with_metadata=False):
    # collect the metadata, buffers

    buffer_dict = {}
    for name, module in model.named_modules():
        for buffer_name, buffer in module.named_buffers():
            buffer_key = f"{name}.{buffer_name}" if name else buffer_name
            buffer_dict[buffer_key] = str(buffer.item())  # Convert buffer value to string

    #save the model normally
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape according to your model input
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    if with_metadata:
        import onnx
        onnx_model = onnx.load(save_path)

        # Option 1: Add metadata normally (not always visible in Netron)
        for key, value in buffer_dict.items():
            meta_entry = onnx_model.metadata_props.add()
            meta_entry.key = key
            meta_entry.value = value

        # # Option 2: Store metadata in a dummy ONNX node (visible in Netron)
        # metadata_node = onnx.helper.make_node(
        #     "Constant",
        #     inputs=[],
        #     outputs=["metadata_dummy"],
        #     value=onnx.helper.make_tensor(
        #         name="metadata_dummy",
        #         data_type=onnx.TensorProto.STRING,
        #         dims=[len(buffer_dict)],
        #         vals=list(buffer_dict.values()),
        #     ),
        # )
        # # Add the dummy node to the graph
        # onnx_model.graph.node.append(metadata_node)

        # Save the updated ONNX model
        onnx.save(onnx_model, save_path)


def export_onnx_with_layer_metadata(model, save_path):
    # Export the model to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    # Load the ONNX model
    onnx_model = onnx.load(save_path)

    # Extract ONNX node names
    onnx_node_names = [node.name for node in onnx_model.graph.node]

    # Extract PyTorch module names and quantization parameters
    quant_params = {}
    for name, module in model.named_modules():
        if hasattr(module, "bitwidth") or hasattr(module, "frac_w") or hasattr(module, "frac_act"):
            quant_params[name] = {
                "bitwidth": getattr(module, "bitwidth", None),
                "frac_w": getattr(module, "frac_weight", None),
                "frac_b": getattr(module, "frac_bias", None),
                "frac_out": getattr(module, "frac_act", None)
            }

    # **Match PyTorch layers to ONNX nodes**
    matched_layers = {}
    for torch_layer_name in quant_params.keys():
        for onnx_node_name in onnx_node_names:
            if torch_layer_name in onnx_node_name:  # Check if ONNX node contains the PyTorch module name
                matched_layers[onnx_node_name] = quant_params[torch_layer_name]
                break  # Stop after the first match

    # **Attach quantization parameters to ONNX nodes**
    for node in onnx_model.graph.node:
        if node.name in matched_layers:
            params = matched_layers[node.name]
            for key, value in params.items():
                if value is not None:
                    attr = onnx.helper.make_attribute(key, int(value))  # Convert value to int
                    node.attribute.append(attr)

    # Save the updated ONNX model
    onnx.save(onnx_model, save_path)