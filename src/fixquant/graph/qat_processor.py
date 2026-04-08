import torch.fx as fx
import torch.nn as nn
import copy

from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
from fixquant.quantization.qat_modules import (QuantizedLinear, QuantizedConv2d, QMaxPool2D, QAdaptiveAvgPool2d,
                                      QElementwiseAdd, QuantStubC)
from fixquant.quantization.fused_conv_bn import FusedConvBN
from fixquant.graph.inference_processor import InferProcessor
from fixquant.quantization.fxp_modules import *
import yaml
import logging
from typing import Sequence, Callable, Dict, Any, Optional

# for testing
import torchvision.models as models


# --- Configuration ---
# with open("quant_config.yaml", "r") as f:
#     quant_config = yaml.safe_load(f)  # Assuming you have a config file

# --- Helper Functions ---
def get_node_name(node: fx.Node) -> str:
    """Helper to get a clean name for a node."""
    return str(node.name).strip()


def replace_module(node: fx.Node, graph_module: fx.GraphModule, modules: dict, new_module: nn.Module):
    """Helper to replace a module in the graph."""
    replace_node_module(node, modules, new_module)


def insert_stub(node: fx.Node, graph_module: fx.GraphModule, stub_fn, args):
    """Helper to insert a stub function."""

    graph_module.add_submodule("quant_stub", QuantStubC(bitwidth=8, tensor_type='act'))
    with graph_module.graph.inserting_after(node):
        stub = graph_module.graph.call_module("quant_stub", args=args)
        stub.name = "QuantStub"
        node.replace_all_uses_with(stub)
        stub.replace_input_with(stub, node)
        return stub


def match_module_type(node: fx.Node, modules: dict, module_types: tuple) -> bool:
    """Helper to match a node's module type."""
    return type(modules[node.target]) in module_types


# --- Transformation Passes ---
class BaseQuantizationPass:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, graph_module: fx.GraphModule):
        raise NotImplementedError

    def _process_node(self, node: fx.Node, graph_module: fx.GraphModule, modules: dict):
        """Logic to apply to a single node (to be overridden)."""
        pass

    def _filter_nodes(self, graph_module: fx.GraphModule):
        """Optional: Filter nodes to process."""
        return graph_module.graph.nodes


class ConvBnFusionPass(BaseQuantizationPass):
    def __init__(self, config):
        super().__init__(config)
        self.modules_patterns = {
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d)
        }

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Fusing Conv and BN modules...")
        modules = dict(graph_module.named_modules())
        for node in graph_module.graph.nodes:
            for pattern in self.modules_patterns:
                if matches_module_pattern(pattern, node, modules):
                    if len(node.args[0].users) > 1:
                        self.logger.debug(f"Skipping fusion for {get_node_name(node)}: multiple users.")
                        continue
                    self._fuse_conv_bn(node, graph_module, modules, pattern)
                    break  # Only fuse once per node
        graph_module.recompile()
        graph_module.delete_all_unused_submodules()

    def _fuse_conv_bn(self, node: fx.Node, graph_module: fx.GraphModule, modules: dict, pattern: tuple):
        conv_node = node.args[0]
        conv = modules[conv_node.target]
        bn = modules[node.target]
        assert isinstance(bn, nn.BatchNorm2d)
        assert isinstance(conv, nn.Conv2d)
        fused_module = FusedConvBN.from_float(conv, bn)
        fused_module.module_name = get_node_name(conv_node)
        replace_module(conv_node, graph_module, modules, fused_module)
        node.replace_all_uses_with(conv_node)
        graph_module.graph.erase_node(node)
        graph_module.graph.lint()
        self.logger.debug(f"Fused {get_node_name(conv_node)} with {get_node_name(node)}.")


class ModuleReplacementPass(BaseQuantizationPass):
    def __init__(self, replacement_type, config):
        super().__init__(config)
        assert replacement_type
        self.replacement_type = replacement_type
        self.module_replacement_maps = {
            'quant': {nn.Conv2d: QuantizedConv2d,
                      nn.Linear: QuantizedLinear,
                      nn.MaxPool2d: QMaxPool2D,
                      nn.AdaptiveAvgPool2d: QAdaptiveAvgPool2d, },
            'compact': {FusedConvBN: QuantizedConv2d},

        }
        self.replacement_map = self.module_replacement_maps[replacement_type]

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Replacing modules based on mapping...")
        modules = dict(graph_module.named_modules())
        for node in self._filter_nodes(graph_module):
            if self._match_node(node, modules):
                self._replace_module(node, graph_module, modules)
        graph_module.recompile()

    def _filter_nodes(self, graph_module: fx.GraphModule):
        return [node for node in graph_module.graph.nodes if node.op == "call_module"]

    def _match_node(self, node: fx.Node, modules: dict) -> bool:
        target_module_type = type(modules[node.target])
        return target_module_type in self.replacement_map

    def _replace_module(self, node: fx.Node, graph_module: fx.GraphModule, modules: dict):
        target_module = modules[node.target]
        replacement_class = self.replacement_map[type(target_module)]
        if self.replacement_type == 'compact':
            new_module = target_module.to_qconv()  # to preserve qconfig parameters
        else:
            new_module = replacement_class.from_float(target_module)
        new_module.module_name = get_node_name(node)
        replace_module(node, graph_module, modules, new_module)
        self.logger.debug(f"Replaced {get_node_name(node)} with {type(new_module).__name__}.")


class ReplaceAddPass(BaseQuantizationPass):
    def __init__(self, config):
        super().__init__(config)

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Replacing 'add' function with quantized version...")
        for node in self._filter_nodes(graph_module):
            if node.target.__name__ == "add":
                self._replace_add(node, graph_module)
        graph_module.recompile()

    def _filter_nodes(self, graph_module: fx.GraphModule):
        return [node for node in graph_module.graph.nodes if node.op == "call_function"]

    def _replace_add(self, node: fx.Node, graph_module: fx.GraphModule):
        with graph_module.graph.inserting_after(node):
            qadd_module = QElementwiseAdd()
            qadd_module.module_name = get_node_name(node)
            graph_module.add_submodule(get_node_name(node), qadd_module)
            qadd_node = graph_module.graph.call_module(get_node_name(node), args=(node.args[0], node.args[1]))
            qadd_node.name = get_node_name(node)
            node.replace_all_uses_with(qadd_node)
        graph_module.graph.erase_node(node)
        self.logger.debug(f"Replaced add function at {get_node_name(node)} with QElementwiseAdd.")


class QuantizeLayerPass(BaseQuantizationPass):
    def __init__(self, config):
        super().__init__(config)
        self.layer_config = self.config["quantize_layers"]

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Replacing layers with quantized versions...")
        modules = dict(graph_module.named_modules())
        for node in self._filter_nodes(graph_module):
            if match_module_type(node, modules, tuple(self.layer_config.keys())):
                self._quantize_module(node, graph_module, modules)
        graph_module.recompile()

    def _filter_nodes(self, graph_module: fx.GraphModule):
        return [node for node in graph_module.graph.nodes if node.op == "call_module"]

    def _quantize_module(self, node: fx.Node, graph_module: fx.GraphModule, modules: dict):
        target_module = modules[node.target]
        quant_module_class = self.layer_config[type(target_module)]
        new_module = quant_module_class.from_float(target_module, self.config)
        new_module.module_name = get_node_name(node)
        new_module.quantize_module()
        replace_module(node, graph_module, modules, new_module)
        self.logger.debug(f"Replaced {get_node_name(node)} with {type(new_module).__name__}.")


class InputQuantStubPass(BaseQuantizationPass):
    def __init__(self, config):
        super().__init__(config)

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Adding input quantization stub...")
        for node in self._filter_nodes(graph_module):
            if node.target == 'x':
                self._add_input_stub(node, graph_module)
        graph_module.recompile()

    def _filter_nodes(self, graph_module: fx.GraphModule):
        return [node for node in graph_module.graph.nodes]

    def _add_input_stub(self, node: fx.Node, graph_module: fx.GraphModule):
        stub = insert_stub(node, graph_module, QuantStubC, args=(node,))
        self.logger.debug(f"Added input quant stub before {get_node_name(node)}.")


class OutputQuantStubPass(BaseQuantizationPass):
    def __init__(self, config):
        super().__init__(config)
        self.output_config = self.config.get("output", {})

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Adding output quantization stubs...")
        for node in self._filter_nodes(graph_module):
            if node.op == 'output':
                self._add_output_stub(node, graph_module)
        graph_module.recompile()

    def _filter_nodes(self, graph_module: fx.GraphModule):
        return [node for node in graph_module.graph.nodes]

    def _add_output_stub(self, node: fx.Node, graph_module: fx.GraphModule):
        last_node = node.args[0]
        stub = insert_stub(node, graph_module, QuantStubC, args=(last_node, self.output_config.get('out', 8)))
        node.replace_input_with(last_node, stub)
        stub.name = 'qfloat_output'
        self.logger.debug(f"Added output quant stub before {get_node_name(node)}.")


class FreezeModulesPass(BaseQuantizationPass):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Freezing modules...")
        modules = dict(graph_module.named_modules())
        for node in self._filter_nodes(graph_module):
            if match_module_type(node, modules, (FusedConvBN,)):
                self._freeze_module(node, modules)
        graph_module.recompile()

    def _filter_nodes(self, graph_module: fx.GraphModule) -> Sequence[fx.Node]:
        return [node for node in graph_module.graph.nodes if node.op == "call_module"]

    def _freeze_module(self, node: fx.Node, modules: Dict[str, nn.Module]):
        target_module = modules[node.target]
        target_module.freeze()
        self.logger.debug(f"Frozen module: {get_node_name(node)}.")


# --- Model Quantizer for QAT ---
class QatProcessor:
    def __init__(self, model: nn.Module, config: dict):
        self.model = copy.deepcopy(model)
        self.qat_model: Optional[fx.GraphModule] = None
        self.std_model: Optional[fx.GraphModule] = None
        self.config = config
        self.qat_passes = self._create_qat_passes()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_qat_passes(self):
        return [
            ConvBnFusionPass(self.config),
            ModuleReplacementPass("quant", self.config),
            ReplaceAddPass(self.config),
            InputQuantStubPass(self.config)
        ]

    def quantize(self) -> fx.GraphModule:
        self.logger.info("Starting quantization process...")
        self.qat_model = fx.symbolic_trace(self.model)

        for pass_ in self.qat_passes:
            pass_.run(self.qat_model)

        self.qat_model.recompile()
        self.qat_model.delete_all_unused_submodules()

        self.logger.info("Qat pre-processing completed.")
        return self.qat_model

    def freeze(self) -> None:
        if self.qat_model is None:
            raise ValueError("Quantize the model first before freezing.")

        freeze_pass = FreezeModulesPass(self.config)
        freeze_pass.run(self.qat_model)
        self.qat_model.recompile()
        self.logger.info("Model frozen.")

    def calibrate(self, calib_loader, device) -> None:
        if self.qat_model is None:
            raise ValueError("Quantize the model first before calibration.")

        self.qat_model.eval()
        self.qat_model.to(device)

        with torch.no_grad():
            for iteration, (input, target) in enumerate(calib_loader):
                input = input.to(device)
                output = self.qat_model(input)
                self.logger.debug(f"Calibration batch {iteration} processed.")

        self.qat_model.train()  # Set back to train mode after calibration
        self.qat_model.to("cpu")  # Move back to cpu
        self.logger.info("Model calibrated.")

    def load_qat_weights(self, checkpoint_path: str) -> None:
        if self.qat_model is None:
            raise ValueError("Quantize the model first before loading weights.")

        checkpoint = torch.load(checkpoint_path)
        self.qat_model.load_state_dict(checkpoint["state_dict"])
        self.logger.info(f"Loaded QAT weights from {checkpoint_path}")

    def compact_model(self):
        if self.qat_model is None:
            raise ValueError("Quantize the model first before converting to standard model.")

        self.freeze()
        self.std_model = copy.deepcopy(self.qat_model)
        std_pass = ModuleReplacementPass("compact", self.config)
        std_pass.run(self.std_model)

        return self.std_model


# --- Main ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)  # Set desired logging level

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    with open("quant_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(config.get("module_replacements", {}).keys())
    quantizer = QatProcessor(model, config)
    quantized_model = quantizer.quantize()

    cmode = quantizer.compact_model()

    InferProcessor = InferProcessor(quantized_model, config)
    stdm = InferProcessor.convert_to_std_model()
    # stdm = convert_to_inference_model(quantized_model)

    print(stdm)
    # print(model)
