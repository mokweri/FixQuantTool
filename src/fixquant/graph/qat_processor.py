import torch.fx as fx
import torch.nn as nn
import copy

from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
from fixquant.quantization.qat_modules import (QuantizedLinear, QuantizedConv2d, QMaxPool2D, QAdaptiveAvgPool2d,
                                      QElementwiseAdd, QuantStubC)
from fixquant.quantization.fused_conv_bn import FusedConvBN, FREEZE_BN_DELAY_DEFAULT
from fixquant.quantization.tqt_quantizer import TQTQuantizer
from fixquant.graph.inference_processor import InferProcessor
import torch
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
        freeze_bn_delay = self.config.get("freeze_bn_delay", FREEZE_BN_DELAY_DEFAULT)
        fused_module = FusedConvBN.from_float(conv, bn, freeze_bn_delay=freeze_bn_delay)
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


class ReplaceFunctionalPoolPass(BaseQuantizationPass):
    """Replace functional adaptive_avg_pool2d calls (used by torchvision
    MobileNetV2) with QAdaptiveAvgPool2d modules so the GAP output gets a
    quantizer and a qconfig entry like its module-form counterpart."""

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Replacing functional adaptive_avg_pool2d with quantized module...")
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            if getattr(node.target, "__name__", "") != "adaptive_avg_pool2d":
                continue
            output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size", (1, 1))
            qpool = QAdaptiveAvgPool2d(output_size)
            qpool.module_name = get_node_name(node)
            graph_module.add_submodule(get_node_name(node), qpool)
            with graph_module.graph.inserting_after(node):
                new_node = graph_module.graph.call_module(get_node_name(node), args=(node.args[0],))
                node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            self.logger.debug(f"Replaced functional pool at {get_node_name(new_node)}.")
        graph_module.recompile()


class ActRangePass(BaseQuantizationPass):
    """Attach analytic output bounds to activation quantizers.

    A conv/linear whose only consumer is a ReLU6 can never contribute values
    outside [0, 6] to the rest of the network (the hardware applies the clamp
    after requantization), so its activation quantizer should spend its range
    on [0, 6] instead of the raw pre-activation distribution. Plain ReLU gives
    a [0, inf) bound. The bound steers warmup init and calibration sampling.
    """

    _ACT_QUANT_ATTRS = ("act_quantizer", "quantizer")

    def run(self, graph_module: fx.GraphModule):
        self.logger.info("Attaching activation range bounds (ReLU/ReLU6)...")
        modules = dict(graph_module.named_modules())
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            mod = modules[node.target]
            quantizer = None
            for attr in self._ACT_QUANT_ATTRS:
                q = getattr(mod, attr, None)
                if isinstance(q, TQTQuantizer) and q.tensor_type == "act":
                    quantizer = q
                    break
            if quantizer is None or len(node.users) != 1:
                continue
            (user,) = node.users
            if user.op != "call_module":
                continue
            user_mod = modules.get(user.target)
            if isinstance(user_mod, nn.ReLU6):
                quantizer.bounded_range = (0.0, 6.0)
                self.logger.debug(f"{get_node_name(node)}: act range bounded to [0, 6] (ReLU6).")
            elif isinstance(user_mod, nn.ReLU):
                quantizer.bounded_range = (0.0, None)
                self.logger.debug(f"{get_node_name(node)}: act range bounded to [0, inf) (ReLU).")


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
            ReplaceFunctionalPoolPass(self.config),
            ActRangePass(self.config),
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

    def calibrate(self, calib_loader, device, max_batches=None, scope=5) -> dict:
        """Multi-batch calibration.

        Runs the QAT model over the calibration loader while every TQTQuantizer
        collects a value reservoir, then sets each threshold via the MSE
        fix-position search (fix_ops.find_fix_pos). Returns {quantizer: frac}.
        """
        if self.qat_model is None:
            raise ValueError("Quantize the model first before calibration.")

        quantizers = {name: m for name, m in self.qat_model.named_modules()
                      if isinstance(m, TQTQuantizer)}
        for q in quantizers.values():
            q.start_calibration()

        self.qat_model.eval()
        self.qat_model.to(device)

        n_batches = 0
        with torch.no_grad():
            for iteration, (input, target) in enumerate(calib_loader):
                if max_batches is not None and iteration >= max_batches:
                    break
                input = input.to(device)
                self.qat_model(input)
                n_batches += 1
                self.logger.debug(f"Calibration batch {iteration} processed.")

        fracs = {name: q.finish_calibration(scope=scope) for name, q in quantizers.items()}

        self.qat_model.train()  # Set back to train mode after calibration
        self.qat_model.to("cpu")  # Move back to cpu
        self.logger.info(f"Model calibrated on {n_batches} batches "
                         f"({len(fracs)} quantizers, MSE fix-pos scope={scope}).")
        return fracs

    def freeze_thresholds(self, frozen: bool = True) -> None:
        """Stop (or resume) training of all TQT log-thresholds."""
        if self.qat_model is None:
            raise ValueError("Quantize the model first.")
        for m in self.qat_model.modules():
            if isinstance(m, TQTQuantizer):
                m.freeze_quant(frozen)
        self.logger.info(f"Quantizer thresholds {'frozen' if frozen else 'unfrozen'}.")

    def load_qat_weights(self, checkpoint_path: str) -> None:
        if self.qat_model is None:
            raise ValueError("Quantize the model first before loading weights.")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        try:
            self.qat_model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Common mistake: a checkpoint trained with cross-layer equalization
            # (qat_train.py --cle) is BN-free (QuantizedConv2d, no conv_mod/bn_mod),
            # so it will not load into a plain FusedConvBN model. Detect that here
            # and turn the giant key-mismatch dump into an actionable message.
            model_is_cle = not any('.conv_mod.' in k or '.bn_mod.' in k
                                   for k in self.qat_model.state_dict())
            ckpt_is_cle = not any('.conv_mod.' in k or '.bn_mod.' in k
                                  for k in state_dict)
            if ckpt_is_cle and not model_is_cle:
                raise RuntimeError(
                    "Checkpoint appears to be BN-free (trained with cross-layer "
                    "equalization) but the model was built with Conv-BN fusion. "
                    "Re-run with --cle so the model architecture matches the "
                    "checkpoint."
                ) from e
            if model_is_cle and not ckpt_is_cle:
                raise RuntimeError(
                    "Model was built with cross-layer equalization (--cle) but the "
                    "checkpoint uses Conv-BN fusion. Drop --cle to match the "
                    "checkpoint."
                ) from e
            raise
        self.logger.info(f"Loaded QAT weights from {checkpoint_path}")

    def compact_model(self):
        if self.qat_model is None:
            raise ValueError("Quantize the model first before converting to standard model.")

        self.freeze()
        self.std_model = copy.deepcopy(self.qat_model)
        std_pass = ModuleReplacementPass("compact", self.config)
        std_pass.run(self.std_model)

        return self.std_model


def preflight_check(model: nn.Module, raise_on_error: bool = True):
    """Verify that a float model only contains ops the QAT + TileCNN export
    pipeline supports. Returns a list of issue strings (empty = clean).
    """
    SUPPORTED_MODULES = (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ReLU, nn.ReLU6,
                         nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Dropout, nn.Flatten,
                         nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Identity)
    SUPPORTED_FUNCTIONS = {"add", "flatten", "adaptive_avg_pool2d"}

    issues = []
    traced = fx.symbolic_trace(copy.deepcopy(model))
    modules = dict(traced.named_modules())
    for node in traced.graph.nodes:
        if node.op == "call_module":
            mod = modules[node.target]
            if not isinstance(mod, SUPPORTED_MODULES):
                issues.append(f"unsupported module '{node.target}' ({type(mod).__name__})")
            elif isinstance(mod, nn.AdaptiveAvgPool2d) and mod.output_size not in (1, (1, 1)):
                issues.append(f"'{node.target}': AdaptiveAvgPool2d output_size {mod.output_size} "
                              "(TileCNN only supports global average pooling)")
        elif node.op == "call_function":
            fname = getattr(node.target, "__name__", str(node.target))
            if fname not in SUPPORTED_FUNCTIONS:
                issues.append(f"unsupported function '{fname}' at node {node.name}")
        elif node.op == "call_method":
            if node.target not in ("view", "reshape", "flatten", "contiguous", "size"):
                issues.append(f"unsupported method '.{node.target}()' at node {node.name}")

    if issues and raise_on_error:
        raise ValueError("Pre-flight check failed:\n  " + "\n  ".join(issues))
    return issues


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
