import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import yaml
import logging
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from fixquant.graph.inference_processor import InferProcessor
# # from fixquant.emulation.param_extractor import ModelParameterExtractor
from fixquant.emulation.model_introspector import StdModelInspector

# ------------------- CONSTANTS -------------------
# Model and quantization config
MODEL_CHECKPOINT = str(PROJECT_ROOT / "qat_models/checkpoint/resnet50_best.pth.tar")
QUANT_CONFIG = str(PROJECT_ROOT / "configs/quant_config.yaml")

# Selected layers and reasons (folder name will be the layer name)
LAYER_SPECS = {
    # name: reason
    "conv1": "7x7, stride 2 (early, big maps). Stresses padding, large kernel loops, big HxW DRAM traffic.",
    "layer1_1_conv2": "3x3, stride 1 @56x56 baseline same-res. Clean correctness case for 3x3 MAC/tile at high spatial size.",
    "layer2_0_conv2": "3x3, stride 2 @56->28. Validates stride-2 address generation, halos, boundary tiles.",
    "layer3_4_conv1": "1x1 reduce @14x14 (Cin≫Cout). Stresses read bandwidth and channel-tiling with very large Cin.",
    "layer4_0_conv3": "1x1 expand @7x7 (Cout≫Cin). Max-Cout write bandwidth, accumulator width/saturation at tiny HxW.",
    "layer4_1_conv2": "large weight size."
}

# Base output directory (per-layer subfolders will be created here)
BASE_OUT_DIR = str(PROJECT_ROOT / "outputs/hw_data_files")

# Test image candidates (in repo)
TEST_IMAGE_CANDIDATES = [
    str(PROJECT_ROOT / "assets/new.JPEG"),
    str(PROJECT_ROOT / "assets/cat.jpg"),
]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hw_layer_test_gen")


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor


def _fmt_tuple_csv(tup):
    if tup is None:
        return ""
    return ",".join(str(int(x)) for x in tup)


def _write_details_txt(path: str, *, name: str, mod: nn.Module, reason: str, qparams: dict,
                       activation_in_dims, weights_dim, conv_params) -> None:
    # Machine-readable header (key=value)
    op_type = type(mod).__name__
    bitwidth = 8
    signed = 1
    has_bias = int(getattr(mod, "bias", None) is not None)

    # Layout and dims
    activation_layout = "CHW" if activation_in_dims is not None else ""
    weights_layout = ""
    if isinstance(mod, nn.Conv2d):
        weights_layout = "OIHW"
    elif isinstance(mod, nn.Linear):
        weights_layout = "OI"

    # Stride/padding/kernel only for Conv2d
    stride_csv = padding_csv = kernel_csv = None
    if isinstance(mod, nn.Conv2d) and conv_params is not None:
        stride_csv = _fmt_tuple_csv(conv_params.get("stride"))
        padding_csv = _fmt_tuple_csv(conv_params.get("padding"))
        kernel_csv = _fmt_tuple_csv(conv_params.get("kernel_size"))

    with open(path, "w") as f:
        # Required keys
        f.write("version=1\n")
        f.write(f"op_type={op_type}\n")
        f.write(f"activation_layout={activation_layout}\n")
        if activation_in_dims is not None:
            f.write(f"activation_dims={_fmt_tuple_csv(activation_in_dims)}\n")
        else:
            f.write("activation_dims=\n")
        f.write(f"weights_layout={weights_layout}\n")
        if weights_dim is not None:
            f.write(f"weights_dims={_fmt_tuple_csv(weights_dim)}\n")
        else:
            f.write("weights_dims=\n")
        if stride_csv is not None:
            f.write(f"stride={stride_csv}\n")
        if padding_csv is not None:
            f.write(f"padding={padding_csv}\n")
        if kernel_csv is not None:
            f.write(f"kernel={kernel_csv}\n")
        f.write(f"bitwidth={bitwidth}\n")
        f.write(f"signed={signed}\n")
        f.write(f"frac_in={qparams.get('frac_in', 0)}\n")
        f.write(f"frac_w={qparams.get('frac_w', 0)}\n")
        f.write(f"frac_b={qparams.get('frac_b', 0)}\n")
        f.write(f"frac_out={qparams.get('frac_out', 0)}\n")
        f.write(f"has_bias={has_bias}\n")
        # Filenames relative to this folder
        f.write("input_file=input.data\n")
        f.write("weights_file=weights.data\n")
        f.write("bias_file=bias.data\n")

        # Human-readable notes as comments
        f.write("\n")
        f.write(f"# Layer: {name}\n")
        f.write(f"# Type: {op_type}\n")
        f.write(f"# Reason: {reason}\n")


def main():
    # Ensure base output directory exists
    os.makedirs(os.path.abspath(BASE_OUT_DIR), exist_ok=True)

    # 1. Load model and quantization config
    with open(QUANT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    # Avoid downloading pretrained weights; QAT weights will be loaded next
    model = models.resnet50(weights=None)
    from fixquant.graph.qat_processor import QatProcessor
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()
    # qat_proc.freeze()
    qat_proc.load_qat_weights(MODEL_CHECKPOINT)
    qat_proc.freeze()

    # 2. Convert to standard model
    infer_proc = InferProcessor(model, config)
    stdm = infer_proc.convert_to_std_model()

    # 3. Build inspector
    inspector = StdModelInspector(stdm, default_input_frac=5, logger=logger)

    # 4. Prepare input and register hooks for all target layers
    test_image_path = None
    for p in TEST_IMAGE_CANDIDATES:
        if os.path.exists(p):
            test_image_path = p
            break
    if test_image_path is not None:
        input_tensor = preprocess_image(test_image_path)
        logger.info(f"Using test image: {test_image_path}")
    else:
        logger.warning("No test image found, using random tensor.")
        input_tensor = torch.rand(1, 3, 224, 224)

    target_layers = list(LAYER_SPECS.keys())
    inspector.register_activation_hooks(target_layers, capture_input=True, capture_output=True, clear_existing=True)

    with torch.no_grad():
        inspector.run_and_capture(input_tensor)

    # 5. Iterate each layer and export into its own folder
    for name in target_layers:
        reason = LAYER_SPECS[name]
        layer_dir = os.path.join(BASE_OUT_DIR, name)
        os.makedirs(layer_dir, exist_ok=True)

        weight_file = os.path.join(layer_dir, "weights.data")
        bias_file = os.path.join(layer_dir, "bias.data")
        act_file = os.path.join(layer_dir, "input.data")
        qparams_file = os.path.join(layer_dir, "qparams.json")
        details_file = os.path.join(layer_dir, "details.txt")

        # Save layer params (full weights; no subset)
        try:
            frac_w, frac_b = inspector.save_layer_params(
                name=name,
                weight_file=weight_file,
                bias_file=bias_file,
                target_weight_shape=None,
                n_bits_out=8,
            )
        except Exception as e:
            logger.error(f"Failed to save params for layer '{name}': {e}")
            continue

        # Determine frac_in/out
        q = inspector.get_quant_params(name)
        frac_out = int(q["frac_out"]) if q["frac_out"] is not None else 0
        frac_in_list = q["frac_in"] if isinstance(q.get("frac_in"), list) else []
        frac_in = int(frac_in_list[0]) if len(frac_in_list) > 0 else inspector.default_input_frac

        # Save activation (input to the layer)
        try:
            inspector.save_activation(
                name=name,
                filepath=act_file,
                which="input",
                n_bits=8,
                n_frac=frac_in,
            )
        except Exception as e:
            logger.error(f"Failed to save activation for layer '{name}': {e}")

        # Gather metadata for JSON and details
        mod = inspector.get_module(name)
        act_in_shapes = inspector.input_shapes.get(name) or []
        activation_in_dims = None
        if len(act_in_shapes) > 0:
            shape0 = tuple(act_in_shapes[0])
            if len(shape0) == 4 and shape0[0] == 1:
                activation_in_dims = (shape0[1], shape0[2], shape0[3])  # CHW
            elif len(shape0) == 3:
                activation_in_dims = shape0
            else:
                activation_in_dims = tuple(shape0[-3:])

        weight_dims = None
        if hasattr(mod, "weight") and mod.weight is not None:
            weight_dims = tuple(mod.weight.shape)

        conv_params = None
        if isinstance(mod, nn.Conv2d):
            def _to_tuple(x):
                try:
                    return tuple(int(v) for v in x)
                except TypeError:
                    return (int(x), int(x))
            conv_params = {
                "padding": _to_tuple(mod.padding),
                "stride": _to_tuple(mod.stride),
                "kernel_size": _to_tuple(mod.kernel_size),
            }

        # Save quantization parameters + metadata (JSON)
        qparams = {
            "layer": name,
            "frac_w": int(frac_w) if frac_w is not None else 0,
            "frac_b": int(frac_b) if frac_b is not None else 0,
            "frac_in": int(frac_in),
            "frac_out": int(frac_out),
            "activation_in_dims": list(activation_in_dims) if activation_in_dims is not None else None,
            "weights_dim": list(weight_dims) if weight_dims is not None else None,
            "conv_params": conv_params,
        }
        with open(qparams_file, "w") as f:
            json.dump(qparams, f, indent=2)
        logger.info(f"Saved quantization parameters to {qparams_file}")

        # Save details.txt (machine header + comments)
        _write_details_txt(
            details_file,
            name=name,
            mod=mod,
            reason=reason,
            qparams=qparams,
            activation_in_dims=activation_in_dims,
            weights_dim=weight_dims,
            conv_params=conv_params,
        )
        logger.info(f"Wrote details to {details_file}")


if __name__ == "__main__":
    main()
