import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import yaml
import logging
import json
from quantization.utils.inference_mod import InferProcessor
# from hw_emulation.Param_extractor import ModelParameterExtractor
from hw_emulation.model_introspector import StdModelInspector
from quantization.fix_ops import to_int_tensor

# ------------------- CONSTANTS -------------------
# Model and quantization config
MODEL_CHECKPOINT = "../qat_models/checkpoint/resnet50_best.pth.tar"
QUANT_CONFIG = "../quantization/utils/quant_config.yaml"

# Layer to test
LAYER_NAME = "layer1_0_conv2"
SUBSET_SHAPE = (64, 64, 3, 3)  # (out_channels, in_channels, H, W)

# Output files (write to project-level hw_data_files)
WEIGHT_FILE = "../hw_data_files/weights_64x64x3x3.data"
BIAS_FILE = "../hw_data_files/biases_1x64.data"
ACTIVATION_FILE = "../hw_data_files/test_image_64x64x64.data"
QPARAMS_FILE = "../hw_data_files/layer1_0_conv2_qparams.json"

# Test image (use a real image from ImageNet or a random tensor)
TEST_IMAGE_PATH = "new.JPEG"  # Change to a real image if available

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


def main():
    # Ensure output directories exist
    for path in {WEIGHT_FILE, BIAS_FILE, ACTIVATION_FILE, QPARAMS_FILE}:
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # 1. Load model and quantization config
    with open(QUANT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    from quantization.utils.graph_trace import QatProcessor
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()
    qat_proc.freeze()
    qat_proc.load_qat_weights(MODEL_CHECKPOINT)

    # 2. Convert to standard model
    infer_proc = InferProcessor(model, config)
    stdm = infer_proc.convert_to_std_model()
    # qconfig = infer_proc.generate_qconfig()

    # 3. Build inspector and save layer params
    inspector = StdModelInspector(stdm, default_input_frac=5, logger=logger)
    frac_w, frac_b = inspector.save_layer_params(
        name=LAYER_NAME,
        weight_file=WEIGHT_FILE,
        bias_file=BIAS_FILE,
        target_weight_shape=SUBSET_SHAPE,
        n_bits_out=8,
    )

    # 4. Capture a real activation and save INPUT activation for the layer
    if os.path.exists(TEST_IMAGE_PATH):
        input_tensor = preprocess_image(TEST_IMAGE_PATH)
    else:
        logger.warning(f"Test image {TEST_IMAGE_PATH} not found, using random tensor.")
        input_tensor = torch.rand(1, 3, 224, 224)

    inspector.register_activation_hooks([LAYER_NAME], capture_input=True, capture_output=True, clear_existing=True)
    with torch.no_grad():
        inspector.run_and_capture(input_tensor)

    # Determine frac_in and frac_out from inspector
    q = inspector.get_quant_params(LAYER_NAME)
    frac_out = int(q["frac_out"]) if q["frac_out"] is not None else 0
    frac_in_list = q["frac_in"] if isinstance(q.get("frac_in"), list) else []
    frac_in = int(frac_in_list[0]) if len(frac_in_list) > 0 else inspector.default_input_frac

    # Save activation (input to the layer)
    inspector.save_activation(
        name=LAYER_NAME,
        filepath=ACTIVATION_FILE,
        which="input",
        n_bits=8,
        n_frac=frac_in,
    )

    # Gather metadata for JSON
    mod = inspector.get_module(LAYER_NAME)
    # Activation input dims (first input if multiple)
    act_in_shapes = inspector.input_shapes.get(LAYER_NAME) or []
    # Prefer CHW to match saved activation layout (we drop batch if present)
    activation_in_dims = None
    if len(act_in_shapes) > 0:
        shape0 = tuple(act_in_shapes[0])
        if len(shape0) == 4 and shape0[0] == 1:
            activation_in_dims = (shape0[1], shape0[2], shape0[3])  # CHW
        elif len(shape0) == 3:
            activation_in_dims = shape0
        else:
            # Fallback: try to drop leading dim if >3
            activation_in_dims = tuple(shape0[-3:])

    # Effective saved weight dims (respecting subset)
    weight_dims = None
    if hasattr(mod, "weight") and mod.weight is not None:
        orig_w_shape = tuple(mod.weight.shape)
        if SUBSET_SHAPE is not None and len(SUBSET_SHAPE) == len(orig_w_shape):
            weight_dims = tuple(min(SUBSET_SHAPE[i], orig_w_shape[i]) for i in range(len(orig_w_shape)))
        else:
            weight_dims = orig_w_shape

    # Conv layer params
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

    # 5. Save quantization parameters + metadata
    qparams = {
        "layer": LAYER_NAME,
        "frac_w": int(frac_w) if frac_w is not None else 0,
        "frac_b": int(frac_b) if frac_b is not None else 0,
        "frac_in": int(frac_in),
        "frac_out": int(frac_out),
        "activation_in_dims": list(activation_in_dims) if activation_in_dims is not None else None,
        "weights_dim": list(weight_dims) if weight_dims is not None else None,
        "conv_params": conv_params,
    }
    with open(QPARAMS_FILE, "w") as f:
        json.dump(qparams, f, indent=2)
    logger.info(f"Saved quantization parameters to {QPARAMS_FILE}")

if __name__ == "__main__":
    main()
