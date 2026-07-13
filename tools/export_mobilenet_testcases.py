"""Export real MobileNet-V2 inverted-residual subgraphs as TileCNN hardware test cases.

Companion to ``export_hw_testcases.py`` (which exports ResNet-50 subgraphs). The
TileCNN accelerator fuses a pointwise conv into the following depthwise conv
(pw -> dw); the *graph* we hand off is the plain sequential order (expand pw,
depthwise, project pw, optional residual add) and the TileCNN graph compiler does
the pw->dw fusion. So here we simply select the block's nodes in normal order and
let the exporter emit them; each case exercises a different layer setting.

The checkpoint is CLE-trained (BN-free, ReLU6->ReLU), so the model is equalized
before quantization — same as ``qat_train.py --cle``. Subgraph inputs and golden
references come from a real forward pass on an image, using the bit-exact TileCNN
integer kernels (post 2026-07 round-half-up fix).

    python tools/export_mobilenet_testcases.py            # all 5 cases
    python tools/export_mobilenet_testcases.py --test_case mbv2_residual_block
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
import torchvision.transforms as transforms
from PIL import Image

from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter
from fixquant.models import get_model


# Five MobileNet-V2 blocks chosen for hardware-relevant diversity. Node names are
# the converted-hardware-model names (features_<block>_conv_<idx>...); an inverted
# residual is expand-pw (conv.0.0) -> depthwise (conv.1.0) -> project-pw (conv.2),
# except block 1 which has expand_ratio=1 (no expand pw: dw is conv.0.0).
TEST_CASES = {
    "mbv2_first_block": {
        "nodes": ["features_1_conv_0_0", "features_1_conv_0_2_0", "features_1_conv_1"],
        "description": "Block 1 (expand_ratio=1): depthwise 3x3 s1 (32ch) -> relu -> pointwise 32->16 @112x112. "
                       "No expansion pw, no residual; standalone dw feeding a pw at the largest resolution.",
    },
    "mbv2_expand_dw2_project": {
        "nodes": ["features_2_conv_0_0", "features_2_conv_0_2_0",
                  "features_2_conv_1_0", "features_2_conv_1_2_0", "features_2_conv_2"],
        "description": "Block 2: pw 16->96 -> dw 3x3 s2 (96ch, 112->56) -> pw 96->24. "
                       "pw->dw fusion with stride-2 downsample, no residual.",
    },
    "mbv2_residual_block": {
        "nodes": ["features_3_conv_0_0", "features_3_conv_0_2_0",
                  "features_3_conv_1_0", "features_3_conv_1_2_0", "features_3_conv_2", "add"],
        "description": "Block 3: pw 24->144 -> dw 3x3 s1 (144ch, 56x56) -> pw 144->24 -> residual add. "
                       "pw->dw fusion plus the fused residual add (add's skip input is the block input).",
    },
    "mbv2_wide_dw2": {
        "nodes": ["features_14_conv_0_0", "features_14_conv_0_2_0",
                  "features_14_conv_1_0", "features_14_conv_1_2_0", "features_14_conv_2"],
        "description": "Block 14: pw 96->576 -> dw 3x3 s2 (576ch, 14->7) -> pw 576->160. "
                       "Wide channels with stride-2 downsample at small resolution, no residual.",
    },
    "mbv2_widest_residual": {
        "nodes": ["features_16_conv_0_0", "features_16_conv_0_2_0",
                  "features_16_conv_1_0", "features_16_conv_1_2_0", "features_16_conv_2", "add_9"],
        "description": "Block 16: pw 160->960 -> dw 3x3 s1 (960ch, 7x7) -> pw 960->160 -> residual add. "
                       "Widest channels (960) with residual add at the smallest spatial size.",
    },
}


def preprocess_image(image_path: str):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return t(Image.open(image_path).convert("RGB")).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Export MobileNet-V2 TileCNN subgraph testcases")
    parser.add_argument("--checkpoint", default=None, help="Path to MobileNet-V2 QAT checkpoint")
    parser.add_argument("--no_cle", action="store_true", default=False,
                        help="Skip cross-layer equalization (only for a non-CLE checkpoint; "
                             "the default mobilenet checkpoint is CLE-trained).")
    parser.add_argument("--quant_config", default=None, help="Path to quant_config.yaml")
    parser.add_argument("--image", default=None, help="Path to test image")
    parser.add_argument("--out_dir", default=None, help="Output base directory")
    parser.add_argument("--test_case", default="all",
                        help="Which test case to export, or 'all'. Choices: " + ", ".join(TEST_CASES))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_mobilenet_testcases")

    REPO_ROOT = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        REPO_ROOT / "qat_models/mobilenet_v2/checkpoint/model_best.pth.tar")
    quant_config = Path(args.quant_config) if args.quant_config else (REPO_ROOT / "configs/quant_config.yaml")
    image_path = Path(args.image) if args.image else (REPO_ROOT / "assets/new.JPEG")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "outputs/mobilenet_hw_testcases")

    logger.info("Loading config and building MobileNet-V2 inference model...")
    with open(quant_config, "r") as f:
        config = yaml.safe_load(f)

    model = get_model("mobilenet_v2", pretrained=True)
    if not args.no_cle:
        from fixquant.quantization.equalization import equalize_model
        model = equalize_model(model)
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()

    if checkpoint.exists():
        qat_proc.load_qat_weights(str(checkpoint))
    else:
        logger.warning(f"Checkpoint not found at {checkpoint}; exporting with default weights!")
    qat_proc.freeze()

    infer_proc = InferProcessor(model, config)
    emu_model = infer_proc.convert_to_hardware_model()
    inspector = StdModelInspector(emu_model,
                                  default_input_frac=infer_proc.input_frac or 5,
                                  logger=logger)

    if image_path.exists():
        inp = preprocess_image(str(image_path))
    else:
        logger.warning("No test image; using random input.")
        inp = torch.rand(1, 3, 224, 224)

    logger.info("Running forward pass to capture activations for subgraph inputs/refs...")
    all_nodes = inspector.topological_order()
    inspector.register_activation_hooks(all_nodes, capture_input=True, capture_output=True, clear_existing=True)
    with torch.no_grad():
        inspector.run_and_capture(inp)

    exporter = TileCNNGraphExporter(
        inspector=inspector,
        model_name="mobilenet_v2_subgraph",
        logger=logger,
    )

    cases_to_run = TEST_CASES.keys() if args.test_case == "all" else [args.test_case]
    for case_name in cases_to_run:
        if case_name not in TEST_CASES:
            logger.error(f"Test case {case_name} not found. Available: {', '.join(TEST_CASES)}")
            continue

        case_def = TEST_CASES[case_name]
        logger.info(f"--- Exporting Test Case: {case_name} ---")
        logger.info(f"Description: {case_def['description']}")

        valid_nodes = [n for n in case_def["nodes"] if n in all_nodes]
        if len(valid_nodes) != len(case_def["nodes"]):
            missing = set(case_def["nodes"]) - set(valid_nodes)
            logger.error(f"Nodes not found in the graph: {missing}; skipping {case_name}.")
            continue

        exporter.export(str(out_dir / case_name), subgraph_nodes=valid_nodes)

    logger.info(f"Done. Test cases written under {out_dir}")


if __name__ == "__main__":
    main()
