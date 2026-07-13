"""
export_uram_refactor_testcases.py
==================================
Exports six single-convolution hardware test cases needed for the URAM
datapath refactor.  Each case packages:

  inputs/   – INT8 boundary input activation
  params/   – INT8 weight + bias
  refs/     – bit-exact reference output (TileCNN hardware arithmetic)
  graph.json

Target layers
-------------
Case                        Conv                   K    Stride   Notes
────────────────────────────────────────────────────────────────────────
best_case_3x3_s1        layer2_1_conv2          3×3    1      115.6M MACs – best-case anchor
stride2_3x3_s2          layer2_0_conv2          3×3    2      115.6M MACs – stride-2 3×3 penalty
proj_bottleneck_1x1_s2  layer2_0_downsample_0   1×1    2      102.8M MACs – projection bottleneck
small_proj_1x1_s2       layer4_0_downsample_0   1×1    2      102.8M MACs – small-spatial projection
deep_k1_s1              layer3_1_conv1          1×1    1      51.4M MACs – 1×1 stride-1 conv deep
late_k1_s1              layer4_1_conv1          1×1    1      51.4M MACs – 1×1 stride-1 conv late

Each test case includes the convolution node plus its immediately
following HLSRelu node (so the exported reference matches the
post-ReLU activation that feeds the next hardware stage).
"""

import argparse
import logging
import torch
import yaml
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision.models as models

from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter


# ---------------------------------------------------------------------------
# Subgraph definitions
# ---------------------------------------------------------------------------
# Each entry contains:
#   nodes       – the ordered list of emu-model node names that form the
#                 subgraph.  The first node's input becomes the exported
#                 boundary input; the last node's output becomes the
#                 exported reference.
#   description – human-readable label
#   macs        – approximate MACs for logging/documentation
#   hw_note     – design-space rationale
#
# Node name convention (from InferProcessor.convert_to_hardware_model):
#   - conv nodes:  <module_path with dots replaced by _>
#   - relu nodes:  <preceding_conv_name>_relu_<N>  (inserted by converter)
# ---------------------------------------------------------------------------
REFACTOR_TEST_CASES = {
    "best_case_3x3_s1": {
        "nodes": ["layer2_1_conv2", "layer2_1_relu_1"],
        "description": "3×3 stride-1 conv — best-case URAM throughput anchor",
        "macs": "115.6M",
        "hw_note": "Largest achievable utilisation; baseline for new weight path",
    },
    "stride2_3x3_s2": {
        "nodes": ["layer2_0_conv2", "layer2_0_relu_1"],
        "description": "3×3 stride-2 conv — stride-2 throughput penalty",
        "macs": "115.6M",
        "hw_note": "Same weight/channel count as above but stride-2 halves spatial output",
    },
    "proj_bottleneck_1x1_s2": {
        "nodes": ["layer2_0_downsample_0"],
        "description": "1×1 stride-2 projection shortcut — channel-expansion bottleneck",
        "macs": "102.8M",
        "hw_note": "No relu immediately after in graph; reference is raw conv output",
    },
    "small_proj_1x1_s2": {
        "nodes": ["layer4_0_downsample_0"],
        "description": "1×1 stride-2 projection shortcut on small 14×14 spatial — small-spatial bottleneck",
        "macs": "102.8M",
        "hw_note": "Quarter the spatial size of layer2_0 equivalent; tests address-gen edge case",
    },
    "deep_k1_s1": {
        "nodes": ["layer3_1_conv1", "layer3_1_relu_0"],
        "description": "1×1 stride-1 conv deep in the network",
        "macs": "51.4M",
        "hw_note": "New 1x1 stride-1 testcase for layer3_1_conv1",
    },
    "late_k1_s1": {
        "nodes": ["layer4_1_conv1", "layer4_1_relu_0"],
        "description": "1×1 stride-1 conv late in the network",
        "macs": "51.4M",
        "hw_note": "New 1x1 stride-1 testcase for layer4_1_conv1",
    },
}

PHASE3_TEST_CASES = {
    "layer3_0_conv2": {
        "nodes": ["layer3_0_conv2", "layer3_0_relu_1"],
        "description": "layer3_0_conv2",
        "macs": "Unknown",
        "hw_note": "Phase 3 test case",
    },
    "layer3_1_conv2": {
        "nodes": ["layer3_1_conv2", "layer3_1_relu_1"],
        "description": "layer3_1_conv2",
        "macs": "Unknown",
        "hw_note": "Phase 3 test case",
    },
    "layer4_0_conv2": {
        "nodes": ["layer4_0_conv2", "layer4_0_relu_1"],
        "description": "layer4_0_conv2",
        "macs": "Unknown",
        "hw_note": "Phase 3 test case",
    },
    "layer4_1_conv2": {
        "nodes": ["layer4_1_conv2", "layer4_1_relu_1"],
        "description": "layer4_1_conv2",
        "macs": "Unknown",
        "hw_note": "Phase 3 test case",
    },
}


def preprocess_image(image_path: str) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return t(Image.open(image_path).convert("RGB")).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(
        description="Export URAM-refactor target single-conv hardware test cases"
    )
    parser.add_argument("--checkpoint",   default=None, help="Path to QAT checkpoint .pth.tar")
    parser.add_argument("--cle", action="store_true", default=False,
                        help="Apply cross-layer equalization before quantizing (match the checkpoint's training).")
    parser.add_argument("--quant_config", default=None, help="Path to quant_config.yaml")
    parser.add_argument("--image",        default=None, help="Path to test image (JPEG/PNG)")
    parser.add_argument("--out_dir",      default=None, help="Output base directory")
    ALL_TEST_CASES = {**REFACTOR_TEST_CASES, **PHASE3_TEST_CASES}
    
    parser.add_argument(
        "--test_case", default="all",
        choices=list(ALL_TEST_CASES.keys()) + ["all", "phase3"],
        help="Which test case to export, or 'all', or 'phase3' (default is 'all')",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logger = logging.getLogger("export_uram_refactor_testcases")

    REPO_ROOT    = Path(__file__).resolve().parents[1]
    checkpoint   = Path(args.checkpoint)   if args.checkpoint   else REPO_ROOT / "qat_models/resnet50/checkpoint/model_best.pth.tar"
    quant_config = Path(args.quant_config) if args.quant_config else REPO_ROOT / "configs/quant_config.yaml"
    image_path   = Path(args.image)        if args.image        else REPO_ROOT / "assets/new.JPEG"
    out_dir      = Path(args.out_dir)      if args.out_dir      else REPO_ROOT / "outputs/phase3_testcases"

    # ------------------------------------------------------------------
    # Build the HLS emulation model (the correct source for hw testcases)
    # ------------------------------------------------------------------
    logger.info("Loading quantization config…")
    with open(quant_config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Building ResNet-50 QAT model…")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if args.cle:
        from fixquant.quantization.equalization import equalize_model
        model = equalize_model(model)
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()

    if checkpoint.exists():
        logger.info(f"Loading checkpoint: {checkpoint}")
        qat_proc.load_qat_weights(str(checkpoint))
    else:
        logger.warning(f"Checkpoint not found at {checkpoint} — using default (untrained) weights")
    qat_proc.freeze()

    logger.info("Converting to HLS emulation model…")
    infer_proc = InferProcessor(model, config)
    emu_model  = infer_proc.convert_to_hardware_model()

    inspector = StdModelInspector(emu_model,
                                  default_input_frac=infer_proc.input_frac or 5,
                                  logger=logger)

    # ------------------------------------------------------------------
    # One forward pass with hooks to capture all activations / shapes
    # ------------------------------------------------------------------
    if image_path.exists():
        logger.info(f"Preprocessing test image: {image_path}")
        inp = preprocess_image(str(image_path))
    else:
        logger.warning(f"Image not found at {image_path} — using random input")
        inp = torch.rand(1, 3, 224, 224)

    logger.info("Running forward pass to capture activations…")
    all_nodes = inspector.topological_order()
    inspector.register_activation_hooks(
        all_nodes,
        capture_input=True,
        capture_output=True,
        clear_existing=True,
    )
    with torch.no_grad():
        inspector.run_and_capture(inp)

    logger.info("Forward pass complete — %d nodes captured", len(all_nodes))

    # ------------------------------------------------------------------
    # Exporter (shared across all cases)
    # ------------------------------------------------------------------
    exporter = TileCNNGraphExporter(
        inspector=inspector,
        model_name="resnet50_uram_refactor",
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Export loop
    # ------------------------------------------------------------------
    if args.test_case == "all":
        cases_to_run = list(ALL_TEST_CASES.keys())
    elif args.test_case == "phase3":
        cases_to_run = list(PHASE3_TEST_CASES.keys())
    else:
        cases_to_run = [args.test_case]

    for case_name in cases_to_run:
        case_def = ALL_TEST_CASES[case_name]
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Test case : {case_name}")
        logger.info(f"Layer(s)  : {case_def['nodes']}")
        logger.info(f"Desc      : {case_def['description']}")
        logger.info(f"MACs      : {case_def['macs']}")
        logger.info(f"HW note   : {case_def['hw_note']}")

        # Validate that every requested node exists in the graph
        valid_nodes = [n for n in case_def["nodes"] if n in all_nodes]
        missing     = set(case_def["nodes"]) - set(valid_nodes)
        if missing:
            logger.warning(f"  Nodes not found in graph (skipped): {missing}")
        if not valid_nodes:
            logger.error(f"  No valid nodes for '{case_name}' — skipping entirely")
            continue

        case_out_dir = out_dir / case_name
        exporter.export(str(case_out_dir), subgraph_nodes=valid_nodes)
        logger.info(f"  → Exported to {case_out_dir}")

    logger.info("")
    logger.info("All requested test cases exported successfully.")
    logger.info(f"Output root: {out_dir}")


if __name__ == "__main__":
    main()
