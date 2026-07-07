import os
import argparse
import logging
import yaml
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter

def preprocess_image(image_path: str):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return t(img).unsqueeze(0)

# Define subgraph test cases by the list of node names they encompass
# Note: Node names are derived from the PyTorch module names in the converted StdModel
TEST_CASES = {
    "single_conv_relu": {
        "nodes": ["conv1", "relu_0"],
        "description": "Standard 7x7 conv with ReLU"
    },
    "single2_conv_relu": {
        "nodes": ["layer1_1_conv2", "layer1_1_relu_1"],
        "description": "Standard 3x3 conv with ReLU"
    },
    "conv_pool": {
        "nodes": ["conv1", "relu_0", "maxpool"],
        "description": "Conv + ReLU + MaxPool"
    },
    "residual_block": {
        "nodes": ["layer1_1_conv1", "layer1_1_relu_0", "layer1_1_conv2", "layer1_1_relu_1", "layer1_1_conv3", "add_1", "layer1_1_relu_2"],
        "description": "Full residual block with identity Add (layer 1.1)"
    },
    "residual_block_downsample": {
        "nodes": ["layer2_0_conv1", "layer2_0_relu_0", "layer2_0_conv2", "layer2_0_relu_1", "layer2_0_conv3", "layer2_0_downsample_0", "add_3", "layer2_0_relu_2"],
        "description": "Full residual block with 1x1 projection shortcut (layer 2.0)"
    },
    "tail_gap_fc": {
        "nodes": ["layer4_2_conv3", "add_15", "layer4_2_relu_2", "avgpool", "flatten", "fc"],
        "description": "Final block convolution, GAP and FC layer"
    }
}

TEST_CASES2 = {
    "conv_stem": {
        "nodes": ["conv1", "relu_0"],
        "description": "Conv + ReLU"
    },
}


def main():
    parser = argparse.ArgumentParser(description="Export TileCNN subgraph testcases")
    parser.add_argument("--checkpoint", default=None, help="Path to best QAT checkpoint")
    parser.add_argument("--quant_config", default=None, help="Path to quant_config.yaml")
    parser.add_argument("--image", default=None, help="Path to test image")
    parser.add_argument("--out_dir", default=None, help="Output base directory")
    parser.add_argument("--test_case", default="all", help="Which test case to export, or 'all'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_hw_testcases")

    REPO_ROOT = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint) if args.checkpoint else (REPO_ROOT / "qat_models/checkpoint/resnet50_best.pth.tar")
    quant_config = Path(args.quant_config) if args.quant_config else (REPO_ROOT / "configs/quant_config.yaml")
    image_path = Path(args.image) if args.image else (REPO_ROOT / "assets/new.JPEG")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "outputs/hw_testcases")

    logger.info("Loading config and building standard inference model...")
    with open(quant_config, "r") as f:
        config = yaml.safe_load(f)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()
    qat_proc.freeze()
    
    if checkpoint.exists():
        qat_proc.load_qat_weights(str(checkpoint))

    infer_proc = InferProcessor(model, config)
    emu_model = infer_proc.convert_to_hardware_model()
    inspector = StdModelInspector(emu_model, default_input_frac=5, logger=logger)

    if image_path.exists():
        inp = preprocess_image(str(image_path))
    else:
        inp = torch.rand(1, 3, 224, 224)

    logger.info("Running forward pass to capture shapes and hooks...")
    # Get all nodes so we capture everything
    all_nodes = inspector.topological_order()
    inspector.register_activation_hooks(all_nodes, capture_input=True, capture_output=True, clear_existing=True)
    with torch.no_grad():
        inspector.run_and_capture(inp)

    exporter = TileCNNGraphExporter(
        inspector=inspector,
        model_name="resnet50_subgraph",
        logger=logger
    )

    cases_to_run = TEST_CASES2.keys() if args.test_case == "all" else [args.test_case]
    
    for case_name in cases_to_run:
        if case_name not in TEST_CASES2:
            logger.error(f"Test case {case_name} not found.")
            continue
            
        case_def = TEST_CASES2[case_name]
        logger.info(f"--- Exporting Test Case: {case_name} ---")
        logger.info(f"Description: {case_def['description']}")
        
        # Verify nodes exist in model
        valid_nodes = [n for n in case_def["nodes"] if n in all_nodes]
        if len(valid_nodes) != len(case_def["nodes"]):
            missing = set(case_def["nodes"]) - set(valid_nodes)
            logger.warning(f"Some nodes were not found in the graph: {missing}")
        
        case_out_dir = out_dir / case_name
        exporter.export(str(case_out_dir), subgraph_nodes=valid_nodes)

if __name__ == "__main__":
    main()
