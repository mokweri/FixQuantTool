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

def main():
    parser = argparse.ArgumentParser(description="Export TileCNN standard model graph and params")
    parser.add_argument("--checkpoint", default=None, help="Path to best QAT checkpoint")
    parser.add_argument("--quant_config", default=None, help="Path to quant_config.yaml")
    parser.add_argument("--image", default=None, help="Path to test image for activation export")
    parser.add_argument("--out_dir", default=None, help="Output directory for TileCNN graph")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_tilecnn_graph")

    # Resolve repo root as parent of this file's directory
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    checkpoint = Path(args.checkpoint) if args.checkpoint else (REPO_ROOT / "qat_models/checkpoint/resnet50_best.pth.tar")
    quant_config = Path(args.quant_config) if args.quant_config else (REPO_ROOT / "configs/quant_config.yaml")
    image_path = Path(args.image) if args.image else (REPO_ROOT / "assets/new.JPEG")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "outputs/resnet50_int8_tilecnn")

    logger.info("Loading quantization config...")
    with open(quant_config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Building and quantizing the model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()
    qat_proc.freeze()
    
    if checkpoint.exists():
        logger.info(f"Loading checkpoint from {checkpoint}")
        qat_proc.load_qat_weights(str(checkpoint))
    else:
        logger.warning(f"Checkpoint not found at {checkpoint}. Exporting with default weights!")

    logger.info("Converting to bit-exact emulation model...")
    infer_proc = InferProcessor(model, config)
    tilecnn_model = infer_proc.convert_to_hardware_model()

    inspector = StdModelInspector(tilecnn_model, default_input_frac=5, logger=logger)

    # Prepare input
    if image_path.exists():
        logger.info(f"Preprocessing test image {image_path}...")
        inp = preprocess_image(str(image_path))
    else:
        logger.warning(f"Image '{image_path}' not found; using random input.")
        inp = torch.rand(1, 3, 224, 224)

    logger.info("Collecting graph shapes with a forward pass...")
    inspector.collect_all_shapes(inp)

    logger.info(f"Exporting TileCNN artifacts to {out_dir}...")
    exporter = TileCNNGraphExporter(
        inspector=inspector,
        model_name="resnet50",
        default_input_frac=5,
        logger=logger
    )
    exporter.export(str(out_dir))

    logger.info("Export completed successfully!")

if __name__ == "__main__":
    main()
