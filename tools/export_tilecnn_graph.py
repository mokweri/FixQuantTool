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
    parser.add_argument("--model", default="resnet50",
                        help="Model to export (resnet18|resnet50|vgg16|mobilenet_v2)")
    parser.add_argument("--checkpoint", default=None, help="Path to best QAT checkpoint")
    parser.add_argument("--cle", action="store_true", default=False,
                        help="Apply cross-layer equalization before quantizing (match the checkpoint's training).")
    parser.add_argument("--quant_config", default=None, help="Path to quant_config.yaml")
    parser.add_argument("--image", default=None, help="Path to test image for activation export")
    parser.add_argument("--out_dir", default=None, help="Output directory for TileCNN graph")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_tilecnn_graph")

    # Resolve repo root as parent of this file's directory
    REPO_ROOT = Path(__file__).resolve().parents[1]

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        REPO_ROOT / f"qat_models/{args.model}/checkpoint/model_best.pth.tar")
    quant_config = Path(args.quant_config) if args.quant_config else (REPO_ROOT / "configs/quant_config.yaml")
    image_path = Path(args.image) if args.image else (REPO_ROOT / "assets/new.JPEG")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / f"outputs/{args.model}_int8_tilecnn")

    logger.info("Loading quantization config...")
    with open(quant_config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Building and quantizing the model...")
    from fixquant.models import get_model
    model = get_model(args.model, pretrained=True)
    if args.cle:
        from fixquant.quantization.equalization import equalize_model
        model = equalize_model(model)
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()

    if checkpoint.exists():
        logger.info(f"Loading checkpoint from {checkpoint}")
        qat_proc.load_qat_weights(str(checkpoint))
    else:
        logger.warning(f"Checkpoint not found at {checkpoint}. Exporting with default weights!")
    qat_proc.freeze()

    logger.info("Converting to bit-exact emulation model...")
    infer_proc = InferProcessor(model, config)
    tilecnn_model = infer_proc.convert_to_hardware_model()

    inspector = StdModelInspector(tilecnn_model,
                                  default_input_frac=infer_proc.input_frac or 5,
                                  logger=logger)

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
        model_name=args.model,
        default_input_frac=infer_proc.input_frac or 5,
        logger=logger
    )
    exporter.export(str(out_dir))

    logger.info("Export completed successfully!")

if __name__ == "__main__":
    main()
