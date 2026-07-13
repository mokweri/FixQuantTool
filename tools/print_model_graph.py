import os
import argparse
import logging
import yaml
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector


def preprocess_image(image_path: str):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return t(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Print and dump standard model graph details")
    parser.add_argument("--model", default="resnet50",
                        help="Model to inspect (resnet18|resnet50|vgg16|mobilenet_v2)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cle", action="store_true", default=False,
                        help="Apply cross-layer equalization before quantizing (match the checkpoint's training).")
    parser.add_argument("--quant_config", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--out_txt", default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--list_layers_only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("print_model_graph")

    # Resolve repo root as parent of this file's directory
    REPO_ROOT = Path(__file__).resolve().parents[1]
    # Defaults
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        REPO_ROOT / f"qat_models/{args.model}/checkpoint/model_best.pth.tar")
    quant_config = Path(args.quant_config) if args.quant_config else (REPO_ROOT / "configs/quant_config.yaml")
    image_path = Path(args.image) if args.image else (REPO_ROOT / "assets/new.JPEG")
    out_txt = Path(args.out_txt) if args.out_txt else (REPO_ROOT / f"outputs/{args.model}_model_graph.txt")
    out_json = Path(args.out_json) if args.out_json else (REPO_ROOT / f"outputs/{args.model}_model_graph.json")

    # Ensure output dirs
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Load model and quantization config
    with open(quant_config, "r") as f:
        config = yaml.safe_load(f)

    from fixquant.models import get_model
    model = get_model(args.model, pretrained=True)
    if args.cle:
        from fixquant.quantization.equalization import equalize_model
        model = equalize_model(model)
    qat_proc = QatProcessor(model, config)
    model = qat_proc.quantize()
    if checkpoint.exists():
        qat_proc.load_qat_weights(str(checkpoint))
    else:
        logger.warning(f"Checkpoint not found at {checkpoint}; using default weights.")
    qat_proc.freeze()

    infer_proc = InferProcessor(model, config)
    hw_model = infer_proc.convert_to_hardware_model()

    inspector = StdModelInspector(hw_model,
                                  default_input_frac=infer_proc.input_frac or 5,
                                  logger=logger)

    # Print a simple ordered list of Conv/Linear layers for quick selection
    ordered_layers = inspector.topological_order()
    conv_linear = [n for n in ordered_layers if isinstance(inspector.get_module(n), (torch.nn.Conv2d, torch.nn.Linear))]
    print("Conv/Linear layers in topological order:")
    for i, n in enumerate(conv_linear):
        print(f"[{i:03d}] {n}")

    if args.list_layers_only:
        return

    # Prepare input
    if image_path.exists():
        inp = preprocess_image(str(image_path))
    else:
        logger.warning("Image '%s' not found; using random input.", image_path)
        inp = torch.rand(1, 3, 224, 224)

    # Collect shapes then dump graph
    inspector.collect_all_shapes(inp)
    inspector.dump_graph_text(str(out_txt))
    inspector.dump_graph_json(str(out_json))
    print(f"Wrote: {out_txt}\nWrote: {out_json}")


if __name__ == "__main__":
    main()
