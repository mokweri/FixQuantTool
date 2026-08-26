import argparse
import hashlib
import json
import logging
import subprocess
import yaml
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms

from fixquant import __version__
from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter


PREPROCESSING = {
    "profile": "fixquant.export-reference.v1",
    "color_mode": "RGB",
    "resize": {"height": 224, "width": 224},
    "tensor_range": [0.0, 1.0],
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_state(repo_root: Path):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def artifact_record(package_dir: Path, relative_path: str):
    path = package_dir / relative_path
    return {"path": relative_path, "sha256": sha256_file(path)}


def write_package_manifest(
        out_dir: Path,
        repo_root: Path,
        args,
        checkpoint: Path,
        quant_config: Path,
        image_path: Path,
        release=None):
    with (out_dir / "graph.json").open("r", encoding="utf-8") as handle:
        graph = json.load(handle)

    revision, dirty = repository_state(repo_root)
    input_paths = sorted({
        spec["file"] for spec in graph["tensors"].values()
        if spec.get("kind") == "input" and spec.get("file")
    })
    parameter_paths = sorted({
        spec["file"] for spec in graph["tensors"].values()
        if spec.get("kind") == "param" and spec.get("file")
    })
    reference_paths = sorted({
        spec["file"] for spec in graph["tensors"].values()
        if spec.get("kind") == "reference" and spec.get("file")
    })

    identity = {"name": args.model}
    if release:
        identity.update({
            "release_id": release["release_id"],
            "dataset": release["dataset"]["name"],
            "quantization_profile": release["profile"],
        })

    checkpoint_record = {
        "file": checkpoint.name,
        "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
    }
    expected_checkpoint_hash = (
        release.get("checkpoint_sha256") if release else None
    )
    if expected_checkpoint_hash:
        checkpoint_record["release_sha256"] = expected_checkpoint_hash

    image_record = {
        "file": image_path.name,
        "sha256": sha256_file(image_path) if image_path.is_file() else None,
    }
    manifest = {
        "schema": "tilecnn.model-package.v1",
        "model": identity,
        "producer": {
            "name": "FixQuant",
            "version": __version__,
            "git_revision": revision,
            "git_dirty": dirty,
        },
        "sources": {
            "checkpoint": checkpoint_record,
            "quantization_config": {
                "file": quant_config.name,
                "sha256": sha256_file(quant_config),
            },
            "reference_image": image_record,
        },
        "preprocessing": PREPROCESSING,
        "integrity": {
            "algorithm": "sha256",
            "coverage": "all-graph-artifacts",
        },
        "artifacts": {
            "graph": artifact_record(out_dir, "graph.json"),
            "parameters": [
                artifact_record(out_dir, path) for path in parameter_paths
            ],
            "inputs": [artifact_record(out_dir, path) for path in input_paths],
            "references": [
                artifact_record(out_dir, path) for path in reference_paths
            ],
        },
        "reference": {
            "implementation": "FixQuant TileCNN bit-exact integer reference",
            "arithmetic": "fused TileCNN graph semantics",
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest


def preprocess_image(image_path: str):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return t(img).unsqueeze(0)


def build_parser():
    parser = argparse.ArgumentParser(description="Export TileCNN standard model graph and params")
    parser.add_argument("--model", default="resnet50",
                        help="Model to export (resnet18|resnet50|vgg16|mobilenet_v2)")
    parser.add_argument("--checkpoint", default=None, help="Path to best QAT checkpoint")
    parser.add_argument("--cle", action="store_true", default=False,
                        help="Apply cross-layer equalization before quantizing (match the checkpoint's training).")
    parser.add_argument("--zoo-model", default=None,
                        help="Released model ID: model/dataset/profile@version")
    parser.add_argument("--zoo-root", default=None,
                        help="Override FIXQUANT_ZOO_ROOT for --zoo-model")
    parser.add_argument("--quant_config", default=None, help="Path to quant_config.yaml")
    parser.add_argument("--image", default=None, help="Path to test image for activation export")
    parser.add_argument("--out_dir", default=None, help="Output directory for TileCNN graph")
    return parser


def resolve_export_model(args, repo_root: Path) -> Path:
    if args.zoo_model:
        from fixquant.model_zoo import resolve_release

        released = resolve_release(args.zoo_root, args.zoo_model)
        args.zoo_release = released
        args.model = released["model"]
        args.checkpoint = released["checkpoint"]
        args.cle = released["cle"]

    return Path(args.checkpoint) if args.checkpoint else (
        repo_root / f"qat_models/{args.model}/checkpoint/model_best.pth.tar")


def main():
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_tilecnn_graph")

    # Resolve repo root as parent of this file's directory
    REPO_ROOT = Path(__file__).resolve().parents[1]

    checkpoint = resolve_export_model(args, REPO_ROOT)
    release = getattr(args, "zoo_release", None)
    if args.zoo_model:
        logger.info("Resolved model zoo release %s", args.zoo_model)
    quant_config = Path(args.quant_config) if args.quant_config else (REPO_ROOT / "configs/quant_config.yaml")
    image_path = Path(args.image) if args.image else (REPO_ROOT / "assets/new.JPEG")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / f"outputs/{args.model}_int8_tilecnn")

    if args.zoo_model and not checkpoint.is_file():
        raise FileNotFoundError(
            f"Released checkpoint not found: {checkpoint}. Fetch it with "
            f"'scripts/model_zoo.sh fetch {args.zoo_model}'"
        )
    expected_checkpoint_hash = release.get("checkpoint_sha256") if release else None
    if expected_checkpoint_hash:
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != expected_checkpoint_hash:
            raise RuntimeError(
                "Released checkpoint checksum mismatch: "
                f"expected {expected_checkpoint_hash}, got {checkpoint_hash}"
            )

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
    write_package_manifest(
        out_dir, REPO_ROOT, args, checkpoint, quant_config, image_path, release
    )

    logger.info("Export completed successfully!")

if __name__ == "__main__":
    main()
