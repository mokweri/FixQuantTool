#!/usr/bin/env python3
"""Export a released QAT model as plain floating-point PyTorch.

Writes, under --out:

    model.py              standalone nn.Module source; imports only torch
    state_dict.pt         folded float weights and biases, no batch norm
    quantization.json     FixQuant's per-tensor fixed-point positions
    export_manifest.json  identities, preprocessing, conversion, verification
    reference/            inputs.pt and logits.pt for checking a loader

The QAT model is rebuilt exactly as tools/deploy_eval.py rebuilds it, the
checkpoint is verified against the release checksum and loaded strictly, and
batch norm is folded by freeze(). The architecture is constructed without
pretrained weights: every parameter comes from the QAT checkpoint.

Verification compares the exported model, loaded back from model.py in a fresh
namespace, with the QAT model computing the same thing (activation quantization
disabled; weight quantization too for --weights latent), on real preprocessed
images and synthetic inputs. With --evaluate, FP32 top-1/top-5
over an ImageNet-style validation directory is recorded as well.

Example:
    conda run -n Obed_Cuda python tools/export_float.py \\
        --release resnet18/imagenet1k/int8-tqt@v1.0.0 \\
        --out ../build/exports/float/resnet18 \\
        --evaluate /home/obed/Documents/datasets/imagenet-mini/val
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fixquant.export.float_exporter import (  # noqa: E402
    FORMAT_VERSION, WEIGHT_MODES, emit_source, quantization_disabled,
    quantization_record, to_float_graph,
)

DEFAULT_IMAGES = "/home/obed/Documents/datasets/imagenet-mini/val"
# Logits are compared absolutely; real ImageNet logits are O(10), so this is
# floating-point reassociation level, far below any change of prediction.
TOLERANCE = 1e-3


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repository_state(repo: Path):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
            text=True, stderr=subprocess.DEVNULL).strip())
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_frozen_qat_model(released: dict, config: dict):
    """The deployed QAT model: architecture, CLE as released, strict load, freeze."""
    from fixquant.graph.qat_processor import QatProcessor
    from fixquant.models import get_model

    model = get_model(released["model"], pretrained=False)
    if released["cle"]:
        from fixquant.quantization.equalization import equalize_model
        model = equalize_model(model, replace_relu6=not released["cle_keep_relu6"])
    proc = QatProcessor(model, config)
    proc.quantize()
    proc.load_qat_weights(released["checkpoint"])
    proc.freeze()
    return proc.qat_model.eval()


def eval_transform():
    """FixQuant's own validation transform: Resize(256), CenterCrop(224), normalize."""
    from fixquant.data.imagenet import ImagenetDataProvider
    provider = ImagenetDataProvider.__new__(ImagenetDataProvider)
    provider.image_size = 224
    return provider.build_valid_transform()


def image_items(root: str):
    """(path, label) for an ImageFolder-style tree, labels in sorted-synset order."""
    items = []
    for label, synset in enumerate(sorted(d for d in os.listdir(root)
                                          if os.path.isdir(os.path.join(root, d)))):
        folder = os.path.join(root, synset)
        for name in sorted(os.listdir(folder)):
            items.append((os.path.join(folder, name), label))
    return items


def reference_inputs(images_root: str, count: int, seed: int):
    """Real preprocessed images (first of every class in order) plus synthetic ones."""
    from PIL import Image
    tf = eval_transform()
    real, seen = [], set()
    for path, label in image_items(images_root):
        if label in seen:
            continue
        seen.add(label)
        real.append(tf(Image.open(path).convert("RGB")))
        if len(real) == count:
            break
    g = torch.Generator().manual_seed(seed)
    synthetic = [torch.randn(3, 224, 224, generator=g) for _ in range(4)]
    return torch.stack(real + synthetic)


def load_exported(out: Path, class_name: str):
    """Import model.py in a fresh module namespace and load state_dict.pt strictly."""
    spec = importlib.util.spec_from_file_location(f"exported_{class_name}", out / "model.py")
    module = importlib.util.module_from_spec(spec)
    # Keep the artifact directory free of a __pycache__ written by this import.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    model = getattr(module, class_name)()
    model.load_state_dict(torch.load(out / "state_dict.pt", map_location="cpu"), strict=True)
    return model.eval()


@torch.no_grad()
def evaluate(model, root: str, device: str, batch_size: int = 64, workers: int = 8):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class Items(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            return self.tf(Image.open(path).convert("RGB")), label

    loader = DataLoader(Items(image_items(root), eval_transform()), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True)
    model = model.to(device)
    n = t1 = t5 = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        top5 = model(x).topk(5, dim=1).indices  # ties go to the lower index
        t1 += (top5[:, 0] == y).sum().item()
        t5 += (top5 == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    model.to("cpu")
    return {"top1": 100.0 * t1 / n, "top5": 100.0 * t5 / n, "images": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, help="release id, e.g. resnet18/imagenet1k/int8-tqt@v1.0.0")
    ap.add_argument("--zoo-root", default=None, help="override FIXQUANT_ZOO_ROOT")
    ap.add_argument("--out", required=True, help="output directory (created; must be empty)")
    ap.add_argument("--format", default="torch", choices=["torch"],
                    help="artifact format (ONNX is planned)")
    ap.add_argument("--weights", default="deployed", choices=WEIGHT_MODES,
                    help="deployed: weights and biases after their quantizers, as "
                         "FixQuant deploys them (default); latent: QAT's float "
                         "parameters before weight quantization")
    ap.add_argument("--images", default=DEFAULT_IMAGES,
                    help="ImageNet-style directory for reference images")
    ap.add_argument("--reference-images", type=int, default=16)
    ap.add_argument("--evaluate", default=None,
                    help="also measure FP32 accuracy over this validation directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from fixquant.model_zoo import resolve_release
    released = resolve_release(args.zoo_root, args.release)
    if not released["checkpoint_available"]:
        sys.exit(f"checkpoint for {args.release} is missing or fails its sha256 check")
    checkpoint_sha = sha256_file(released["checkpoint"])
    if checkpoint_sha != released["checkpoint_sha256"]:
        sys.exit(f"checkpoint sha256 {checkpoint_sha} != release {released['checkpoint_sha256']}")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    (out / "reference").mkdir(parents=True, exist_ok=True)

    with open(REPO / "configs/quant_config.yaml") as f:
        config = yaml.safe_load(f)
    qat = build_frozen_qat_model(released, config)
    float_gm, conversions = to_float_graph(qat, weights=args.weights)

    class_name = "".join(part.capitalize() for part in released["model"].split("_")) + "Float"
    header = (f"{released['model']} as plain float PyTorch, exported by FixQuant "
              f"tools/export_float.py from {args.release}.\n\n"
              "Batch norm is folded into the convolutions and quantization is removed.\n"
              f"Weights: {args.weights} (see export_manifest.json).\n"
              "Load with: model = " + class_name + "(); "
              "model.load_state_dict(torch.load('state_dict.pt'))")
    (out / "model.py").write_text(emit_source(float_gm, class_name, header))
    torch.save(float_gm.state_dict(), out / "state_dict.pt")
    # FixQuant's own fixed-point positions, for a downstream quantizer that can
    # take them (Vitis AI Stage C2): with them, its INT8 model saturates and
    # rounds where FixQuant's does.
    (out / "quantization.json").write_text(json.dumps(
        {"schema": "fixquant.quantization_positions.v1", "tensors": quantization_record(qat)},
        indent=2) + "\n")

    # Verify: the exported model, loaded back without FixQuant, against the QAT
    # model with quantization disabled.
    inputs = reference_inputs(args.images, args.reference_images, seed=0)
    exported = load_exported(out, class_name)
    with torch.no_grad(), quantization_disabled(qat, weights=args.weights):
        want = qat(inputs)
    with torch.no_grad():
        got = exported(inputs)
    max_abs = (got - want).abs().max().item()
    agree = bool((got.argmax(1) == want.argmax(1)).all())
    # The same inputs through the quantized QAT model: shows the comparison
    # above really ran without quantization, and how far quantization moves it.
    with torch.no_grad():
        quantized = qat(inputs)
    quantized_max_abs = (got - quantized).abs().max().item()
    torch.save(inputs, out / "reference" / "inputs.pt")
    torch.save(got, out / "reference" / "logits.pt")
    print(f"{args.release} [{args.weights}]: max |exported - reference| = {max_abs:.3e} over "
          f"{inputs.shape[0]} inputs; top-1 agreement {agree} "
          f"(quantized QAT differs by up to {quantized_max_abs:.3f})")
    if max_abs > TOLERANCE or not agree:
        sys.exit("verification failed: exported model does not reproduce the float QAT model")

    accuracy = None
    if args.evaluate:
        accuracy = evaluate(exported, args.evaluate, args.device)
        print(f"FP32 export accuracy over {accuracy['images']} images: "
              f"top-1 {accuracy['top1']:.3f}, top-5 {accuracy['top5']:.3f}")

    revision, dirty = repository_state(REPO)
    manifest = {
        "schema": FORMAT_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "release_id": args.release,
        "model": released["model"],
        "profile": released["profile"],
        "cle": released["cle"],
        "cle_keep_relu6": released["cle_keep_relu6"],
        "weights": {
            "mode": args.weights,
            "meaning": ("weights and biases after their quantizers, as deployed, held in float"
                        if args.weights == "deployed" else
                        "QAT float parameters before weight quantization"),
        },
        "checkpoint": {"path": os.path.relpath(released["checkpoint"], REPO),
                       "sha256": checkpoint_sha},
        "fixquant": {"revision": revision, "dirty": dirty},
        "torch_version": torch.__version__,
        "class_name": class_name,
        "input": {"shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"},
        "output": {"shape": [1, 1000], "meaning": "logits, ImageNet-1k classes in sorted-synset order"},
        "preprocessing": {
            "color": "RGB", "resize_short_side": 256, "long_side": "int(256 * long / short)",
            "interpolation": "bilinear (PIL, antialiased)", "center_crop": 224,
            "scale": "x / 255", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        },
        "conversion": conversions,
        "verification": {
            "reference": ("QAT model with activation quantization disabled"
                          if args.weights == "deployed" else
                          "QAT model with all fake quantization disabled"),
            "inputs": int(inputs.shape[0]), "real_images": args.reference_images,
            "max_abs_logit_difference": max_abs, "tolerance": TOLERANCE,
            "top1_agreement": agree,
            "max_abs_logit_difference_to_quantized_qat": quantized_max_abs,
        },
        "fp32_accuracy": accuracy and {**accuracy, "images_root": args.evaluate},
        "files": {name: sha256_file(out / name) for name in
                  ("model.py", "state_dict.pt", "quantization.json",
                   "reference/inputs.pt", "reference/logits.pt")},
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
