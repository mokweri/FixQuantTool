"""Per-layer quantization sensitivity analysis.

Quantizes ONE layer at a time (all other quantizers pass float through) and
measures the top-1 drop on a small validation subset. Layers with the biggest
drops are the ones to consider for relaxed handling (see roadmap §10).

Usage:
    python tools/layer_sensitivity.py --model mobilenet_v2 --eval_batches 4
"""

import argparse
import os
import yaml
import torch
from pathlib import Path

from fixquant.data.imagenet import ImagenetDataProvider
from fixquant.graph.qat_processor import QatProcessor
from fixquant.models import get_model
from fixquant.quantization.tqt_quantizer import TQTQuantizer
from fixquant.utils import accuracy

parser = argparse.ArgumentParser(description="Layer-wise quantization sensitivity")
parser.add_argument("--model", type=str, default="mobilenet_v2")
parser.add_argument("--dataroot", type=str,
                    default=os.environ.get("FIXQUANT_DATA_DIR", "/home/obed/Documents/datasets/imagenet-mini"))
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--calib_batches", type=int, default=4)
parser.add_argument("--eval_batches", type=int, default=4,
                    help="Validation batches per probe (keep small; one probe per layer).")
parser.add_argument("--manual_seed", type=int, default=0)


@torch.no_grad()
def evaluate(net, loader, device, max_batches):
    net.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        acc1, _ = accuracy(net(images), labels, topk=(1, 5))
        correct += acc1[0].item() * images.size(0) / 100.0
        total += images.size(0)
    return 100.0 * correct / max(total, 1)


def owner_layers(model):
    """Group quantizers by the module that owns them."""
    groups = {}
    for name, mod in model.named_modules():
        qs = [q for q in (getattr(mod, "weight_quantizer", None),
                          getattr(mod, "bias_quantizer", None),
                          getattr(mod, "act_quantizer", None),
                          getattr(mod, "quantizer", None))
              if isinstance(q, TQTQuantizer)]
        if qs:
            groups[getattr(mod, "module_name", None) or name] = qs
    return groups


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ImagenetDataProvider.DEFAULT_PATH = args.dataroot
    provider = ImagenetDataProvider(test_batch_size=args.batch_size)
    calib_loader = provider.build_sub_train_loader(
        args.calib_batches * args.batch_size, args.batch_size)

    model = get_model(args.model, pretrained=True)
    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "configs/quant_config.yaml") as f:
        config = yaml.safe_load(f)

    proc = QatProcessor(model, config)
    qat_model = proc.quantize()
    proc.calibrate(calib_loader, device, max_batches=args.calib_batches)
    proc.freeze()
    qat_model.to(device)

    groups = owner_layers(qat_model)
    all_quantizers = [q for qs in groups.values() for q in qs]

    # Float reference (all quantizers off)
    for q in all_quantizers:
        q.enable_quant(False)
    float_acc = evaluate(qat_model, provider.test_loader, device, args.eval_batches)
    print(f"float reference top-1: {float_acc:.2f}%  ({len(groups)} layers to probe)\n")

    results = []
    for layer, qs in groups.items():
        for q in qs:
            q.enable_quant(True)
        acc = evaluate(qat_model, provider.test_loader, device, args.eval_batches)
        results.append((layer, float_acc - acc))
        print(f"  {layer:45s} drop {float_acc - acc:+6.2f}%")
        for q in qs:
            q.enable_quant(False)

    results.sort(key=lambda r: -r[1])
    print("\nMost sensitive layers:")
    for layer, drop in results[:15]:
        print(f"  {drop:+6.2f}%  {layer}")
