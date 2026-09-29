#!/usr/bin/env python3
"""Convert timm's pretrained EfficientNet-Lite0 into FixQuant's plain definition.

Run once, anywhere timm and network access are available; training then needs
neither. The converted model must reproduce timm's logits on seeded inputs, so
the conversion is verified rather than assumed.

    python tools/convert_timm_efficientnet_lite.py
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fixquant.models.efficientnet_lite import (  # noqa: E402
    EFFICIENTNET_LITE0_CHECKPOINT, EfficientNetLite)

TIMM_MODEL = "efficientnet_lite0.ra_in1k"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=EFFICIENTNET_LITE0_CHECKPOINT)
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="Largest accepted absolute logit difference")
    args = parser.parse_args()

    import timm
    reference = timm.create_model(TIMM_MODEL, pretrained=True).eval()
    state = reference.state_dict()
    model = EfficientNetLite().eval()
    model.load_state_dict(state, strict=True)

    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(8, 3, 224, 224, generator=generator)
    with torch.no_grad():
        expected, actual = reference(inputs), model(inputs)
    difference = (expected - actual).abs().max().item()
    agree = (expected.argmax(1) == actual.argmax(1)).all().item()
    print(f"max |logit difference| = {difference:.3e}; top-1 agreement: {agree}")
    if difference > args.tolerance or not agree:
        raise SystemExit("converted model does not reproduce timm's logits")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state,
                "source": {"timm_model": TIMM_MODEL, "timm_version": timm.__version__,
                           "pretrained_cfg": dict(reference.pretrained_cfg)},
                "verification": {"inputs": "8x3x224x224 standard normal, seed 0",
                                 "max_abs_logit_difference": difference}},
               args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}\nsha256 {digest}")


if __name__ == "__main__":
    main()
