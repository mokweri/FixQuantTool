#!/usr/bin/env python3
"""Re-estimate BatchNorm running statistics, then evaluate.

Diagnostic for architecture surgery on a batch-normalized model. Changing a
parameter-free op (pooling geometry, for instance) leaves every weight intact
but shifts the activation distribution each BatchNorm was calibrated against,
so the stale running mean/variance alone can account for a large accuracy
drop. This tool resets the running statistics, re-estimates them over a few
hundred training batches in train mode without taking a single gradient step,
and evaluates the result.

The gap between the plain evaluation and this one separates "the statistics are
stale" from "the representation is damaged" -- the first is recovered by any
fine-tuning, the second is not.
"""

import argparse
import logging

import torch
import torch.nn as nn

from fixquant.data.imagenet import ImagenetDataProvider
from fixquant.training import RunConfig, RunManager


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Float checkpoint to load instead of pretrained weights.")
    parser.add_argument("--batches", type=int, default=200,
                        help="Training batches used to re-estimate the statistics.")
    parser.add_argument("--momentum", type=float, default=None,
                        help="BatchNorm momentum during re-estimation. The default, "
                             "None, makes each layer accumulate a cumulative moving "
                             "average over all batches, which is what we want here.")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--n_worker", type=int, default=12)
    parser.add_argument("--valid_size", default=None)
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet"])
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--save_dir", default="./bn_recalibrate")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--validation_frequency", type=int, default=1)
    return parser


def reset_bn_statistics(model, momentum=None):
    """Clear every BatchNorm's running statistics and count of observed batches."""
    layers = [m for m in model.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    for bn in layers:
        bn.reset_running_stats()
        bn.momentum = momentum  # None => cumulative moving average
    return layers


@torch.no_grad()
def estimate_bn_statistics(model, loader, device, batches):
    """Run forward passes in train mode so BatchNorm accumulates statistics."""
    was_training = model.training
    model.train()
    model.to(device)
    seen = 0
    for images, _ in loader:
        if seen >= batches:
            break
        model(images.to(device, non_blocking=True))
        seen += 1
    model.train(was_training)
    return seen


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("bn_recalibrate")
    args = build_parser().parse_args()
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.manual_seed)

    ImagenetDataProvider.DEFAULT_PATH = args.dataroot

    from fixquant.models import get_model
    model = get_model(args.model, pretrained=True)
    if args.checkpoint:
        from fixquant.utils import load_float_checkpoint
        load_float_checkpoint(model, args.checkpoint)
        logger.info("Loaded float checkpoint %s", args.checkpoint)

    args_dict = args.__dict__.copy()
    args_dict["image_size"] = 224
    run_config = RunConfig(**args_dict, is_qat=False)
    run_config.print_config()
    run_manager = RunManager(args.save_dir, model, run_config)
    device = run_manager.device

    before = run_manager.validate(0)
    logger.info("Before re-estimation: loss=%.6f top1=%.4f top5=%.4f",
                *(float(v) for v in before))

    layers = reset_bn_statistics(run_manager.network, args.momentum)
    logger.info("Reset %d BatchNorm layers; re-estimating over %d batches of %d",
                len(layers), args.batches, args.train_batch_size)
    seen = estimate_bn_statistics(run_manager.network, run_config.train_loader,
                                  device, args.batches)
    logger.info("Re-estimated over %d batches (%d images)",
                seen, seen * args.train_batch_size)

    after = run_manager.validate(0)
    logger.info("After re-estimation:  loss=%.6f top1=%.4f top5=%.4f",
                *(float(v) for v in after))

    print(f"BN re-estimation: top1 {float(before[1]):.4f} -> {float(after[1]):.4f} "
          f"({float(after[1]) - float(before[1]):+.4f}), "
          f"top5 {float(before[2]):.4f} -> {float(after[2]):.4f}")

    if args.metrics_output:
        from fixquant.model_zoo import utc_now, write_json
        write_json(args.metrics_output, {
            "schema_version": 1,
            "created_at": utc_now(),
            "representation": "fp32",
            "model": args.model,
            "weights": (f"float checkpoint {args.checkpoint}" if args.checkpoint
                        else "torchvision pretrained (fixquant.models.get_model)"),
            "dataset": {"name": args.dataset, "path": args.dataroot},
            "validation_samples": len(run_config.val_loader.sampler),
            "bn_recalibration": {
                "batch_size": args.train_batch_size,
                "batches": seen,
                "images": seen * args.train_batch_size,
                "batchnorm_layers": len(layers),
            },
            "metrics_before": {
                "loss": float(before[0]), "top1": float(before[1]),
                "top5": float(before[2]),
            },
            "metrics": {
                "loss": float(after[0]), "top1": float(after[1]),
                "top5": float(after[2]),
            },
        })
        print(f"Metrics written to {args.metrics_output}")


if __name__ == "__main__":
    main()
