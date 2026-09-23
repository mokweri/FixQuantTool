import argparse
import logging

import torch

from fixquant.data.imagenet import ImagenetDataProvider
from fixquant.training import RunConfig, RunManager


parser = argparse.ArgumentParser(
    description="Evaluate the floating-point reference model on the same data "
                "pipeline as tools/qat_test.py and tools/deploy_eval.py")

# Hyperparameters
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--valid_size", default=None)

parser.add_argument("--test_criterion", type=str, default="ce", choices=["ce"])

# Performance options
parser.add_argument("--n_worker", type=int, default=8,
                    help='Number of Workers')
parser.add_argument("--pin-memory", default=True, action="store_true")
parser.add_argument("--device", type=torch.device, default="cuda")
parser.add_argument('--gpus',
                    type=str, default='0', help='gpu ids to be used for evaluation, seperated by commas')

# Model options
parser.add_argument("--model", type=str, default="resnet50",
                    help="Model to evaluate (resnet18|resnet50|vgg16|mobilenet_v2). "
                         "Uses the same pretrained weights QAT starts from. CLE is "
                         "function-preserving in floating point, so it is not applied.")
parser.add_argument("--checkpoint", default=None,
                    help="Float checkpoint to evaluate instead of the torchvision "
                         "pretrained weights. Required for models whose head is "
                         "trained here (e.g. vgg16_tilecnn).")
parser.add_argument("--zoo-model", default=None,
                    help="Released model ID: model/dataset/profile@version. Selects the "
                         "model and the dataset recorded for that release.")
parser.add_argument("--zoo-root", default=None,
                    help="Override FIXQUANT_ZOO_ROOT for --zoo-model")
parser.add_argument("--metrics-output", default=None,
                    help="Write machine-readable evaluation metrics JSON")

# Misc. options
parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet"])
parser.add_argument("--dataroot", type=str,
                    default=None,
                    help=(
                        "Dataset path. Precedence: this option, FIXQUANT_DATA_DIR, "
                        "released-model path, local fallback."
                    ))

parser.add_argument('--display_freq',
                    default=100, type=int, help='Display metrics every n steps.')
parser.add_argument('--validation_frequency',
                    default=1, type=int, help='Validate model every n epochs.')
parser.add_argument('--save_dir',
                    default='./fp32_eval', help='Directory for run output.')
parser.add_argument('--manual_seed',
                    default=0, type=int, help='Seed.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fp32_eval")

    args = parser.parse_args()
    release_dataset_path = None
    if args.zoo_model:
        from fixquant.model_zoo import resolve_release
        released = resolve_release(args.zoo_root, args.zoo_model)
        args.model = released["model"]
        release_dataset_path = released["dataset"].get("path")
    from fixquant.model_zoo import select_dataset_path
    args.dataroot = select_dataset_path(
        args.dataroot,
        release_dataset_path,
        fallback="/home/obed/Documents/datasets/imagenet-mini",
    )
    args.cuda = torch.cuda.is_available()

    ImagenetDataProvider.DEFAULT_PATH = args.dataroot

    from fixquant.models import get_model
    model = get_model(args.model, pretrained=True)
    weights_description = "torchvision pretrained (fixquant.models.get_model)"
    if args.checkpoint:
        from fixquant.utils import load_float_checkpoint
        loaded = load_float_checkpoint(model, args.checkpoint)
        weights_description = f"float checkpoint {args.checkpoint}"
        logger.info("Loaded float checkpoint %s (epoch %s, val top1 %s)",
                    args.checkpoint, loaded.get("epoch"), loaded.get("val_top1"))

    args_dict = args.__dict__.copy()
    if 'image_size' not in args_dict:
        args_dict['image_size'] = 224

    # is_qat only selects the TQT optimizer and learning-rate schedule; the
    # validation loader and preprocessing are identical to the QAT evaluation.
    run_config = RunConfig(**args_dict, is_qat=False)
    run_config.print_config()
    run_manager = RunManager(args.save_dir, model, run_config)
    loss, top1, top5 = run_manager.validate(0)
    print(
        f"FP32 evaluation: loss={float(loss):.6f}, "
        f"top1={float(top1):.4f}, top5={float(top5):.4f}"
    )
    if args.metrics_output:
        from fixquant.model_zoo import utc_now, write_json
        write_json(args.metrics_output, {
            "schema_version": 1,
            "created_at": utc_now(),
            "representation": "fp32",
            "model": args.model,
            "weights": weights_description,
            "dataset": {"name": args.dataset, "path": args.dataroot},
            "validation_samples": len(run_config.val_loader.sampler),
            "metrics": {
                "loss": float(loss),
                "top1": float(top1),
                "top5": float(top5),
            },
        })
        print(f"Metrics written to {args.metrics_output}")
