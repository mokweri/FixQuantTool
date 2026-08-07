import argparse
import os
import numpy as np
import random
import torchvision.models as models
import torch
from torch.utils.checkpoint import checkpoint
import platform
import sys
import yaml
import logging

from fixquant.data import Cifar10DataProvider
from fixquant.data.imagenet import ImagenetDataProvider
from fixquant.training import RunConfig, RunManager
from fixquant.graph.qat_processor import QatProcessor
from fixquant.models.cifar import *

parser = argparse.ArgumentParser(description="FixQuant Tool")

# Hyperparameters
parser.add_argument("--train_batch_size", type=int, default=50)
parser.add_argument("--test_batch_size", type=int, default=50)
parser.add_argument("--valid_size", default=None)
parser.add_argument('--n_epochs',
                    default=10, type=int, help='No. of training epochs.')
parser.add_argument('--warmup-epochs', type=float, default=0,
                    help='number of warmup epochs')
parser.add_argument('--warmup_lr', type=float,
                    default=-1, metavar='LR', help='warmup learning rate')
parser.add_argument('--init_lr', '--learning-rate',
                    default=1e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--quantizer_lr',
                    default=1e-2, type=float, help='Initial learning rate of quantizer.')
parser.add_argument('--quantizer_lr_decay',
                    default=0.5, type=float, help='Learning rate decay ratio of quantizer.')
parser.add_argument('--threshold_freeze_frac', default=0.7, type=float,
                    help='Fraction of epochs after which TQT thresholds are frozen.')

parser.add_argument('--momentum',
                    default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--no_nesterov', default=False)
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--train_criterion", type=str, default="ce", choices=["ce"])
parser.add_argument("--test_criterion", type=str, default="ce", choices=["ce"])
parser.add_argument("--lr_schedule_type", type=str, default="cosine", choices=["cosine"])

# Performance options
parser.add_argument("--n_worker", type=int, default=8,
                    help='Number of Workers')
parser.add_argument("--image_size", type=int, default=224,
                    help='Input crop size (224 for ImageNet).')
parser.add_argument("--pin-memory", default=True, action="store_true")
parser.add_argument("--device", type=torch.device, default="cuda")
parser.add_argument('--gpus',
                    type=str, default='0', help='gpu ids to be used for training, seperated by commas')

# Horovod Settings
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--independent_distributed_sampling', default=False,
                    help='independent_distributed_sampling')
parser.add_argument('--dynamic_batch_size', default=1,
                    help='dynamic_batch_size')

# Model / quantization options
parser.add_argument("--model", type=str, default="mobilenet_v2",
                    help="Model to quantize (resnet18|resnet50|vgg16|mobilenet_v2)")
parser.add_argument("--calib_batches", type=int, default=20,
                    help="Number of calibration batches (batch size = train_batch_size).")
parser.add_argument("--calib_scope", type=int, default=5,
                    help="MSE fix-position search width during calibration.")
parser.add_argument("--cle", action="store_true", default=False,
                    help="Fold BN and apply cross-layer equalization before QAT "
                         "(replaces ReLU6 with ReLU; recommended for MobileNet).")
parser.add_argument("--bias_corr", action="store_true", default=False,
                    help="Apply empirical bias correction after calibration (requires --cle).")

# Misc. options
parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--dataroot", type=str,
                    default=os.environ.get("FIXQUANT_DATA_DIR", "/home/obed/Documents/datasets/imagenet-mini"), )

parser.add_argument('--display_freq',
                    default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--validation_frequency',
                    default=1, type=int, help='Validate model every n epochs.')
parser.add_argument('--save_dir',
                    default='../qat_models', help='Directory to save trained models.')
parser.add_argument('--output_dir',
                    default='qat_result', help='Directory to save qat result.')
parser.add_argument('--manual_seed',
                    default=0, type=int, help='Seed.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # Reproducibility
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and args.cuda else "cpu"

    # keep checkpoints of different models separate
    args.save_dir = os.path.join(args.save_dir, args.model)
    os.makedirs(args.save_dir, exist_ok=True)

    from pathlib import Path
    from fixquant.model_zoo import git_commit, utc_now, write_yaml
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = Path(args.save_dir) / "run_manifest.yaml"
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "model": {
            "name": args.model,
            "initialization": "torchvision pretrained weights",
        },
        "dataset": {
            "name": "imagenet1k" if args.dataset == "imagenet" else args.dataset,
            "path": os.path.abspath(args.dataroot),
        },
        "quantization": {
            "method": "tqt",
            "weight_bits": 8,
            "activation_bits": 8,
            "cle": args.cle,
            "bias_correction": args.bias_corr,
            "calibration_batches": args.calib_batches,
            "calibration_scope": args.calib_scope,
            "threshold_freeze_fraction": args.threshold_freeze_frac,
        },
        "training": {
            "epochs": args.n_epochs,
            "train_batch_size": args.train_batch_size,
            "test_batch_size": args.test_batch_size,
            "workers": args.n_worker,
            "image_size": args.image_size,
            "initial_lr": args.init_lr,
            "quantizer_lr": args.quantizer_lr,
            "weight_decay": args.weight_decay,
            "seed": args.manual_seed,
        },
        "provenance": {
            "git_commit": git_commit(repo_root),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "command": [sys.executable, *sys.argv],
        },
    }
    write_yaml(manifest_path, run_manifest)

    """Calibration Dataset"""
    ImagenetDataProvider.DEFAULT_PATH = os.environ.get(
        "FIXQUANT_DATA_DIR", args.dataroot
    )

    data_provider = ImagenetDataProvider(
        save_path=args.dataroot,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        n_worker=args.n_worker,
        image_size=args.image_size,
        pin_memory=True,
    )
    run_manifest["dataset"].update({
        "train_samples": len(data_provider.train_loader.dataset),
        "validation_samples": len(data_provider.val_loader.sampler),
    })
    write_yaml(manifest_path, run_manifest)

    n_calib_images = args.calib_batches * args.train_batch_size
    calib_loader = data_provider.build_sub_train_loader(n_calib_images, args.train_batch_size)

    from fixquant.models import get_model
    model = get_model(args.model, pretrained=True)

    config_path = repo_root / "configs/quant_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    float_ref = None
    if args.cle:
        from fixquant.quantization.equalization import equalize_model
        model = equalize_model(model)
        float_ref = model  # BN-free, equalized float reference for bias correction

    Qatprocessor = QatProcessor(model, config)
    model = Qatprocessor.quantize()
    Qatprocessor.calibrate(calib_loader, device,
                           max_batches=args.calib_batches, scope=args.calib_scope)

    if args.bias_corr:
        if float_ref is None:
            raise SystemExit("--bias_corr requires --cle (needs a BN-free float reference).")
        from fixquant.quantization.equalization import apply_bias_correction
        apply_bias_correction(float_ref, model, calib_loader, device)

    from fixquant.diagnostics import quantizer_report, write_report_csv
    report_path = os.path.join(args.save_dir, f"{args.model}_calib_report.csv")
    write_report_csv(quantizer_report(model), report_path)
    print(f"Post-calibration quantizer report written to {report_path}")

    run_config = RunConfig(**args.__dict__, is_qat=True)
    run_config.print_config()
    run_manager = RunManager(args.save_dir, model, run_config)

    """ NOTE:
        During QAT of VGG16 if you encounter CUDA out of Memory Issue,
        reduce the train batch size to 50 or less
    """
    training_summary = run_manager.train()

    final_loss, final_top1, final_top5 = run_manager.validate(0)
    run_manifest.update({
        "status": "completed",
        "completed_at": utc_now(),
        "best": {
            "epoch": (
                training_summary["best_epoch"] + 1
                if training_summary["best_epoch"] is not None else None
            ),
            "metrics": training_summary["best_metrics"],
        },
        "final_evaluation": {
            "loss": float(final_loss),
            "top1": float(final_top1),
            "top5": float(final_top5),
        },
        "artifacts": {
            "best_checkpoint": "checkpoint/model_best.pth.tar",
            "latest_checkpoint": "checkpoint/latest.pth.tar",
            "calibration_report": os.path.basename(report_path),
            "threshold_log": "logs/quant_thresholds.csv",
        },
    })
    write_yaml(manifest_path, run_manifest)
    print(f"Run manifest written to {manifest_path}")
