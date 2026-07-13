import argparse
import os
import numpy as np
import random
import torchvision.models as models
import torch

from fixquant.training import RunConfig, RunManager
from fixquant.models.cifar import *

parser = argparse.ArgumentParser(description="FixQuant Tool")

# Hyperparameters
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--valid_size", default=None)
parser.add_argument('--n_epochs', default=150, type=int, help='No. of training epochs.')
parser.add_argument('--warmup-epochs', type=float, default=0, help='number of warmup epochs')
parser.add_argument('--warmup_lr',type=float, default=-1, metavar='LR', help='warmup learning rate')
parser.add_argument('--init_lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--no_nesterov', default=False)
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument("--train_criterion", type=str, default="ce",choices=["ce"])
parser.add_argument("--test_criterion", type=str, default="ce",choices=["ce"])
parser.add_argument("--lr_schedule_type", type=str, default="cosine",choices=["cosine"])

# Performance options
parser.add_argument("--n_worker", type=int, default=8, help='Number of Workers')
parser.add_argument("--pin-memory", default=True, action="store_true")
parser.add_argument("--device", type=torch.device, default="cuda")
parser.add_argument('--gpus', type=str, default='0',
                    help='gpu ids to be used for training, seperated by commas')

# Model / misc. options
parser.add_argument("--model", type=str, default="vgg16",
                    help="Model to train (resnet18|resnet50|vgg16|mobilenet_v2)")
parser.add_argument("--pretrained", action="store_true", default=False,
                    help="Start from torchvision pretrained weights.")
parser.add_argument("--eval_only", action="store_true", default=False,
                    help="Skip training and only run validation.")
parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--dataroot", type=str,
                    default=os.environ.get("FIXQUANT_DATA_DIR", "/home/obed/Documents/data"),)

parser.add_argument('--display_freq',
                    default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--validation_frequency',
                    default=1, type=int, help='Validate model every n epochs.')
parser.add_argument('--save_dir',
                    default='./models', help='Directory to save trained models.')
parser.add_argument('--manual_seed',
                    default=0, type=int, help='Seed.')

"""
    Float (non-QAT) training / evaluation. Used to establish the float
    baselines in docs/baselines.md.
"""

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and args.cuda else "cpu"

    from fixquant.models import get_model
    model = get_model(args.model, pretrained=args.pretrained)

    args.save_dir = os.path.join(args.save_dir, args.model)

    run_config = RunConfig(**args.__dict__, is_qat=False, image_size=224)
    run_config.print_config()

    run_manager = RunManager(args.save_dir, model, run_config)
    if not args.eval_only:
        run_manager.train()

    run_manager.validate(0)


