import argparse
import numpy as np
import os
import random
import torchvision.models as models

# using for distributed training
import horovod.torch as hvd
import torch

from data_providers.imagenet import ImagenetDataProvider
from data_providers.cifar10 import Cifar10DataProvider
from data_providers.cifar100 import Cifar100DataProvider
from run_manager import (
    DistributedClassificationRunConfig,
    DistributedRunManager,
    ClassificationRunConfig,
    RunManager)

parser = argparse.ArgumentParser(description="FixQuant Tool")

# Hyperparameters
parser.add_argument('--epochs',
                    default=100, type=int, help='No. of training epochs.')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument('--base_lr', '--learning-rate',
                    default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--no_nesterov', default=False)
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--train_criterion", type=str, default="ce",choices=["ce"])
parser.add_argument("--test_criterion", type=str, default="ce",choices=["ce"])

# Performance options
parser.add_argument("--num-workers", type=int, default=4,
                    help='Workers')
parser.add_argument("--pin-memory", default=True, action="store_true")
parser.add_argument("--device", type=torch.device, default="cuda")
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")

# Horovod Settings
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--dy_conv_scaling_mode', type=int, default=1,
                    help='dy_conv_scaling_mode')
parser.add_argument('--independent_distributed_sampling',default=False,
                    help='independent_distributed_sampling')
parser.add_argument('--dynamic_batch_size',default=1,
                    help='dynamic_batch_size')

# Misc. options
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--dataroot", type=str,
                    default="/mimer/NOBACKUP/groups/naiss2024-22-1034/PipeCNN_Interface/dataset/imagenet",)

parser.add_argument('--display_freq',
                    default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--val_freq',
                    default=100000, type=int, help='Validate model every n steps.')  # for imagenet increase it
parser.add_argument('--save_dir',
                    default='./qat_models', help='Directory to save trained models.')
parser.add_argument('--output_dir',
                    default='qat_result', help='Directory to save qat result.')

args = parser.parse_args()
if args.dataset =="imagenet":
    args.image_size = "224"
else:
    args.image_size = "32"

args.path = './qat_models/' + args.dataset + "/"

if __name__ == "__main__":
    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    num_gpus = hvd.size()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    run_config = DistributedClassificationRunConfig(
        **args.__dict__, num_replicas=num_gpus, rank=hvd.rank()
    )
    # print run config information
    if hvd.rank() == 0:
        print("Run config:")
        for k, v in run_config.config.items():
            print("\t%s: %s" % (k, v))

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    """ ============= Distributed RunManager ====================="""
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path,
        model,
        run_config,
        compression,
        backward_steps=args.dynamic_batch_size,
        is_root=(hvd.rank() == 0),
    )
    distributed_run_manager.save_config()
    distributed_run_manager.broadcast()

    distributed_run_manager.train(args)

    """ ================ Evaluation - Classification ==================== """
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]

    ImagenetDataProvider.DEFAULT_PATH = args.dataroot

    run_config = ClassificationRunConfig(dataset=args.dataset, test_batch_size=args.batch_size, n_worker=args.workers)
    print("Run config:")
    for k, v in run_config.config.items():
        print("\t%s: %s" % (k, v))
    run_manager = RunManager(".tmp/net", model, run_config, init=False)
    loss, (top1, top5) = run_manager.validate(net=model, is_test=True)
    print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f,\t robust1=%.1f,\t robust5=%.1f" % (loss, top1, top5))
