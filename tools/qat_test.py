import argparse
import os
import numpy as np
import random
import torchvision.models as models
import torch
from torch.utils.checkpoint import checkpoint
import platform
import yaml
import logging

from fixquant.data import Cifar10DataProvider
from fixquant.data.imagenet import ImagenetDataProvider
from fixquant.training import RunConfig, RunManager
from fixquant.graph.qat_processor import QatProcessor
from fixquant.models.cifar import *

parser = argparse.ArgumentParser(description="FixQuant Tool")

# Hyperparameters
parser.add_argument("--train_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--valid_size", default=None)

# Performance options
parser.add_argument("--n_worker", type=int, default=8,
                    help='Number of Workers')
parser.add_argument("--pin-memory", default=True, action="store_true")
parser.add_argument("--device", type=torch.device, default="cuda")
parser.add_argument('--gpus',
                    type=str, default='0', help='gpu ids to be used for training, seperated by commas')

# Misc. options
parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--dataroot", type=str, default=os.environ.get("FIXQUANT_DATA_DIR", "/home/obed/Documents/datasets/imagenet-mini"), )

parser.add_argument('--display_freq',
                    default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--validation_frequency',
                    default=1, type=int, help='Validate model every n epochs.')
parser.add_argument('--save_dir',
                    default='./qat_models', help='Directory to save trained models.')
parser.add_argument('--output_dir',
                    default='qat_result', help='Directory to save qat result.')
parser.add_argument('--manual_seed', default=0, type=int, help='Seed.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # Set desired logging level
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and args.cuda else "cpu"

    """Calibration Dataset"""
    ImagenetDataProvider.DEFAULT_PATH = os.environ.get(
        "FIXQUANT_DATA_DIR", args.dataroot
    )

    data_provider = ImagenetDataProvider()
    # data_provider = Cifar10DataProvider()

    calib_loader = data_provider.build_sub_train_loader(24, 24)

    """Imagenet models"""
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    """cifar models"""
    # model = resnet18_cifar10()

    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "configs/quant_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    Qatprocessor = QatProcessor(model, config)
    model = Qatprocessor.quantize()
    Qatprocessor.calibrate(calib_loader, device)
    # Qatprocessor.load_qat_weights(str(repo_root / 'qat_models/checkpoint/vgg16_best.pth.tar'))
    Qatprocessor.load_qat_weights(str(repo_root / 'qat_models/checkpoint/mobilenet_v2_best.pth.tar'))
    Qatprocessor.freeze()

    args_dict = args.__dict__.copy()
    if 'image_size' not in args_dict:
        args_dict['image_size'] = 224

    run_config = RunConfig(**args_dict, is_qat=True)
    run_config.print_config()
    run_manager = RunManager(args.save_dir, model, run_config)
    # with torch.autograd.set_detect_anomaly(True):
    #     run_manager.train()
    #
    run_manager.validate(0)
