import argparse
import numpy as np
import random
import torchvision.models as models
import torch
from torch.utils.checkpoint import checkpoint
import platform
import yaml
import logging

from data_providers import Cifar10DataProvider
from data_providers.imagenet import ImagenetDataProvider
from run_manager import RunConfig, RunManager
from quantization.utils.graph_trace import QatProcessor
from models.cifar_models import *

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
parser.add_argument("--dataroot", type=str, default="/home/obed/Documents/imagenet-mini", )

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
    if platform.system() == "Windows":
        ImagenetDataProvider.DEFAULT_PATH = r"C:\Users\oma02\Downloads\imagenet-mini"
    elif platform.system() == "Linux":
        ImagenetDataProvider.DEFAULT_PATH = "/home/obed/Documents/imagenet-mini"
    else:
        raise RuntimeError("Unsupported OS")

    data_provider = ImagenetDataProvider()
    # data_provider = Cifar10DataProvider()

    calib_loader = data_provider.build_sub_train_loader(24, 24)

    """Imagenet models"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    """cifar models"""
    # model = resnet18_cifar10()

    with open("quantization/utils/quant_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    Qatprocessor = QatProcessor(model, config)
    model = Qatprocessor.quantize()
    Qatprocessor.calibrate(calib_loader, device)
    Qatprocessor.freeze()
    Qatprocessor.load_qat_weights('qat_models/checkpoint/resnet50_best.pth.tar')
    """ NOTE:
        Call Qatprocessor.freeze() before loading resnet50weights 
        - it behaves normal - frozen during training -- resnet18 and vgg16 does not behave
        Call Qatprocessor.freeze() after loading vgg16/resnet18weights 
    """
    # Qatprocessor.freeze()

    run_config = RunConfig(**args.__dict__, is_qat=True)
    run_config.print_config()
    run_manager = RunManager(args.save_dir, model, run_config)
    # with torch.autograd.set_detect_anomaly(True):
    #     run_manager.train()
    #
    run_manager.validate(0)
