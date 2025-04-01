import argparse
import numpy as np
import random
import torchvision.models as models
import torch
from torch.utils.checkpoint import checkpoint
import platform

from data_providers import Cifar10DataProvider
from data_providers.imagenet import ImagenetDataProvider
from run_manager import RunConfig, RunManager
from quantization.utils.graph_editing import create_quantized_model, freeze, calibrate
from quantization.utils.inference_model import make_inference_model
from models.cifar_models import *

parser = argparse.ArgumentParser(description="FixQuant Tool")

# Hyperparameters
parser.add_argument("--train_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--valid_size", default=None)
parser.add_argument('--n_epochs',
                    default=2, type=int, help='No. of training epochs.')
parser.add_argument('--warmup-epochs', type=float, default=0,
                    help='number of warmup epochs')
parser.add_argument('--warmup_lr', type=float,
                    default=-1, metavar='LR', help='warmup learning rate')
parser.add_argument('--init_lr', '--learning-rate',
                    default=1e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--quantizer_lr',
                    default=1e-2, type=float, help='Initial learning rate of quantizer.')
parser.add_argument('--quantizer_lr_decay',
                    default=0.5, type=int, help='Learning rate decay ratio of quantizer.')

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

# Misc. options
parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--dataroot", type=str,
                    default="/home/obed/Documents/imagenet-mini", )

parser.add_argument('--display_freq',
                    default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--validation_frequency',
                    default=1, type=int, help='Validate model every n epochs.')
parser.add_argument('--save_dir',
                    default='./qat_models', help='Directory to save trained models.')
parser.add_argument('--output_dir',
                    default='qat_result', help='Directory to save qat result.')
parser.add_argument('--manual_seed',
                    default=0, type=int, help='Seed.')

if __name__ == '__main__':
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
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    """cifar models"""
    # model = resnet18_cifar10()

    checkpoint = torch.load('qat_models/checkpoint/model_best.pth.tar')

    model = create_quantized_model(model, verbose=False)
    # freeze(model)
    calibrate(model, calib_loader)

    model.load_state_dict(checkpoint['state_dict'])
    freeze(model)

    # print(model)
    # trying to save the model
    # example_input = torch.randn(1, 3, 224, 224).cuda()

    # inf_model = make_inference_model(model)

    # print(inf_model)

    run_config = RunConfig(**args.__dict__, is_qat=True)
    run_config.print_config()

    run_manager = RunManager(args.save_dir, model, run_config)
    # with torch.autograd.set_detect_anomaly(True):
    #     run_manager.train()
    #
    run_manager.validate(0)
