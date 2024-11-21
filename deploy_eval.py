import argparse
import torchvision.models as models
import torch

from data_providers.imagenet import ImagenetDataProvider
from data_providers.cifar10 import Cifar10DataProvider
from run_manager import RunConfig, RunManager
from quantization.utils.graph_editing import create_quantized_model,freeze,calibrate
from models.cifar_models import *

parser = argparse.ArgumentParser(description="FixQuant Tool")

# Hyperparameters
parser.add_argument("--train_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--valid_size", default=None)

parser.add_argument("--test_criterion", type=str, default="ce",choices=["ce"])

# Performance options
parser.add_argument("--n_worker", type=int, default=8,
                    help='Number of Workers')
parser.add_argument("--pin-memory", default=True, action="store_true")
parser.add_argument("--device", type=torch.device, default="cuda")
parser.add_argument('--gpus',
                    type=str, default='0', help='gpu ids to be used for training, seperated by commas')

# Misc. options
parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--dataroot", type=str,
                    default="/mimer/NOBACKUP/groups/naiss2024-22-1034/PipeCNN_Interface/dataset/imagenet",)

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

    """Imagenet models"""
    ImagenetDataProvider.DEFAULT_PATH = '/home/obed/Documents/imagenet'
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    """Cifar models"""
    # model = resnet18_cifar10()

    model = create_quantized_model(model, verbose=False)
    freeze(model)

    checkpoint = torch.load('qat_models/checkpoint/vgg16-imagenet.tar')
    model.load_state_dict(checkpoint['state_dict'])

    run_config = RunConfig(**args.__dict__, is_qat=False)
    run_config.print_config()

    run_manager = RunManager(args.save_dir, model, run_config)
    run_manager.validate(0)