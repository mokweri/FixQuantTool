import argparse
import torchvision.models as models
import torch
import torch.nn as nn
import platform
from PIL import Image
from collections import OrderedDict
from typing import Dict, Any
from pathlib import Path
import torchvision.transforms as transforms
import yaml
import logging
from torch.utils.data.datapipes.gen_pyi import iterDP_files_to_exclude

from models.cifar_models import *
from quantization.fix_ops import to_int_tensor, to_float_tensor, fake_quantize_tensor
from quantization.utils.graph_trace import QatProcessor
from quantization.utils.inference_mod import InferProcessor

from data_providers.imagenet import ImagenetDataProvider
from data_providers.cifar10 import Cifar10DataProvider
from run_manager import RunConfig, RunManager


parser = argparse.ArgumentParser(description="FixQuant Tool")

# Hyperparameters
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
    logging.basicConfig(level=logging.DEBUG)  # Set desired logging level

    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)

        return tensor

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and args.cuda else "cpu"

    """Set Dataset"""
    if platform.system() == "Windows":
        ImagenetDataProvider.DEFAULT_PATH = r"C:\Users\oma02\Downloads\imagenet-mini"
    elif platform.system() == "Linux":
        ImagenetDataProvider.DEFAULT_PATH = "/home/obed/Documents/imagenet-mini"
    else:
        raise RuntimeError("Unsupported OS")

    """Imagenet models"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    """Cifar models"""
    # model = resnet18_cifar10()

    """ ---- QUANTIZATION PROCESSING -----"""
    with open("quantization/utils/quant_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    Qatprocessor = QatProcessor(model, config)
    model = Qatprocessor.quantize()
    Qatprocessor.freeze()
    Qatprocessor.load_qat_weights('qat_models/checkpoint/resnet50_best.pth.tar')
    """ NOTE:
            Call Qatprocessor.freeze() before loading resnet50weights 
            - it behaves normal - frozen during training -- resnet18 and vgg16 does not behave
            Call Qatprocessor.freeze() after loading vgg16/resnet18weights 
        """
    # Qatprocessor.freeze()

    """ ---- INFERENCE PROCESSING -----"""
    infer_processor = InferProcessor(model, config)
    stdm = infer_processor.convert_to_std_model()
    # infer_processor.export_onnx_with_layer_metadata("res.onnx")
    qconfig = infer_processor.generate_qconfig()
    print(qconfig)

    infer_processor.export_weights_to_file()

    """ ---- TEST GENERATION  -----"""
    """Extract a subset of a layer"""
    layer_name = "conv1"
    subset_shape = (16, 3, 2, 2)  # (out_channels, in_channels, H, W)

    fp_w, fp_b, frac_w, frac_b = infer_processor.extract_and_subset_layer_parameters(
        layer_name=layer_name,
        output_filename=f"{layer_name}_subset.data",
        target_weight_shape=subset_shape
    )
    if fp_w is not None:
        print(f"Returned FP Weight for '{layer_name}': shape={fp_w.shape}, dtype={fp_w.dtype}")
    if fp_b is not None:
        print(f"Returned FP Bias for '{layer_name}': shape={fp_b.shape}, dtype={fp_b.dtype}")

    # ---------------------------------------------
    # test_image = preprocess_image("new.JPEG")
    # # test_image = to_int_tensor(test_image, signed=True, n_bits=8, n_frac=5)
    # test_image = fake_quantize_tensor(test_image, signed=True, n_bits=8, n_frac=5)
    # print(test_image)
    # # Save processed test image to a .data file
    # # test_image.numpy().astype('int8').tofile("hw_outputs/test_image.data")
    # # print(test_image)
    # pred = emu_model(test_image)
    # # Save model's output to a .data file
    # # pred = to_float_tensor(pred,n_frac=2)
    # pred = to_int_tensor(pred, n_frac=2)
    # # pred = fake_quantize_tensor(pred, signed=True, n_bits=8, n_frac=2)
    # pred.detach().numpy().astype('int8').tofile("hw_outputs/ref_output.data")
    # # print(pred)
    # # print(torch.argmax(pred))
    # # print(inf_model)
    # ---------------------------------------------

    # run_config = RunConfig(**args.__dict__, is_qat=False)
    # run_config.print_config()
    #
    # run_manager = RunManager(args.save_dir, stdm, run_config)
    # run_manager.validate(0)