import argparse
import torchvision.models as models
import torch
import torch.nn as nn
import platform
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import yaml

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
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)

        return tensor


    @torch.no_grad()
    def save_layer_params(layer: nn.Module, path: str | Path) -> None:
        """
        layer : module that *still* holds float weights / bias produced by a fake-quant pass.  It also exposes
                       layer.frac_weight   (# fractional bits for weights)
                       layer.frac_bias     (# fractional bits for bias)
        path  : destination file (e.g. "conv_int8.pth")

        The function converts W/B to signed INT8 according to the (Qm.n) format, then serialises:
            { weight_int8, bias_int8?, frac_w, frac_b }
        """

        # -------- 1.  fetch frac parameters --------------------------------
        def _to_int(x):
            return int(x.item()) if torch.is_tensor(x) else int(x)

        frac_w = _to_int(layer.frac_weight)
        frac_b = _to_int(layer.frac_bias) if hasattr(layer, "frac_bias") else 0
        frac_act = _to_int(layer.frac_act)

        # -------- 2.  convert float → int8  ----------------
        w_int8 = to_int_tensor( layer.weight.detach(), signed=True, n_bits=8, n_frac=frac_w).to(torch.int8).cpu().clone()

        if layer.bias is not None:
            b_int8 = to_int_tensor( layer.bias.detach(), signed=True, n_bits=8, n_frac=frac_b).to(torch.int8).cpu().clone()
        else:
            b_int8 = None

        # -------- 3.  build payload & save ---------------------------------
        payload: Dict[str, Any] = {
            "weight_int8": w_int8,
            "frac_w": frac_w,
            "frac_out": frac_act
        }
        if b_int8 is not None:
            payload.update({"bias_int8": b_int8, "frac_b": frac_b})

        torch.save(payload, path)
        print(f"[save_layer_params] wrote quantised params to {path}")


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
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    """Cifar models"""
    # model = resnet18_cifar10()

    with open("quantization/utils/quant_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    Qatprocessor = QatProcessor(model, config)
    model = Qatprocessor.quantize()
    Qatprocessor.load_qat_weights('qat_models/checkpoint/model_best.pth.tar')
    Qatprocessor.freeze()

    # @TODO - the model doesnt freeze during training

    infer_processor = InferProcessor(model, config)
    stdm = infer_processor.convert_to_std_model()
    infer_processor.export_onnx_with_layer_metadata("res.onnx")
    qconfig = infer_processor.generate_qconfig()
    print(qconfig)

    #print(stdm)

    # for name, _ in stdm.named_modules():
    #     print(name)
    layer_name = "conv1"  # pick any name visible in .named_modules()
    conv_q = dict(stdm.named_modules())[layer_name]
    print(conv_q)
    save_layer_params(conv_q, "hw_fxp/conv1.pth")




    # create_compact_model(model)
    # qconfig = create_qconfig(model)
    """ NOTE: dont use compact_model with make_inference_model"""
    # ---------------------------------------------
    # inf_model = convert_to_inference_model(model)
    # gconfig = generate_qconfig(inf_model)
    # std_qconfig = standardize_qconfig(gconfig)
    # print(gconfig)
    # print(std_qconfig)
    #
    # emu_model = create_emulation_model(inf_model)
    #
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