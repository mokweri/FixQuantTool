import argparse
import os
import torchvision.models as models
import torch
import platform
import yaml
import logging

from fixquant.models.cifar import *
from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor

from fixquant.data.imagenet import ImagenetDataProvider
from fixquant.data.cifar10 import Cifar10DataProvider
from fixquant.training import RunConfig, RunManager


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
                    default=os.environ.get("FIXQUANT_DATA_DIR", "/home/obed/Documents/imagenet-mini"),)

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

parser.add_argument("--model_type", type=str, default="emu",
                    choices=["emu", "tilecnn"],
                    help=(
                        "'emu'     → HLS sequential emulation (convert_to_emu_model) — fast to build.\n"
                        "'tilecnn' → TileCNN digital-twin with fused residual add and hardware-exact\n"
                        "            GAP / MaxPool kernels (convert_to_tilecnn_model) — bit-identical\n"
                        "            to the real FPGA hardware accuracy."
                    ))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("deploy_eval")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # GPU selection remains via args.gpus for RunManager compatibility
    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]

    """Set Dataset"""
    ImagenetDataProvider.DEFAULT_PATH = os.environ.get(
        "FIXQUANT_DATA_DIR", args.dataroot
    )

    """Imagenet models"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    """Cifar models"""
    # model = resnet18_cifar10()

    """ ---- QUANTIZATION PROCESSING -----"""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "configs/quant_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    Qatprocessor = QatProcessor(model, config)
    model = Qatprocessor.quantize()
    Qatprocessor.freeze()
    Qatprocessor.load_qat_weights(str(repo_root / 'qat_models/checkpoint/resnet50_best.pth.tar'))

    """ ---- INFERENCE PROCESSING -----"""
    infer_processor = InferProcessor(model, config)

    if args.model_type == "tilecnn":
        logger.info("Building TileCNN digital-twin model (bit-exact FPGA accuracy)...")
        eval_model = infer_processor.convert_to_tilecnn_model()
    else:
        logger.info("Building HLS sequential emulation model...")
        eval_model = infer_processor.convert_to_emu_model()

    qconfig = infer_processor.generate_qconfig()
    logger.info("Model type: %s | qconfig entries: %d", args.model_type, len(qconfig))

    # Evaluation-only script: uncomment below to run validation
    run_config = RunConfig(**args.__dict__, is_qat=False)
    run_config.print_config()
    run_manager = RunManager(args.save_dir, eval_model, run_config)
    run_manager.validate(0)