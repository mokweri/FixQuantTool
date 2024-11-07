import argparse
import torch
import torchvision.models as models
from quantization.utils.model_transforms import create_quantizable_model
import json

from data_providers.imagenet import ImagenetDataProvider
from run_manager import ClassificationRunConfig, RunManager

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default=r"C:\Users\oma02\Downloads\ILSVRC\Data\CLS-LOC",
    help='Data set directory.')
parser.add_argument(
    '--workers',
    default=10,
    type=int,
    help='Number of data loading workers to be used.')
parser.add_argument('--epochs', default=3, type=int, help='Training epochs.')
parser.add_argument(
    '--weight_lr',
    default=1e-5,
    type=float,
    help='Initial learning rate of network weights.')
parser.add_argument(
    '--weight_lr_decay',
    default=0.94,
    type=int,
    help='Learning rate decay ratio of network weights.')
parser.add_argument(
    '--train_batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument(
    '--val_batch_size',
    default=100,
    type=int,
    help='Batch size for validation.')
parser.add_argument(
    '--weight_decay', default=1e-4, type=float, help='Weight decay.')
parser.add_argument(
    '--display_freq',
    default=100,
    type=int,
    help='Display training metrics every n steps.')
parser.add_argument(
    '--val_freq', default=1000, type=int, help='Validate model every n steps.')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'deploy'],
    help='Running mode.')
parser.add_argument(
    '--save_dir',
    default='./qat_models',
    help='Directory to save trained models.')
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
parser.add_argument(
    '--gpus',
    type=str,
    default='0',
    help='gpu ids to be used for training, seperated by commas')
parser.add_argument(
    "--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"]
)
args, _ = parser.parse_known_args()
ImagenetDataProvider.DEFAULT_PATH = args.data_dir

torch.backends.cudnn.enabled = True  # Enable cuDNN
torch.backends.cudnn.benchmark = True  # Use cuDNN's auto-tuner for the best performance

if __name__ == '__main__':

    run_config = ClassificationRunConfig(dataset=args.dataset, test_batch_size=args.val_batch_size, n_worker=args.workers,
                                         valid_size=1)

    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # print(model)

    # qconfig = create_qconfig(model, run_config.valid_loader, bitwidth=8)
    # qconfig = standardize_qconfig(qconfig)
    # # save the dict for ease of future use
    # with open('qconfig_vgg.json', 'w') as json_file:
    #     json.dump(qconfig, json_file)

    # load qconfig
    with open('qconfig_files/qconfig_vgg.json', 'r') as json_file:
        qconfig = json.load(json_file)
    # print
    # for key, value in qconfig.items():
    #     print(f"{key}: {value}")

    # print(f"qconfig.out = {qconfig['conv1']['out']}")
    qfused_model = create_quantizable_model(model, qconfig)
    print(qfused_model)

    run_manager = RunManager(".tmp/eval_subnet", qfused_model, run_config, init=False)
    loss, (top1, top5), output = run_manager.validate(net=qfused_model, is_test=True, extra_preprocess=None)

# test_image = torch.randn(1, 3, 224, 224)
#
# qfused_model.eval()
# model2.eval()
# output = qfused_model(test_image)
# output2 = model2(test_image)
#
#
# print(output)
# print('-------------------------------------------')
# print(output2)
# for name, param in q_model.named_parameters():
#     print(name)

# net_params = []
# for param in qfused_model.parameters():
#     if param.requires_grad:
#         net_params.append(param)
#         print(param)
#         break

# print(qfused_model.parameters().__next__().device)
# # Check if a GPU is available
# if torch.cuda.is_available():
#     print("GPU is available!")
#     print(f"Device name: {torch.cuda.get_device_name(0)}")
# else:
#     print("GPU is not available.")


# ------test
#     from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
#     import copy
#     import torch.fx as fx
#
#     model = copy.deepcopy(model)
#     fx_model: fx.GraphModule = fx.symbolic_trace(model)
#     modules = dict(fx_model.named_modules())
#
#     modules_patterns = {
#         (torch.nn.Conv1d, torch.nn.BatchNorm1d),
#         (torch.nn.Conv2d, torch.nn.BatchNorm2d),
#         (torch.nn.Conv3d, torch.nn.BatchNorm3d)
#     }
#
#     # Fuse conv with BN according to matching pattern
#     for pattern in modules_patterns:
#         for node in fx_model.graph.nodes:
#             if matches_module_pattern(pattern, node, modules):
#                 if len(node.args[0].users) > 1:  # conv that has multiple consumers is not fused
#                     conv = modules[node.args[0].target]
#                     continue
#                 # Fuse
#                 print('Fusing Conv2d with BatchNorm2d --> QuantizedConvBatchNorm2d')
#                 print(node.args[0])
# ------------------------------------