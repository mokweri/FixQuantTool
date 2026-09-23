"""Model architectures (ResNet, VGG for ImageNet and CIFAR)."""

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg_tilecnn import VGGTileCNN, vgg16_tilecnn, vgg16_bn_pool3


def get_model(name: str, pretrained: bool = True):
    """Build an ImageNet model by name (torchvision-backed).

    Used by the CLI tools so that runs are reproducible from the command line
    instead of by editing the scripts.

    ``vgg16_tilecnn`` is FixQuant's own VGG-16 variant: the ``vgg16_bn`` feature
    stack with TileCNN-legal 3x3/s2/p1 pooling and a convolutional head that
    fits the accelerator's weight buffer. ``pretrained=True`` transfers the
    torchvision convolution stack; its head is always randomly initialized and
    has to be trained (see docs/vgg16_tilecnn.md).
    """
    import torchvision.models as tvm

    zoo = {
        "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
        "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT),
        "vgg16": (tvm.vgg16, tvm.VGG16_Weights.DEFAULT),
        "vgg16_bn": (tvm.vgg16_bn, tvm.VGG16_BN_Weights.DEFAULT),
        "mobilenet_v2": (tvm.mobilenet_v2, tvm.MobileNet_V2_Weights.DEFAULT),
    }
    local_zoo = {
        "vgg16_tilecnn": vgg16_tilecnn,
        "vgg16_bn_pool3": vgg16_bn_pool3,
    }
    if name in local_zoo:
        return local_zoo[name](pretrained=pretrained)
    if name not in zoo:
        raise ValueError(
            f"Unknown model '{name}'. Choices: {sorted(set(zoo) | set(local_zoo))}")
    ctor, weights = zoo[name]
    return ctor(weights=weights if pretrained else None)


MODEL_CHOICES = [
    "resnet18",
    "resnet50",
    "vgg16",
    "vgg16_bn",
    "vgg16_bn_pool3",
    "vgg16_tilecnn",
    "mobilenet_v2",
]
