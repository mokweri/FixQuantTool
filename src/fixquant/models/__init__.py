"""Model architectures (ResNet, VGG for ImageNet and CIFAR)."""

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def get_model(name: str, pretrained: bool = True):
    """Build an ImageNet model by name (torchvision-backed).

    Used by the CLI tools so that runs are reproducible from the command line
    instead of by editing the scripts.
    """
    import torchvision.models as tvm

    zoo = {
        "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
        "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT),
        "vgg16": (tvm.vgg16, tvm.VGG16_Weights.DEFAULT),
        "mobilenet_v2": (tvm.mobilenet_v2, tvm.MobileNet_V2_Weights.DEFAULT),
    }
    if name not in zoo:
        raise ValueError(f"Unknown model '{name}'. Choices: {sorted(zoo)}")
    ctor, weights = zoo[name]
    return ctor(weights=weights if pretrained else None)


MODEL_CHOICES = ["resnet18", "resnet50", "vgg16", "mobilenet_v2"]
