"""VGG-16 variant that satisfies the DeepTile fabric constraints.

Stock torchvision VGG-16 cannot be compiled for the DeepTile accelerator for two
independent reasons:

1. Pooling geometry. The accelerator implements exactly one pooling post-op --
   3x3, stride 2, padding 1, max. VGG's five ``MaxPool2d(2, 2)`` layers are
   rejected outright.
2. Classifier weight footprint. The weight buffer requires
   ``ceil(Cin / 16) * K^2 <= 512`` for every convolution and linear layer.
   ``classifier.0`` (25088 -> 4096) needs 1568, three times the budget.

This module builds ``vgg16_tilecnn``: the ``vgg16_bn`` feature stack with the
legal pooling geometry, and a convolutional head that replaces
``classifier.0``. Batch norm is folded into the convolutions at export
(``FusedConvBN``), so the exported INT8 graph is structurally identical to a
BN-free VGG while remaining easy to fine-tune.

All five pooling stages are shape-identical to the stock model because every
input is even: 224 -> 112 -> 56 -> 28 -> 14 -> 7.
"""

import torch.nn as nn

__all__ = [
    "VGGTileCNN",
    "vgg16_tilecnn",
    "vgg16_bn_pool3",
    "DEEPTILE_MAXPOOL",
    "TILECNN_MAXPOOL",
    "WEIGHT_BUFFER_BUDGET",
]


# The single pooling post-op the fabric implements.
DEEPTILE_MAXPOOL = dict(kernel_size=3, stride=2, padding=1)
TILECNN_MAXPOOL = DEEPTILE_MAXPOOL  # former name

# ceil(Cin / 16) * Kh * Kw must not exceed this for any conv or linear layer.
WEIGHT_BUFFER_BUDGET = 512

# VGG-16 feature configuration (torchvision cfg "D").
VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512, "M"]


def make_features(cfg=VGG16_CFG, batch_norm=True, in_channels=3):
    """Build the VGG feature stack with DeepTile-legal pooling.

    Layer indices match torchvision's ``vgg16_bn.features`` exactly, so the
    pretrained convolution and batch-norm weights load without remapping.
    """
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(**DEEPTILE_MAXPOOL)]
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGTileCNN(nn.Module):
    """VGG-16 with DeepTile-legal pooling and a convolutional classifier head.

    The head replaces VGG's 25088 -> 4096 fully-connected layer with a strided
    3x3 convolution followed by global average pooling:

        fc1   Conv2d(512, 4096, k=3, s=2, p=0)   512x7x7 -> 4096x3x3
        relu
        gap   AdaptiveAvgPool2d(1)               -> 4096x1x1
        flatten / dropout
        fc2   Linear(4096, 4096) + relu
        dropout
        fc3   Linear(4096, num_classes)

    ``fc1`` costs ``ceil(512 / 16) * 3 * 3 = 288`` weight-buffer entries, inside
    the 512 budget. Nothing transfers from the original classifier -- only the
    14.7M-parameter convolution stack carries over from ``vgg16_bn``.

    ``gap`` follows a convolution directly (no pooling in between) and emits a
    ``[C, 1, 1]`` tensor, so both linear layers see the ``[C, 1, 1]`` input the
    fabric requires.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5,
                 batch_norm: bool = True):
        super().__init__()
        self.features = make_features(batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=3, stride=2, padding=0),  # fc1
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                   # gap
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),                                     # fc2
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),                              # fc3
        )
        self._initialize_weights()
        self._backbone_frozen = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    """ backbone freezing (head warm-up) """

    def freeze_backbone(self, frozen: bool = True):
        """Freeze or unfreeze the pretrained feature stack.

        Gradients from the randomly initialized head would otherwise damage the
        pretrained features in the first few hundred steps. Freezing also holds
        the batch-norm running statistics still (see ``train``), which matters
        because the pooling change has already perturbed them.
        """
        for param in self.features.parameters():
            param.requires_grad = not frozen
        self._backbone_frozen = frozen
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_backbone_frozen", False):
            # Keep BN in inference mode so the frozen backbone's running
            # statistics are not updated while only the head is learning.
            self.features.eval()
        return self


def _load_pretrained_features(model):
    """Copy torchvision ``vgg16_bn`` convolution and BN weights into ``model``.

    Only ``features`` transfers: the head is a new architecture. Layer indices
    are identical, so this is a straight state-dict load.
    """
    import torchvision.models as tvm

    source = tvm.vgg16_bn(weights=tvm.VGG16_BN_Weights.IMAGENET1K_V1)
    model.features.load_state_dict(source.features.state_dict())
    return model


def vgg16_tilecnn(pretrained: bool = True, num_classes: int = 1000, **kwargs):
    """VGG-16 variant runnable on DeepTile, optionally with pretrained features."""
    model = VGGTileCNN(num_classes=num_classes, **kwargs)
    if pretrained:
        if num_classes != 1000:
            raise ValueError(
                "Pretrained features are only defined for the ImageNet stack; "
                "pass pretrained=False for other class counts.")
        _load_pretrained_features(model)
    return model


def vgg16_bn_pool3(pretrained: bool = True, **kwargs):
    """Stock torchvision ``vgg16_bn`` with only the pooling geometry swapped.

    Used for the free pooling ablation: pooling is parameter-free, so this
    isolates the cost of the pooling change from the cost of the new head. The
    classifier is left untouched and is *not* DeepTile-legal -- this model is an
    evaluation probe, never an export target.
    """
    import torchvision.models as tvm

    weights = tvm.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.vgg16_bn(weights=weights, **kwargs)
    for name, module in list(model.features.named_children()):
        if isinstance(module, nn.MaxPool2d):
            setattr(model.features, name, nn.MaxPool2d(**DEEPTILE_MAXPOOL))
    return model
