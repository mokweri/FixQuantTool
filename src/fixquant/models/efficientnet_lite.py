"""EfficientNet-Lite0 built from plain torch modules.

EfficientNet-Lite is the edge variant of EfficientNet: no squeeze-and-excitation
and ReLU6 in place of swish, which suits fixed-point accelerators. This is the
topology of timm's ``efficientnet_lite0``, which uses symmetric padding;
timm's TensorFlow port ``tf_efficientnet_lite0`` pads asymmetrically and is not
DeepTile-legal.

The model is defined here rather than taken from timm because timm fuses each
batch norm and its activation into one ``BatchNormAct2d`` module, a subclass of
``nn.BatchNorm2d``. FixQuant folds every ``(Conv2d, BatchNorm2d)`` pair into the
convolution, and would discard the ReLU6 hidden inside that module. Here every
batch norm is a plain ``nn.BatchNorm2d`` followed by its own ``nn.ReLU6``.

Attribute names follow timm's, so timm's pretrained state dict loads without
renaming; ``tools/convert_timm_efficientnet_lite.py`` converts and verifies it.
"""

from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["EfficientNetLite", "efficientnet_lite0", "EFFICIENTNET_LITE0_CHECKPOINT"]

# Converted timm weights (efficientnet_lite0.ra_in1k); not version-controlled.
EFFICIENTNET_LITE0_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "checkpoints" / "efficientnet_lite0_ra_in1k.pth")

# (expansion ratio, kernel, stride, output channels, repeats) per stage.
LITE0_STAGES = [
    (1, 3, 1, 16, 1),
    (6, 3, 2, 24, 2),
    (6, 5, 2, 40, 2),
    (6, 3, 2, 80, 3),
    (6, 5, 1, 112, 3),
    (6, 5, 2, 192, 4),
    (6, 3, 1, 320, 1),
]


class DepthwiseSeparableConv(nn.Module):
    """First-stage block: depthwise convolution, then a linear pointwise projection."""

    def __init__(self, in_chs, out_chs, kernel, stride):
        super().__init__()
        self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel, stride, kernel // 2,
                                 groups=in_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.act1 = nn.ReLU6(inplace=True)
        self.conv_pw = nn.Conv2d(in_chs, out_chs, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)
        self.has_skip = stride == 1 and in_chs == out_chs

    def forward(self, x):
        y = self.bn2(self.conv_pw(self.act1(self.bn1(self.conv_dw(x)))))
        return x + y if self.has_skip else y


class InvertedResidual(nn.Module):
    """Pointwise expansion, depthwise convolution, and a linear projection."""

    def __init__(self, in_chs, out_chs, kernel, stride, expansion):
        super().__init__()
        mid = in_chs * expansion
        self.conv_pw = nn.Conv2d(in_chs, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1 = nn.ReLU6(inplace=True)
        self.conv_dw = nn.Conv2d(mid, mid, kernel, stride, kernel // 2,
                                 groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.act2 = nn.ReLU6(inplace=True)
        self.conv_pwl = nn.Conv2d(mid, out_chs, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)
        self.has_skip = stride == 1 and in_chs == out_chs

    def forward(self, x):
        y = self.act1(self.bn1(self.conv_pw(x)))
        y = self.act2(self.bn2(self.conv_dw(y)))
        y = self.bn3(self.conv_pwl(y))
        return x + y if self.has_skip else y


class EfficientNetLite(nn.Module):
    def __init__(self, stages=LITE0_STAGES, stem_chs=32, head_chs=1280,
                 num_classes=1000, drop_rate=0.2):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, stem_chs, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU6(inplace=True)
        blocks, chs = [], stem_chs
        for expansion, kernel, stride, out_chs, repeats in stages:
            stage = []
            for index in range(repeats):
                s = stride if index == 0 else 1
                if expansion == 1:
                    stage.append(DepthwiseSeparableConv(chs, out_chs, kernel, s))
                else:
                    stage.append(InvertedResidual(chs, out_chs, kernel, s, expansion))
                chs = out_chs
            blocks.append(nn.Sequential(*stage))
        self.blocks = nn.Sequential(*blocks)
        self.conv_head = nn.Conv2d(chs, head_chs, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(head_chs)
        self.act2 = nn.ReLU6(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(head_chs, num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act2(self.bn2(self.conv_head(x)))
        x = torch.flatten(self.global_pool(x), 1)
        return self.classifier(self.dropout(x))


def efficientnet_lite0(pretrained=True, checkpoint=None):
    """Build EfficientNet-Lite0, optionally with converted timm weights."""
    model = EfficientNetLite()
    if pretrained:
        path = Path(checkpoint or EFFICIENTNET_LITE0_CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Create it once with "
                "tools/convert_timm_efficientnet_lite.py, which needs timm.")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("state_dict", state), strict=True)
        model.pretrained_source = f"timm efficientnet_lite0.ra_in1k, converted ({path.name})"
    return model
