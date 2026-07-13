"""Small test networks exercising every op the TileCNN pipeline supports."""

import torch
import torch.nn as nn


class TinyMobileBlockNet(nn.Module):
    """Stem conv + one inverted-residual-style block (expand 1x1 -> depthwise
    3x3 -> project 1x1, ReLU6 activations, residual add) + GAP + FC."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.stem_act = nn.ReLU(inplace=True)

        self.expand = nn.Conv2d(8, 16, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(16)
        self.expand_act = nn.ReLU6(inplace=True)

        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.dw_bn = nn.BatchNorm2d(16)
        self.dw_act = nn.ReLU6(inplace=True)

        self.project = nn.Conv2d(16, 8, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(8)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem(x)))
        identity = x
        y = self.expand_act(self.expand_bn(self.expand(x)))
        y = self.dw_act(self.dw_bn(self.dw(y)))
        y = self.project_bn(self.project(y))
        x = identity + y
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def synthetic_loader(n_batches=4, batch_size=4, size=32, num_classes=10):
    """List of (images, labels) batches usable as a calibration loader."""
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(batch_size, 3, size, size, generator=g),
             torch.randint(0, num_classes, (batch_size,), generator=g))
            for _ in range(n_batches)]
