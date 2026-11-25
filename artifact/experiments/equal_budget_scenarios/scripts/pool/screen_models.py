""

from __future__ import annotations

import torch
import torch.nn as nn

INPUT_SHAPE = (3, 32, 32)
R18_BASE = (64, 128, 256, 512)


def snap8(v: float) -> int:
    ""
    return max(8, int(v / 8 + 0.5) * 8)


def r18_widths(w: float) -> tuple[int, ...]:
    ""
    return tuple(snap8(c * w) for c in R18_BASE)


class BasicBlock(nn.Module):
    ""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
        return self.relu_out(y + (x if self.shortcut is None else self.shortcut(x)))


class CifarResNet18(nn.Module):
    ""

    def __init__(self, width: float, num_classes: int = 100):
        super().__init__()
        widths = r18_widths(width)
        self.widths = widths
        self.stem_conv = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(widths[0])
        self.stem_relu = nn.ReLU()
        units, cin = [], widths[0]
        for s, cout in enumerate(widths):
            for b in range(2):
                units.append(BasicBlock(cin, cout, 2 if (s > 0 and b == 0) else 1))
                cin = cout
        self.stages = nn.Sequential(*units)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(cin, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem_relu(self.stem_bn(self.stem_conv(x))))
        return self.classifier(self.flatten(self.head_pool(x)))


def build_member(spec: dict, num_classes: int = 100) -> nn.Module:
    ""
    builder, config = spec["builder"], dict(spec.get("config", {}))
    if builder == "resnet18_cifar":
        return CifarResNet18(float(config["width"]), num_classes)
    from mimarsinan.models.builders import BUILDERS_REGISTRY

    return BUILDERS_REGISTRY[builder]("cpu", INPUT_SHAPE, num_classes, {}).build(config)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
