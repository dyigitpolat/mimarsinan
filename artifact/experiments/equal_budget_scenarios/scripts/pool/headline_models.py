""

from __future__ import annotations

import torch
import torch.nn as nn

from screen_models import BasicBlock, r18_widths

DEPTHS = {"R10": (1, 1, 1, 1), "R18": (2, 2, 2, 2), "R34": (3, 4, 6, 3)}


class CifarResNet(nn.Module):
    ""

    def __init__(self, blocks: tuple[int, int, int, int], width: float, num_classes: int = 100):
        super().__init__()
        widths = r18_widths(width)
        self.widths = widths
        self.blocks = tuple(int(b) for b in blocks)
        self.stem_conv = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(widths[0])
        self.stem_relu = nn.ReLU()
        units, cin = [], widths[0]
        for s, cout in enumerate(widths):
            for b in range(self.blocks[s]):
                units.append(BasicBlock(cin, cout, 2 if (s > 0 and b == 0) else 1))
                cin = cout
        self.stages = nn.Sequential(*units)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(cin, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem_relu(self.stem_bn(self.stem_conv(x))))
        return self.classifier(self.flatten(self.head_pool(x)))


def build_arch(arch: dict, num_classes: int = 100) -> nn.Module:
    ""
    return CifarResNet(DEPTHS[arch["depth"]], float(arch["width"]), num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def crossbar_weight_params(model: nn.Module) -> int:
    ""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
