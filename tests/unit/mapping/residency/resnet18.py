"""CIFAR ResNet-18 built in the tests (3x3 stem, no max-pool, BasicBlock [2, 2, 2, 2], 1x1 projection shortcuts).

Widths follow the RA channel rule (``x9_0_v3.channels``: the nearest multiple of 8 of 64 w, 128 w, 256 w, 512 w,
ties up, at least 8), so a member here is the RA member of the same width.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10


def channels(base: int, width: float) -> int:
    return max(8, int(base * width / 8 + 0.5) * 8)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
        return self.relu2(out + self.shortcut(x))


class CifarResNet18(nn.Module):
    def __init__(self, width: float = 1.0, num_classes: int = NUM_CLASSES):
        super().__init__()
        w = [channels(c, width) for c in (64, 128, 256, 512)]
        self.stem = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(w[0])
        self.relu = nn.ReLU()
        blocks, cin = [], w[0]
        for stage, cout in enumerate(w):
            for b in range(2):
                blocks.append(BasicBlock(cin, cout, 2 if stage > 0 and b == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.blocks(self.relu(self.bn(self.stem(x))))
        return self.fc(torch.flatten(self.pool(x), 1))


def conv_layers(model: CifarResNet18) -> List[Tuple[str, nn.Module, Tuple[int, int]]]:
    """(name, module, output H x W) of every conv and the head, in execution order (c1, c2, shortcut per block)."""
    out, hw = [("stem", model.stem, (32, 32))], 32
    for i, block in enumerate(model.blocks):
        hw = hw // block.conv1.stride[0]
        out.append((f"blocks.{i}.conv1", block.conv1, (hw, hw)))
        out.append((f"blocks.{i}.conv2", block.conv2, (hw, hw)))
        if len(block.shortcut):
            out.append((f"blocks.{i}.shortcut.0", block.shortcut[0], (hw, hw)))
    out.append(("fc", model.fc, (1, 1)))
    return out


def mvm_mapper_repr(width: float, seed: int = 0, dtype=torch.float32):
    """The MVM (value-domain) conversion of the member at random weights, perceptron indices assigned."""
    torch.manual_seed(seed)
    model = CifarResNet18(width).eval().to(dtype)
    flow = convert_torch_model(model, INPUT_SHAPE, num_classes=NUM_CLASSES, packaging=MVM_PACKAGING)
    rep = flow.get_mapper_repr()
    rep.assign_perceptron_indices()
    return model, flow, rep
