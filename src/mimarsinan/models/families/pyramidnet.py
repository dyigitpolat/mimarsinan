"""CIFAR additive PyramidNet with basic units (han2016deep Table 1 p3, eq. 2, Fig. 6(d) p5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.cifar_models import _parse_image_shape

def pyramidnet_units_per_group(depth: int) -> int:
    """N basic units per group; depth must be 6N + 2 with N >= 1."""
    depth = int(depth)
    if depth < 8 or (depth - 2) % 6 != 0:
        raise ValueError(f"PyramidNet (basic) depth must be 6N + 2 with N >= 1, got {depth}")
    return (depth - 2) // 6


def pyramidnet_widths(depth: int, alpha: int) -> list[int]:
    """Output width of every unit: floor(16 + alpha * j / U) for j = 1..U, U = 3N
    (han2016deep Table 1), in exact integer arithmetic so the last is 16 + alpha."""
    units = 3 * pyramidnet_units_per_group(depth)
    alpha = int(alpha)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    return [(16 * units + alpha * j) // units for j in range(1, units + 1)]


class PyramidBasicUnit(nn.Module):
    """BN-Conv3x3-BN-ReLU-Conv3x3-BN plus a (pooled) zero-padded identity shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("PyramidNet widths never decrease")
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut_pool: nn.Module | None = nn.AvgPool2d(2) if stride == 2 else None
        self.pad_channels = out_channels - in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.bn3(self.conv2(self.relu(self.bn2(self.conv1(self.bn1(x))))))
        shortcut = x if self.shortcut_pool is None else self.shortcut_pool(x)
        if self.pad_channels > 0:
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))
        return branch + shortcut


class CifarPyramidNet(nn.Module):
    """16-channel stem + BN, 3 groups of N basic units widening additively by alpha, final BN-ReLU, GAP, FC.

    Module names avoid the LLRD depth patterns, so layer-wise LR decay stays inert as for the other vehicles.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        depth: int = 110,
        alpha: int = 48,
    ):
        super().__init__()
        in_channels, _, _ = _parse_image_shape(type(self).__name__, input_shape, downsampling=4)
        n = pyramidnet_units_per_group(depth)
        widths = pyramidnet_widths(depth, alpha)
        self.depth, self.alpha, self.widths = int(depth), int(alpha), tuple(widths)

        self.stem_conv = nn.Conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        units: list[nn.Module] = []
        prev = 16
        for j, width in enumerate(widths):
            stride = 2 if j in (n, 2 * n) else 1
            units.append(PyramidBasicUnit(prev, width, stride))
            prev = width
        self.units = nn.Sequential(*units)
        self.final_bn = nn.BatchNorm2d(prev)
        self.final_relu = nn.ReLU()
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.units(self.stem_bn(self.stem_conv(x)))
        x = self.final_relu(self.final_bn(x))
        return self.classifier(self.flatten(self.head_pool(x)))
