"""CIFAR DenseNet-BC (huang2016densely p4) in the published and post-activation layer orders."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.cifar_models import _parse_image_shape

DENSENET_LAYER_ORDERS = ("published", "post_activation")


def densenet_bc_layers_per_block(depth: int) -> int:
    """(L - 4) / 6 bottleneck layers per block; L must be 6n + 4 with n >= 1."""
    depth = int(depth)
    if depth < 10 or (depth - 4) % 6 != 0:
        raise ValueError(f"DenseNet-BC depth must be 6n + 4 with n >= 1, got {depth}")
    return (depth - 4) // 6


class DenseBottleneckLayer(nn.Module):
    """One DenseNet-BC layer: m channels in, k new channels out."""

    def __init__(self, in_channels: int, growth_rate: int, layer_order: str):
        super().__init__()
        inner = 4 * growth_rate
        self.layer_order = layer_order
        self.conv1 = nn.Conv2d(in_channels, inner, 1, bias=False)
        self.conv2 = nn.Conv2d(inner, growth_rate, 3, padding=1, bias=False)
        if layer_order == "published":
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(inner)
        else:
            self.bn1 = nn.BatchNorm2d(inner)
            self.bn2 = nn.BatchNorm2d(growth_rate)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_order == "published":
            return self.conv2(self.relu2(self.bn2(self.conv1(self.relu1(self.bn1(x))))))
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))


class DenseBlock(nn.Module):
    """n layers; layer i reads the concatenation of the block input and layers < i."""

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int, layer_order: str):
        super().__init__()
        self.units = nn.ModuleList([
            DenseBottleneckLayer(in_channels + i * growth_rate, growth_rate, layer_order)
            for i in range(num_layers)
        ])
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for unit in self.units:
            features.append(unit(torch.cat(features, 1)))
        return torch.cat(features, 1)


class DenseTransition(nn.Module):
    """1x1 convolution to ``out_channels`` then 2x2 average pooling."""

    def __init__(self, in_channels: int, out_channels: int, layer_order: str):
        super().__init__()
        self.layer_order = layer_order
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels if layer_order == "published" else out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_order == "published":
            return self.pool(self.conv(self.relu(self.bn(x))))
        return self.pool(self.relu(self.bn(self.conv(x))))


class CifarDenseNetBC(nn.Module):
    """2k-channel 3x3 stem, three dense blocks of (L-4)/6 bottleneck layers, compression transitions, GAP, FC.

    ``published``: pre-activation BN-ReLU-Conv with a final BN-ReLU. ``post_activation``:
    Conv-BN-ReLU everywhere (every BatchNorm directly follows its conv; no final BN).
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        depth: int = 100,
        growth_rate: int = 12,
        compression: float = 0.5,
        layer_order: str = "published",
    ):
        super().__init__()
        in_channels, _, _ = _parse_image_shape(type(self).__name__, input_shape, downsampling=4)
        if layer_order not in DENSENET_LAYER_ORDERS:
            raise ValueError(f"layer_order must be one of {DENSENET_LAYER_ORDERS}, got {layer_order!r}")
        growth_rate = int(growth_rate)
        if growth_rate < 1:
            raise ValueError(f"growth_rate must be >= 1, got {growth_rate}")
        compression = float(compression)
        if not 0.0 < compression <= 1.0:
            raise ValueError(f"compression must be in (0, 1], got {compression}")
        n = densenet_bc_layers_per_block(depth)
        self.depth, self.growth_rate, self.compression = int(depth), growth_rate, compression
        self.layer_order = layer_order

        channels = 2 * growth_rate
        self.stem_conv = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        if layer_order == "post_activation":
            self.stem_bn = nn.BatchNorm2d(channels)
            self.stem_relu = nn.ReLU()

        self.dense1 = DenseBlock(channels, n, growth_rate, layer_order)
        out1 = int(compression * self.dense1.out_channels)
        self.trans1 = DenseTransition(self.dense1.out_channels, out1, layer_order)
        self.dense2 = DenseBlock(out1, n, growth_rate, layer_order)
        out2 = int(compression * self.dense2.out_channels)
        self.trans2 = DenseTransition(self.dense2.out_channels, out2, layer_order)
        self.dense3 = DenseBlock(out2, n, growth_rate, layer_order)
        final = self.dense3.out_channels

        if layer_order == "published":
            self.final_bn = nn.BatchNorm2d(final)
            self.final_relu = nn.ReLU()
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(final, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        if self.layer_order == "post_activation":
            x = self.stem_relu(self.stem_bn(x))
        x = self.dense3(self.trans2(self.dense2(self.trans1(self.dense1(x)))))
        if self.layer_order == "published":
            x = self.final_relu(self.final_bn(x))
        return self.classifier(self.flatten(self.head_pool(x)))
