"""DeepCNN: configurable-depth plain Conv-BN-ReLU stack (the deep-conv depth-probe vehicle)."""

from __future__ import annotations

import torch
import torch.nn as nn


_MIN_DEPTH = 4
_MAX_DEPTH = 16
_CHANNEL_CAP = 128
_SPATIAL_FLOOR = 2  # a k3-pad1 conv block always needs at least 2x2 to operate on


def _get_activation(name: str) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(inplace=True)
    if name == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    if name == "GELU":
        return nn.GELU()
    return nn.ReLU(inplace=True)


def allowed_pool_count(spatial_size: int) -> int:
    """Max number of /2 MaxPools that keep the feature map at >= _SPATIAL_FLOOR.

    A 28x28 (or 32x32) input never collapses below 1x1 because we stop pooling
    while the size still has room for a k3-pad1 conv block.
    """
    n = 0
    size = int(spatial_size)
    while size // 2 >= _SPATIAL_FLOOR:
        size //= 2
        n += 1
    return n


def _pool_after_block(depth: int, n_pools: int) -> set[int]:
    """Block indices (0-based) after which to insert a MaxPool, spread evenly across depth."""
    if n_pools <= 0:
        return set()
    n_pools = min(n_pools, depth)
    # Evenly space the pools so they fall in the interior, never after the final block.
    step = depth / (n_pools + 1)
    return {min(depth - 1, int(round(step * (i + 1))) - 1) for i in range(n_pools)}


def _stage_channels(depth: int, width: int, n_pools: int) -> list[int]:
    """Per-block out_channels; doubles at each pool boundary, capped at _CHANNEL_CAP."""
    pool_blocks = _pool_after_block(depth, n_pools)
    channels: list[int] = []
    current = int(width)
    for block in range(depth):
        channels.append(min(current, _CHANNEL_CAP))
        if block in pool_blocks:
            current = min(current * 2, _CHANNEL_CAP)
    return channels


class DeepCNN(nn.Module):
    """[Conv(k3,pad1) -> BatchNorm -> ReLU] x depth (periodic MaxPool) -> AdaptiveAvgPool -> Linear.

    Plain sequential deep conv stack — no grouped/depthwise conv, no residual connections —
    so it maps fully on-chip (SAME padding keeps the multi-channel convs off the
    LayoutSourceView no-pad limitation). CNNs train far deeper than the plain deep_mlp,
    so this is the vehicle for the deep cascaded single-spike firing-gain probe.

    ``depth`` is the number of Conv-BN-ReLU blocks (4..16); ``width`` is the base channel
    count (channels double at each pool boundary, capped at 128). Pools are capped from the
    input spatial size so a 28x28 or 32x32 input never collapses below 1x1.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        depth: int,
        width: int = 16,
        base_activation: str = "ReLU",
    ):
        super().__init__()
        shape = tuple(input_shape)
        if len(shape) != 3:
            raise ValueError(f"DeepCNN expects input_shape (C, H, W), got {shape}")
        depth = int(depth)
        width = int(width)
        if depth < _MIN_DEPTH or depth > _MAX_DEPTH:
            raise ValueError(
                f"DeepCNN depth must be in [{_MIN_DEPTH}, {_MAX_DEPTH}], got {depth}."
            )
        if width < 1:
            raise ValueError(f"DeepCNN width must be >= 1, got {width}.")

        in_channels, h_in, w_in = int(shape[0]), int(shape[1]), int(shape[2])
        self.depth = depth
        self.width = width

        n_pools = min(allowed_pool_count(min(h_in, w_in)), depth)
        pool_blocks = _pool_after_block(depth, n_pools)
        channels = _stage_channels(depth, width, n_pools)

        layers: list[nn.Module] = []
        prev_channels = in_channels
        for block in range(depth):
            out_channels = channels[block]
            layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(_get_activation(base_activation))
            if block in pool_blocks:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(prev_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head_pool(x)
        x = self.flatten(x)
        return self.classifier(x)
