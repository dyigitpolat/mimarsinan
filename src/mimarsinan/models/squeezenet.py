"""SqueezeNet: a Fire-module conv vehicle (squeeze 1x1 -> expand 1x1 + 3x3, concat)."""

from __future__ import annotations

import torch
import torch.nn as nn


_SPATIAL_FLOOR = 2


def _allowed_pool_count(spatial_size: int) -> int:
    """Max /2 MaxPools that keep the feature map at >= ``_SPATIAL_FLOOR``."""
    n = 0
    size = int(spatial_size)
    while size // 2 >= _SPATIAL_FLOOR:
        size //= 2
        n += 1
    return n


class FireModule(nn.Module):
    """SqueezeNet Fire module: squeeze 1x1 -> (expand 1x1, expand 3x3) -> concat.

    Output channels = ``expand1x1 + expand3x3``; spatial size is preserved.
    """

    def __init__(
        self,
        in_channels: int,
        squeeze: int,
        expand1x1: int,
        expand3x3: int,
    ):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze, expand1x1, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze, expand3x3, kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)
        self.out_channels = expand1x1 + expand3x3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_act(self.squeeze(x))
        e1 = self.expand1x1_act(self.expand1x1(x))
        e3 = self.expand3x3_act(self.expand3x3(x))
        return torch.cat([e1, e3], dim=1)


class SqueezeNet(nn.Module):
    """Scaled, input-adaptive SqueezeNet: stem conv -> Fire modules -> conv readout.

    Only mappable ops (Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, torch.cat); 3x3 expands use
    SAME padding and pools are capped so a 28x28/32x32 input never collapses below 1x1.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        width: int = 24,
    ):
        super().__init__()
        shape = tuple(input_shape)
        if len(shape) != 3:
            raise ValueError(f"SqueezeNet expects input_shape (C, H, W), got {shape}")
        width = int(width)
        if width < 1:
            raise ValueError(f"SqueezeNet width must be >= 1, got {width}.")

        in_channels, h_in, w_in = int(shape[0]), int(shape[1]), int(shape[2])
        self.width = width

        stem_channels = width * 2
        max_pools = _allowed_pool_count(min(h_in, w_in))

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        prev_channels = stem_channels
        pools_used = 0
        if max_pools > 0:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            pools_used += 1

        squeeze_plan = [width, width, width * 2, width * 2, width * 3, width * 3]
        for stage_index, squeeze in enumerate(squeeze_plan):
            expand = squeeze * 2
            fire = FireModule(
                prev_channels,
                squeeze=squeeze,
                expand1x1=expand,
                expand3x3=expand,
            )
            layers.append(fire)
            prev_channels = fire.out_channels
            is_stage_boundary = stage_index % 2 == 1
            if is_stage_boundary and pools_used < max_pools:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                pools_used += 1

        self.features = nn.Sequential(*layers)
        self.classifier_conv = nn.Conv2d(prev_channels, num_classes, kernel_size=1)
        self.classifier_act = nn.ReLU(inplace=True)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier_act(self.classifier_conv(x))
        x = self.head_pool(x)
        return self.flatten(x)
