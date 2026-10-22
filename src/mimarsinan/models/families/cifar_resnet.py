"""CIFAR ResNet family over declared options (E294).

One network family instead of one builder per network: the 3x3 stem (stem width),
three stages of n residual blocks at widths base x k x (1, 2, 4) (width multiplier
k; a bottleneck block expands its last 1x1 conv by 4), the block kind (basic two
3x3 convs, he2016deep Fig. 2; bottleneck 1x1-3x3-1x1, he2016deep Fig. 5), the block
order (post-activation conv-BN-ReLU with a ReLU after the sum, he2016deep;
pre-activation BN-ReLU-conv with a final BN-ReLU, he2016identity Fig. 4(e)) and the
shortcut where the shape changes (projection: a 1x1 conv, with BN in the
post-activation order and on the pre-activated input in the pre-activation order;
zero-padded identity: stride subsampling and zero channels, he2016deep option A).
The defaults are ``cifar_resnet20``'s, and a post-activation basic projection
block is ``CifarBasicBlock`` module for module, so every ``CifarResNet``
configuration is the same network under the same seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.cifar_models import _parse_image_shape

BLOCK_KINDS = ("basic", "bottleneck")
BLOCK_ORDERS = ("post_activation", "pre_activation")
SHORTCUTS = ("projection", "zero_pad")
MAX_BLOCKS_PER_STAGE = 111  # ResNet-1001: 9n + 2 with n = 111 (he2016identity)
BOTTLENECK_EXPANSION = 4
STAGE_STRIDES = (1, 2, 2)


class ResNetFamilyOptionError(ToolchainRefusal, ValueError):
    """A family member declared outside the family's options (block kind, order, shortcut, counts, widths)."""


def resnet_family_depth(blocks_per_stage: int, block_kind: str) -> int:
    """Weight layers (stem, block convs, classifier; shortcuts not counted): 6n + 2 or 9n + 2."""
    convs_per_block = 2 if block_kind == "basic" else 3
    return 3 * convs_per_block * int(blocks_per_stage) + 2


def _choice(name: str, value: str, legal: tuple[str, ...]) -> str:
    if value not in legal:
        raise ResNetFamilyOptionError(f"{name} must be one of {legal}, got {value!r}")
    return value


def _positive(name: str, value, upper: int | None = None) -> int:
    value = int(value)
    if value < 1 or (upper is not None and value > upper):
        bound = f" and at most {upper}" if upper is not None else ""
        raise ResNetFamilyOptionError(f"{name} must be at least 1{bound}, got {value}")
    return value


class ZeroPadShortcut(nn.Module):
    """Parameter-free shortcut: x[:, :, ::stride, ::stride] with zero channels appended."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if out_channels < in_channels:
            raise ResNetFamilyOptionError(
                f"a zero-padded shortcut cannot narrow {in_channels} channels to {out_channels}"
            )
        # A 1x1 window at the stride selects every stride-th position exactly.
        self.subsample: nn.Module | None = nn.AvgPool2d(1, stride=stride) if stride != 1 else None
        self.pad_channels = out_channels - in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subsample is not None:
            x = self.subsample(x)
        if self.pad_channels > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.pad_channels))
        return x


class CifarResidualBlock(nn.Module):
    """One residual block of the family; layers ``conv{i}``, ``bn{i}``, ``relu{i}`` in execution order."""

    def __init__(
        self, in_planes: int, planes: int, stride: int, *,
        block_kind: str, block_order: str, shortcut: str,
    ):
        super().__init__()
        self.out_planes = planes * (BOTTLENECK_EXPANSION if block_kind == "bottleneck" else 1)
        if block_kind == "basic":
            convs = [(in_planes, planes, 3, stride), (planes, planes, 3, 1)]
        else:
            convs = [(in_planes, planes, 1, 1), (planes, planes, 3, stride), (planes, self.out_planes, 1, 1)]
        self.pre_activation = block_order == "pre_activation"
        self.depth = len(convs)
        for i, (cin, cout, kernel, conv_stride) in enumerate(convs, start=1):
            conv = nn.Conv2d(cin, cout, kernel, stride=conv_stride, padding=kernel // 2, bias=False)
            if self.pre_activation:
                setattr(self, f"bn{i}", nn.BatchNorm2d(cin))
                setattr(self, f"relu{i}", nn.ReLU())
                setattr(self, f"conv{i}", conv)
            else:
                setattr(self, f"conv{i}", conv)
                setattr(self, f"bn{i}", nn.BatchNorm2d(cout))
                if i < self.depth:
                    setattr(self, f"relu{i}", nn.ReLU())
        self.shortcut: nn.Module | None = None
        if stride != 1 or in_planes != self.out_planes:
            if shortcut == "zero_pad":
                self.shortcut = ZeroPadShortcut(in_planes, self.out_planes, stride)
            elif self.pre_activation:
                self.shortcut = nn.Conv2d(in_planes, self.out_planes, 1, stride=stride, bias=False)
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.out_planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.out_planes),
                )
        # The pre-activation projection reads the pre-activated input; every other shortcut reads x.
        self.shortcut_reads_preactivation = self.pre_activation and isinstance(self.shortcut, nn.Conv2d)
        if not self.pre_activation:
            self.relu_out = nn.ReLU()

    def _layer(self, kind: str, i: int) -> nn.Module:
        return getattr(self, f"{kind}{i}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pre_activation:
            out = x
            for i in range(1, self.depth + 1):
                out = self._layer("bn", i)(self._layer("conv", i)(out))
                if i < self.depth:
                    out = self._layer("relu", i)(out)
            identity = x if self.shortcut is None else self.shortcut(x)
            return self.relu_out(out + identity)
        pre = self._layer("relu", 1)(self._layer("bn", 1)(x))
        if self.shortcut is None:
            skip = x
        else:
            skip = self.shortcut(pre if self.shortcut_reads_preactivation else x)
        out = self._layer("conv", 1)(pre)
        for i in range(2, self.depth + 1):
            out = self._layer("conv", i)(self._layer("relu", i)(self._layer("bn", i)(out)))
        return out + skip


class CifarResNetFamily(nn.Module):
    """3x3 stem -> three stages of residual blocks -> (final BN-ReLU) -> GAP -> Linear.

    Module names follow ``CifarResNet`` (``stem_*``, ``stages``, ``head_pool``,
    ``classifier``), so the defaults load ``CifarResNet`` state dicts and layer-wise
    LR decay stays inert as for the other CIFAR vehicles.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        blocks_per_stage: int = 3,
        base_width: int = 16,
        *,
        stem_width: int | None = None,
        width_multiplier: int = 1,
        block_kind: str = "basic",
        block_order: str = "post_activation",
        shortcut: str = "projection",
    ):
        super().__init__()
        in_channels, _, _ = _parse_image_shape(type(self).__name__, input_shape, downsampling=4)
        self.blocks_per_stage = _positive("blocks_per_stage", blocks_per_stage, MAX_BLOCKS_PER_STAGE)
        self.base_width = _positive("base_width", base_width)
        self.width_multiplier = _positive("width_multiplier", width_multiplier)
        self.stem_width = self.base_width if stem_width is None else _positive("stem_width", stem_width)
        self.block_kind = _choice("block_kind", block_kind, BLOCK_KINDS)
        self.block_order = _choice("block_order", block_order, BLOCK_ORDERS)
        self.shortcut = _choice("shortcut", shortcut, SHORTCUTS)
        pre_activation = self.block_order == "pre_activation"

        self.stem_conv = nn.Conv2d(in_channels, self.stem_width, 3, padding=1, bias=False)
        self.stem_bn: nn.Module | None = None if pre_activation else nn.BatchNorm2d(self.stem_width)
        self.stem_relu: nn.Module | None = None if pre_activation else nn.ReLU()

        stages: list[nn.Module] = []
        in_planes = self.stem_width
        for stage_index, stride in enumerate(STAGE_STRIDES):
            planes = self.base_width * self.width_multiplier * (2 ** stage_index)
            blocks: list[nn.Module] = []
            for block_index in range(self.blocks_per_stage):
                block = CifarResidualBlock(
                    in_planes, planes, stride if block_index == 0 else 1,
                    block_kind=self.block_kind, block_order=self.block_order, shortcut=self.shortcut,
                )
                blocks.append(block)
                in_planes = block.out_planes
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.final_bn: nn.Module | None = nn.BatchNorm2d(in_planes) if pre_activation else None
        self.final_relu: nn.Module | None = nn.ReLU() if pre_activation else None
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        if self.stem_bn is not None and self.stem_relu is not None:
            x = self.stem_relu(self.stem_bn(x))
        x = self.stages(x)
        if self.final_bn is not None and self.final_relu is not None:
            x = self.final_relu(self.final_bn(x))
        return self.classifier(self.flatten(self.head_pool(x)))
