"""The 32px BC-2 checkpoint vehicles: CIFAR-style ResNet family and VGG-8.

Both stay inside the declared torch_mapping conversion subset: ungrouped
3x3/1x1 Conv2d (+BN absorption), ReLU, MaxPool/AdaptiveAvgPool and the
residual ``+`` as host ComputeOps (multi-input joins are opaque to liveness
transfers by design), Linear heads.

`CifarResNet` (He et al. 2016 sec. 4.2): 3 stages x n BasicBlocks with
16/32/64 base channels; shape changes use projection shortcuts (1x1 conv +
BN, "option B") because the zero-pad "option A" shortcut has no conversion
rule. Depth = 6n+2; `cifar_resnet20()` is n=3.

`CifarVGG8`: 6 conv+BN+ReLU in 3 maxpool stages ([64,64]-[128,128]-[256,256]
doubling trunk, ~2.6M params) + avgpool(2) + 2 FC. The head pools to 2x2
before flattening so EVERY neuron's fan-in stays <= 2304 (largest: the
256-channel 3x3 conv patch = 2304, then FC1 = 1024); with the softcore bias
row (+1 axon) the worst core is 2305 axons, so the BC-2 config declares
2560-axon cores — no input splitting needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _parse_image_shape(owner: str, input_shape, downsampling: int) -> tuple[int, int, int]:
    """(C, H, W) with H, W divisible by the net's total downsampling and roomy enough."""
    shape = tuple(input_shape)
    if len(shape) != 3:
        raise ValueError(f"{owner} expects input_shape (C, H, W), got {shape}")
    c, h, w = (int(d) for d in shape)
    if h % downsampling != 0 or w % downsampling != 0 or h < 2 * downsampling or w < 2 * downsampling:
        raise ValueError(
            f"{owner} needs H, W divisible by {downsampling} and >= {2 * downsampling}; got {h}x{w}"
        )
    return c, h, w


# ── ResNet family ────────────────────────────────────────────────────────────

_STAGE_STRIDES = (1, 2, 2)


class CifarBasicBlock(nn.Module):
    """conv3x3-BN-ReLU-conv3x3-BN + (projection) shortcut, ReLU after the add."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.relu_out(out + identity)


class CifarResNet(nn.Module):
    """Standard CIFAR ResNet: 3x3 stem -> 3 stages (w, 2w, 4w) -> avgpool -> Linear.

    Depth = 6 * blocks_per_stage + 2 (blocks_per_stage=3 -> ResNet-20).
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        blocks_per_stage: int = 3,
        base_width: int = 16,
    ):
        super().__init__()
        in_channels, _, _ = _parse_image_shape(
            type(self).__name__, input_shape, downsampling=4  # two stride-2 stages
        )
        blocks_per_stage = int(blocks_per_stage)
        base_width = int(base_width)
        if blocks_per_stage < 1:
            raise ValueError(f"blocks_per_stage must be >= 1, got {blocks_per_stage}")
        if base_width < 1:
            raise ValueError(f"base_width must be >= 1, got {base_width}")
        self.blocks_per_stage = blocks_per_stage
        self.base_width = base_width

        self.stem_conv = nn.Conv2d(in_channels, base_width, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(base_width)
        self.stem_relu = nn.ReLU()

        stages: list[nn.Module] = []
        in_planes = base_width
        for stage_index, stride in enumerate(_STAGE_STRIDES):
            planes = base_width * (2 ** stage_index)
            blocks: list[nn.Module] = []
            for block_index in range(blocks_per_stage):
                blocks.append(CifarBasicBlock(
                    in_planes, planes, stride=stride if block_index == 0 else 1,
                ))
                in_planes = planes
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.stages(x)
        x = self.flatten(self.head_pool(x))
        return self.classifier(x)


def cifar_resnet20(
    num_classes: int = 10, input_shape: tuple[int, ...] = (3, 32, 32)
) -> CifarResNet:
    """ResNet-20 (16/32/64, 9 BasicBlocks, ~0.27M params): the CIFAR baseline rung."""
    return CifarResNet(
        input_shape=input_shape, num_classes=num_classes,
        blocks_per_stage=3, base_width=16,
    )


# ── VGG-8 ────────────────────────────────────────────────────────────────────

_POOL_COUNT = 3
_STAGE_MULTIPLIERS = (1, 2, 4)  # channel doubling per pool stage
_CONVS_PER_STAGE = 2
_HEAD_POOL_SIZE = 2


class CifarVGG8(nn.Module):
    """[Conv-BN-ReLU x2, MaxPool] x3 -> AdaptiveAvgPool(2) -> FC-ReLU -> FC."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        base_channels: int = 64,
        fc_width: int = 512,
    ):
        super().__init__()
        in_channels, _, _ = _parse_image_shape(
            type(self).__name__, input_shape, downsampling=2 ** _POOL_COUNT
        )
        base_channels = int(base_channels)
        fc_width = int(fc_width)
        if base_channels < 1:
            raise ValueError(f"base_channels must be >= 1, got {base_channels}")
        if fc_width < 1:
            raise ValueError(f"fc_width must be >= 1, got {fc_width}")
        self.base_channels = base_channels
        self.fc_width = fc_width

        layers: list[nn.Module] = []
        prev = in_channels
        for multiplier in _STAGE_MULTIPLIERS:
            channels = base_channels * multiplier
            for _ in range(_CONVS_PER_STAGE):
                layers.append(nn.Conv2d(prev, channels, 3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(channels))
                layers.append(nn.ReLU())
                prev = channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)

        self.head_pool = nn.AdaptiveAvgPool2d(_HEAD_POOL_SIZE)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(prev * _HEAD_POOL_SIZE ** 2, fc_width)
        self.fc_relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(self.head_pool(x))
        return self.fc2(self.fc_relu(self.fc1(x)))


def cifar_vgg8(
    num_classes: int = 10, input_shape: tuple[int, ...] = (3, 32, 32)
) -> CifarVGG8:
    """VGG-8 at the standard CIFAR scale (base 64 channels, 512-wide FC)."""
    return CifarVGG8(
        input_shape=input_shape, num_classes=num_classes,
        base_channels=64, fc_width=512,
    )
