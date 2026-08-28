"""NarrowConvNet: a conv vehicle whose every on-chip layer is fan-in bounded."""

from __future__ import annotations

import torch.nn as nn


def _activation(name: str) -> nn.Module:
    if name == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    return nn.ReLU(inplace=True)


def _out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class NarrowConvNet(nn.Module):
    """A strided stem, constant-width 3x3 stride-2 body stages, a collapse conv
    that consumes the WHOLE remaining spatial extent, and a bare Linear readout.

    The vehicle's contract is a fan-in bound the ARCHITECTURE carries rather
    than the mapper: a body stage's fan-in is ``9 * body_channels + 1`` however
    deep the stack grows, the collapse conv's is ``h * w * body_channels + 1``
    for the residual extent it consumes, and the readout's is ``head_width + 1``
    — so widening the vehicle (``head_width``, more body stages) never widens a
    crossbar row. Only the stem is unbounded, which is why it is the layer a
    ``subsume`` placement hands to the host.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        stem_channels: int = 16,
        stem_kernel: int = 5,
        stem_stride: int = 2,
        body_blocks: int = 2,
        body_channels: int = 16,
        head_width: int = 128,
        base_activation: str = "ReLU",
    ):
        super().__init__()
        shape = tuple(int(v) for v in input_shape)
        if len(shape) != 3:
            raise ValueError(f"NarrowConvNet expects input_shape (C, H, W), got {shape}")
        in_channels, height, width = shape
        stem_kernel = int(stem_kernel)
        stem_stride = int(stem_stride)
        body_blocks = int(body_blocks)
        if stem_kernel < 1 or stem_stride < 1 or body_blocks < 0:
            raise ValueError(
                f"NarrowConvNet: stem_kernel={stem_kernel}, stem_stride="
                f"{stem_stride}, body_blocks={body_blocks} must be positive "
                f"(body_blocks may be zero)")

        stem_padding = stem_kernel // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, int(stem_channels), stem_kernel,
                      stride=stem_stride, padding=stem_padding),
            _activation(base_activation),
        )
        height = _out_size(height, stem_kernel, stem_stride, stem_padding)
        width = _out_size(width, stem_kernel, stem_stride, stem_padding)

        body: list[nn.Module] = []
        channels = int(stem_channels)
        for block in range(body_blocks):
            self._require_reducible(height, width, block, shape)
            body.append(nn.Conv2d(channels, int(body_channels), 3, stride=2, padding=1))
            body.append(_activation(base_activation))
            channels = int(body_channels)
            height = _out_size(height, 3, 2, 1)
            width = _out_size(width, 3, 2, 1)
        self.body = nn.Sequential(*body)

        self.collapse = nn.Sequential(
            nn.Conv2d(channels, int(head_width), (height, width),
                      stride=(height, width)),
            _activation(base_activation),
        )
        self.flatten = nn.Flatten()
        self.head = nn.Linear(int(head_width), int(num_classes))

    @staticmethod
    def _require_reducible(height: int, width: int, block: int, shape) -> None:
        """A stride-2 stage entered at 1x1 leaves it at 1x1 — depth that buys nothing."""
        if height < 2 and width < 2:
            raise ValueError(
                f"NarrowConvNet: body stage {block} would run on a "
                f"({height}, {width}) feature map, which a stride-2 stage "
                f"cannot reduce further; input {shape} is too small for this "
                f"stem stride and body depth")

    def forward(self, x):
        x = self.collapse(self.body(self.stem(x)))
        return self.head(self.flatten(x))
