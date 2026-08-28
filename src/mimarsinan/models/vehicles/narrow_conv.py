"""NarrowConvNet: a conv vehicle whose every on-chip layer is fan-in bounded."""

from __future__ import annotations

import torch.nn as nn

#: The readout that scores classes ON CHIP. A bare final layer cannot be a
#: spiking core (nothing thresholds it), so it rides the host as a readout
#: suffix and the deployed segment's last core carries features, not scores.
ACTIVATED_READOUT = "activated"
BARE_READOUT = "bare"


def _activation(name: str) -> nn.Module:
    if name == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    return nn.ReLU(inplace=True)


def _out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class NarrowConvNet(nn.Module):
    """A strided Conv-BN stem, constant-width 3x3 stride-2 Conv-BN body stages,
    a collapse conv that consumes the WHOLE residual spatial extent, pointwise
    Conv-BN trunk stages on the resulting 1x1 map, and a pointwise readout.

    FAN-IN IS ARCHITECTURAL. A body stage's fan-in is ``9 * body_channels + 1``
    however deep the stack grows, the collapse's is ``h * w * body_channels + 1``
    for the residual extent it consumes, and every trunk stage and the readout
    cost ``trunk_width + 1`` — so widening (``trunk_width``) or deepening
    (``trunk_blocks``) the vehicle never widens a crossbar row. Only the stem is
    unbounded, which is why it is the layer a ``subsume`` placement hands to the
    host.

    EVERY STAGE IS A CONV, AND NOTHING RESHAPES BETWEEN THEM, because an
    event-serial soma law folds each core's charge in the mapper's own slot
    order and the training twin reproduces that order only where a hop's
    upstream is another hop (an intervening ``Flatten`` carries no event train)
    and its effective weight spans its whole input (a partial receptive field is
    an unfold the twin does not reproduce). The collapse and pointwise stages
    satisfy the second condition by construction; the single trailing flatten
    sits after the last hop, where nothing downstream needs events.
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
        trunk_width: int = 128,
        trunk_blocks: int = 1,
        readout: str = BARE_READOUT,
        base_activation: str = "ReLU",
    ):
        super().__init__()
        shape = tuple(int(v) for v in input_shape)
        if len(shape) != 3:
            raise ValueError(f"NarrowConvNet expects input_shape (C, H, W), got {shape}")
        if readout not in (BARE_READOUT, ACTIVATED_READOUT):
            raise ValueError(
                f"NarrowConvNet: readout={readout!r} must be {BARE_READOUT!r} "
                f"or {ACTIVATED_READOUT!r}")
        in_channels, height, width = shape
        stem_kernel = int(stem_kernel)
        stem_stride = int(stem_stride)
        body_blocks = int(body_blocks)
        trunk_blocks = int(trunk_blocks)
        if stem_kernel < 1 or stem_stride < 1 or body_blocks < 0 or trunk_blocks < 1:
            raise ValueError(
                f"NarrowConvNet: stem_kernel={stem_kernel}, stem_stride="
                f"{stem_stride}, body_blocks={body_blocks}, trunk_blocks="
                f"{trunk_blocks} are out of range (body_blocks may be zero, "
                f"the trunk carries at least its collapse stage)")

        stem_padding = stem_kernel // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, int(stem_channels), stem_kernel,
                      stride=stem_stride, padding=stem_padding),
            nn.BatchNorm2d(int(stem_channels)),
            _activation(base_activation),
        )
        height = _out_size(height, stem_kernel, stem_stride, stem_padding)
        width = _out_size(width, stem_kernel, stem_stride, stem_padding)

        body: list[nn.Module] = []
        channels = int(stem_channels)
        for block in range(body_blocks):
            self._require_reducible(height, width, block, shape)
            body.append(nn.Conv2d(channels, int(body_channels), 3, stride=2, padding=1))
            body.append(nn.BatchNorm2d(int(body_channels)))
            body.append(_activation(base_activation))
            channels = int(body_channels)
            height = _out_size(height, 3, 2, 1)
            width = _out_size(width, 3, 2, 1)
        self.body = nn.Sequential(*body)

        trunk: list[nn.Module] = [
            nn.Conv2d(channels, int(trunk_width), (height, width),
                      stride=(height, width)),
            nn.BatchNorm2d(int(trunk_width)),
            _activation(base_activation),
        ]
        for _ in range(trunk_blocks - 1):
            trunk.append(nn.Conv2d(int(trunk_width), int(trunk_width), 1))
            trunk.append(nn.BatchNorm2d(int(trunk_width)))
            trunk.append(_activation(base_activation))
        self.trunk = nn.Sequential(*trunk)

        head: list[nn.Module] = [nn.Conv2d(int(trunk_width), int(num_classes), 1)]
        if readout == ACTIVATED_READOUT:
            head.append(_activation(base_activation))
        self.head = nn.Sequential(*head)
        self.flatten = nn.Flatten()

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
        return self.flatten(self.head(self.trunk(self.body(self.stem(x)))))
