"""A host op whose output carries the zero padding of the convolutions that read it (U-SC2, the lifted boundary).

A negative-value boundary lifted into the op's own function adds sigma to every element the op emits and the reading
core's bias compensates sigma times its whole kernel. A convolution that pads its input with zeros reads no sigma at its
padded taps, so every position whose kernel reaches the padding departs by sigma times those taps' weights. With the
padding emitted by the op (``ZeroPaddedOutput``) and the convolution reading the padded output unpadded, the lift
reaches the padded zeros as well (they become the shifted zero), and the compensation is exact at every position: the
same function in every representation, since ``conv(pad0(v)) == conv_unpadded(pad0(v))``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ZeroPaddedOutput"]


class ZeroPaddedOutput(nn.Module):
    """``module``'s output with ``padding`` zeros on each side of its trailing spatial axes (``(pad_h, pad_w)`` for a
    channel-first ``(C, H, W)`` output, the order of ``nn.Conv2d.padding``)."""

    def __init__(self, module: nn.Module, padding) -> None:
        super().__init__()
        self.module = module
        self.padding = tuple(int(p) for p in padding)

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.module(*inputs, **kwargs)
        pads: list[int] = []
        for p in reversed(self.padding):
            pads += [p, p]
        return F.pad(out, pads)
