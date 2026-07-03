"""An in-place rate carrier so an adaptation ramp need not rebuild decorators."""

from __future__ import annotations

import torch
import torch.nn as nn


class RateBuffer(nn.Module):
    """A shared scalar rate held in a registered buffer; ``set`` fills in place.
    A ``RateAdjustedDecorator`` reads the live ``alpha`` at transform time, so
    advancing the rate is an O(1) in-place write, not a decorator rebuild."""

    alpha: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.zeros(()))

    def set(self, a) -> None:
        self.alpha.fill_(float(a))

    def __float__(self) -> float:
        return float(self.alpha)
