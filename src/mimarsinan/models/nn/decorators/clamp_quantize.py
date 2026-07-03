"""Clamp and quantize decorators."""

from __future__ import annotations

from mimarsinan.models.nn.activations.autograd import DifferentiableClamp, StaircaseFunction


class ClampDecorator:
    """Clamps activations into ``[clamp_min, clamp_max]`` (fixed tensors)."""

    def __init__(self, clamp_min, clamp_max):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.clamp_min = self.clamp_min.to(x.device)
        self.clamp_max = self.clamp_max.to(x.device)
        return DifferentiableClamp.apply(x, self.clamp_min, self.clamp_max)


class QuantizeDecorator:
    def __init__(self, levels_before_c, c):
        self.levels_before_c = levels_before_c
        self.c = c

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.levels_before_c = self.levels_before_c.to(x.device)
        self.c = self.c.to(x.device)
        return StaircaseFunction.apply(x, self.levels_before_c / self.c)

