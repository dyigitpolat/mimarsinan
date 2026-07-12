"""Clamp and quantize decorators."""

from __future__ import annotations

import torch

from mimarsinan.models.nn.activations.autograd import (
    DifferentiableClamp,
    StaircaseFunction,
    TTFSComparatorHalfStepStaircaseFunction,
    TTFSStaircaseFunction,
)


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


class TTFSCeilStaircaseDecorator:
    """The deployed TTFS ceil kernel ``y = θ·ceil_staircase(x/θ, S)`` with STE.

    The exact synchronized/ttfs_quantized deployment activation (MBH T6
    endpoint) — replaces the floor staircase + half-step shift QAT proxy.
    ``comparator_half_step`` (the contract's [E3] flag) runs the shifted
    compare ladder so the NF twin moves with the deployed comparator."""

    def __init__(self, simulation_steps, activation_scale, comparator_half_step=False):
        self.simulation_steps = int(simulation_steps)
        self.activation_scale = activation_scale
        self.comparator_half_step = bool(comparator_half_step)

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        scale = self.activation_scale
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(float(scale))
        scale = scale.to(device=x.device, dtype=x.dtype)
        safe_scale = scale.clamp(min=1e-12)
        staircase = (
            TTFSComparatorHalfStepStaircaseFunction
            if self.comparator_half_step
            else TTFSStaircaseFunction
        )
        return staircase.apply(x / safe_scale, self.simulation_steps) * safe_scale


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

