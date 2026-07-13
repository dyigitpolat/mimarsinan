"""Clamp and quantize decorators."""

from __future__ import annotations

import torch

from mimarsinan.models.nn.activations.autograd import (
    DifferentiableClamp,
    LIFCountStaircaseFunction,
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


class LIFCountStaircaseDecorator:
    """[lif_exact_qat] the deployed LIF count staircase ``θ·clamp(F(T·z/θ),0,T)/T``
    under staircase-STE with the in-loop LSQ theta gradient; strict/inclusive per
    ``thresholding_mode``. The LIF exact-QAT AQ activation — replaces the
    T-anneal + one-shot-fold proxy (lif_exact_qat_program.md §6.1)."""

    def __init__(self, simulation_steps, activation_scale, thresholding_mode="<="):
        if thresholding_mode not in ("<", "<="):
            raise ValueError(
                f"LIFCountStaircaseDecorator thresholding_mode must be '<' or "
                f"'<='; got {thresholding_mode!r}"
            )
        self.simulation_steps = int(simulation_steps)
        self.activation_scale = activation_scale
        self.thresholding_mode = str(thresholding_mode)

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        scale = self.activation_scale
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(float(scale))
        scale = scale.to(device=x.device, dtype=x.dtype)
        if scale.dim() > 1 or (
            scale.dim() == 1 and (x.dim() < 1 or scale.numel() != int(x.shape[-1]))
        ):
            raise ValueError(
                f"LIFCountStaircaseDecorator theta must be scalar or a "
                f"channels-last vector matching x's last dim; got theta shape "
                f"{tuple(scale.shape)} for x shape {tuple(x.shape)}"
            )
        return LIFCountStaircaseFunction.apply(
            x, scale, self.simulation_steps, self.thresholding_mode == "<",
        )


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

