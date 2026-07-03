"""Thin re-exporter of activations and decorators, plus standalone layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import (
    DifferentiableClamp as DifferentiableClamp,
    LeakyGradReLU as LeakyGradReLU,
    LeakyGradReLUFunction as LeakyGradReLUFunction,
    LIFActivation as LIFActivation,
    StaircaseFunction as StaircaseFunction,
)
from mimarsinan.models.nn.decorators import (
    ActivationReplacementDecorator as ActivationReplacementDecorator,
    ClampDecorator as ClampDecorator,
    DecoratedActivation as DecoratedActivation,
    MixAdjustmentStrategy as MixAdjustmentStrategy,
    NestedAdjustmentStrategy as NestedAdjustmentStrategy,
    NestedDecoration as NestedDecoration,
    NoisyDropout as NoisyDropout,
    QuantizeDecorator as QuantizeDecorator,
    RandomMaskAdjustmentStrategy as RandomMaskAdjustmentStrategy,
    RateAdjustedDecorator as RateAdjustedDecorator,
    SavedTensorDecorator as SavedTensorDecorator,
    ScaleDecorator as ScaleDecorator,
    ShiftDecorator as ShiftDecorator,
    StatsDecorator as StatsDecorator,
)


def norm_affine_params(normalization):
    """``(u, beta, mean)`` of a normalization's frozen-stats affine form
    ``u*(z-mean)+beta`` (differentiable through ``weight``/``bias``); works for
    ``nn.BatchNorm1d/2d`` and ``FrozenStatsNormalization``."""
    weight = normalization.weight
    var = normalization.running_var.to(weight.device)
    mean = normalization.running_mean.to(weight.device)
    u = weight / torch.sqrt(var + normalization.eps)
    return u, normalization.bias, mean


class TransformedActivation(nn.Module):
    def __init__(self, base_activation, decorators):
        super(TransformedActivation, self).__init__()
        self.base_activation = base_activation
        self.decorators = decorators

    def decorate(self, decorator):
        self.decorators.append(decorator)

    def pop_decorator(self):
        return self.decorators.pop()

    def forward(self, x):
        for dec in reversed(self.decorators):
            x = dec.input_transform(x)
        x = self.base_activation(x)
        for dec in self.decorators:
            x = dec.output_transform(x)
        return x


class FrozenStatsNormalization(nn.Module):
    def __init__(self, normalization):
        super(FrozenStatsNormalization, self).__init__()
        self.running_mean = normalization.running_mean.clone().detach()
        self.running_var = normalization.running_var.clone().detach()
        self.weight = normalization.weight
        self.bias = normalization.bias
        self.eps = normalization.eps

        self.affine = normalization.affine

    def forward(self, x):
        self.weight = self.weight.to(x.device)
        self.bias = self.bias.to(x.device)
        self.running_mean = self.running_mean.to(x.device)
        self.running_var = self.running_var.to(x.device)

        return nn.functional.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias, False, 0, self.eps
        )


class MaxValueScaler(nn.Module):
    def __init__(self):
        super(MaxValueScaler, self).__init__()
        self.max_value = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        max_x = torch.max(x)

        if self.training:
            # Floor the EMA at a positive epsilon so an all-negative input stream cannot drive the divisor negative and flip output signs.
            ema = 0.1 * max_x + 0.9 * self.max_value
            self.max_value.data = torch.clamp(ema, min=1e-6)

        return x / self.max_value


class FrozenStatsMaxValueScaler(nn.Module):
    def __init__(self, scaler: MaxValueScaler):
        super(FrozenStatsMaxValueScaler, self).__init__()
        self.max_value = scaler.max_value.clone().detach()

    def forward(self, x):
        return x / self.max_value
