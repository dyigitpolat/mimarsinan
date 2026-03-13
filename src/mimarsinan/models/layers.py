"""Thin re-exporter of activations and decorators, plus standalone layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.activations import (
    DifferentiableClamp,
    LeakyGradReLU,
    LeakyGradReLUFunction,
    StaircaseFunction,
)
from mimarsinan.models.decorators import (
    AnyDecorator,
    ClampDecorator,
    DecoratedActivation,
    MixAdjustmentStrategy,
    NestedAdjustmentStrategy,
    NestedDecoration,
    NoisyDropout,
    NoiseDecorator,
    QuantizeDecorator,
    RandomMaskAdjustmentStrategy,
    RateAdjustedDecorator,
    SavedTensorDecorator,
    ScaleDecorator,
    ShiftDecorator,
    StatsDecorator,
)


class TransformedActivation(nn.Module):
    def __init__(self, base_activation, decorators):
        super(TransformedActivation, self).__init__()
        self.base_activation = base_activation
        self.decorators = decorators
        self._update_activation()

    def decorate(self, decorator):
        self.decorators.append(decorator)
        self._update_activation()

    def pop_decorator(self):
        popped_decorator = self.decorators.pop()
        self._update_activation()
        return popped_decorator

    def forward(self, x):
        return self.act(x)

    def _update_activation(self):
        self.act = self.base_activation
        for decorator in self.decorators:
            self.act = DecoratedActivation(self.act, decorator)


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
            self.max_value.data = 0.1 * max_x + 0.9 * self.max_value

        return x / self.max_value


class FrozenStatsMaxValueScaler(nn.Module):
    def __init__(self, scaler: MaxValueScaler):
        super(FrozenStatsMaxValueScaler, self).__init__()
        self.max_value = scaler.max_value.clone().detach()

    def forward(self, x):
        return x / self.max_value
