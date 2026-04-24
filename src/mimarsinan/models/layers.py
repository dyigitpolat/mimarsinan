"""Thin re-exporter of activations and decorators, plus standalone layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.activations import (
    DifferentiableClamp,
    LeakyGradReLU,
    LeakyGradReLUFunction,
    LIFActivation,
    StaircaseFunction,
)
from mimarsinan.models.decorators import (
    ActivationReplacementDecorator,
    ClampDecorator,
    DecoratedActivation,
    MixAdjustmentStrategy,
    NestedAdjustmentStrategy,
    NestedDecoration,
    NoisyDropout,
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

    def decorate(self, decorator):
        self.decorators.append(decorator)

    def pop_decorator(self):
        return self.decorators.pop()

    def forward(self, x):
        # Flattened equivalent of the previous nested-DecoratedActivation chain:
        #   DecoratedActivation(DecoratedActivation(base, d0), d1)(x)
        # The outermost decorator's input_transform runs first, its
        # output_transform runs last — hence reversed for inputs, forward for
        # outputs. Avoids N nn.Module wrappers per forward (one per decorator).
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
            self.max_value.data = 0.1 * max_x + 0.9 * self.max_value

        return x / self.max_value


class FrozenStatsMaxValueScaler(nn.Module):
    def __init__(self, scaler: MaxValueScaler):
        super(FrozenStatsMaxValueScaler, self).__init__()
        self.max_value = scaler.max_value.clone().detach()

    def forward(self, x):
        return x / self.max_value
