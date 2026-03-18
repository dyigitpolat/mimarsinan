"""Composable decorators and adjustment strategies for activations."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.activations import DifferentiableClamp, StaircaseFunction


class NoisyDropout(nn.Module):
    def __init__(self, dropout_p, rate, noise_radius):
        super(NoisyDropout, self).__init__()
        self.dropout_p = dropout_p
        self.rate = rate
        self.noise_radius = noise_radius

    def forward(self, x):
        random_mask = torch.rand(x.shape, device=x.device)
        random_mask = (random_mask < self.rate).float()

        out = nn.Dropout(self.dropout_p)(x)
        out = out + self.noise_radius * torch.rand_like(out) - 0.5 * self.noise_radius
        return random_mask * out + (1.0 - random_mask) * x


class NoiseDecorator:
    def __init__(self, rate, noise_radius):
        self.rate = rate
        self.noise_radius = noise_radius

    def input_transform(self, x):
        return nn.Identity()(x)

    def output_transform(self, x):
        return NoisyDropout(torch.tensor(0.0), self.rate, self.noise_radius)(x)


class SavedTensorDecorator(nn.Module):
    def __init__(self):
        super(SavedTensorDecorator, self).__init__()
        self.latest_input = None
        self.latest_output = None

    def input_transform(self, x):
        if len(x.shape) > 1:
            self.latest_input = x

        return nn.Identity()(x)

    def output_transform(self, x):
        if len(x.shape) > 1:
            self.latest_output = x

        return nn.Identity()(x)


class StatsDecorator:
    def __init__(self):
        self.in_mean = None
        self.in_var = None
        self.in_max = None
        self.in_min = None

        self.in_hist = None
        self.in_hist_bin_edges = None

        self.out_mean = None
        self.out_var = None
        self.out_max = None
        self.out_min = None

        self.out_hist = None
        self.out_hist_bin_edges = None

    def input_transform(self, x):
        if len(x.shape) > 1:
            self.in_mean = torch.mean(x)
            self.in_var = torch.var(x)
            self.in_max = torch.max(x)
            self.in_min = torch.min(x)

            self.in_hist = torch.histc(x.flatten(), bins=100, min=self.in_min.item(), max=self.in_max.item())
            self.in_hist_bin_edges = torch.linspace(self.in_min.item(), self.in_max.item(), steps=101)

        return nn.Identity()(x)

    def output_transform(self, x):
        if len(x.shape) > 1:
            self.out_mean = torch.mean(x)
            self.out_var = torch.var(x)
            self.out_max = torch.max(x)
            self.out_min = torch.min(x)

            self.out_hist = torch.histc(x.flatten(), bins=100, min=self.out_min.item(), max=self.out_max.item())
            self.out_hist_bin_edges = torch.linspace(self.out_min.item(), self.out_max.item(), steps=101)

        return nn.Identity()(x)


class ShiftDecorator:
    def __init__(self, shift):
        self.shift = shift

    def input_transform(self, x):
        self.shift = self.shift.to(x.device)
        return torch.sub(x, self.shift)

    def output_transform(self, x):
        return nn.Identity()(x)


class ScaleDecorator:
    def __init__(self, scale):
        self.scale = scale

    def input_transform(self, x):
        return nn.Identity()(x)

    def output_transform(self, x):
        self.scale = self.scale.to(x.device)
        return self.scale * x


class ClampDecorator:
    def __init__(self, clamp_min, clamp_max):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def input_transform(self, x):
        return nn.Identity()(x)

    def output_transform(self, x):
        self.clamp_min = self.clamp_min.to(x.device)
        self.clamp_max = self.clamp_max.to(x.device)
        return DifferentiableClamp.apply(x, self.clamp_min, self.clamp_max)


class QuantizeDecorator:
    def __init__(self, levels_before_c, c):
        self.levels_before_c = levels_before_c
        self.c = c

    def input_transform(self, x):
        return nn.Identity()(x)

    def output_transform(self, x):
        self.levels_before_c = self.levels_before_c.to(x.device)
        self.c = self.c.to(x.device)
        return StaircaseFunction.apply(x, self.levels_before_c / self.c)


class RandomMaskAdjustmentStrategy:
    def adjust(self, base, target, rate):
        random_mask = torch.rand(base.shape, device=base.device)
        random_mask = (random_mask < rate).float()
        return random_mask * target + (1.0 - random_mask) * base


class NestedAdjustmentStrategy:
    def __init__(self, strategies):
        self.strategies = strategies

    def adjust(self, base, target, rate):
        for strategy in self.strategies:
            target = strategy.adjust(base, target, rate)
        return target


class RateAdjustedDecorator:
    def __init__(self, rate, decorator, adjustment_strategy):
        self.rate = rate
        self.decorator = decorator
        self.adjustment_strategy = adjustment_strategy

    def input_transform(self, x):
        return self.adjustment_strategy.adjust(x, self.decorator.input_transform(x), self.rate)

    def output_transform(self, x):
        return self.adjustment_strategy.adjust(x, self.decorator.output_transform(x), self.rate)


class NestedDecoration:
    def __init__(self, decorators):
        self.decorators = decorators

    def input_transform(self, x):
        for decorator in reversed(self.decorators):
            x = decorator.input_transform(x)
        return x

    def output_transform(self, x):
        for decorator in self.decorators:
            x = decorator.output_transform(x)
        return x


class DecoratedActivation(nn.Module):
    def __init__(self, base_activation, decorator):
        super(DecoratedActivation, self).__init__()
        self.base_activation = base_activation
        self.decorator = decorator

    def forward(self, x):
        out = self.decorator.input_transform(x)
        out = self.base_activation(out)
        out = self.decorator.output_transform(out)
        return out


class MixAdjustmentStrategy:
    def adjust(self, base, target, rate):
        return rate * target + (1.0 - rate) * base


class ActivationReplacementDecorator:
    """Blends the base activation output with a target activation (e.g. ReLU).

    At rate 0 the output is entirely from the base activation; at rate 1
    it is entirely from the target activation.  Uses MixAdjustmentStrategy
    for a smooth linear blend.
    """

    def __init__(self, target_activation):
        self.target_activation = target_activation
        self._saved_input = None

    def input_transform(self, x):
        self._saved_input = x
        return x

    def output_transform(self, x):
        target_out = self.target_activation(self._saved_input)
        self._saved_input = None
        return target_out


class AnyDecorator:
    def __init__(self, any_module):
        self.module = any_module

    def input_transform(self, x):
        return nn.Identity()(x)

    def output_transform(self, x):
        return self.module(x)
