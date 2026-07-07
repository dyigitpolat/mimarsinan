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


def freeze_batchnorm_running_stats(model) -> None:
    """Hold every BatchNorm in eval mode so training forwards cannot mutate
    running statistics (call after ``model.train()``)."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def norm_affine_params(normalization):
    """``(u, beta, mean)`` of a normalization's frozen-stats affine form
    ``u*(z-mean)+beta`` (differentiable through ``weight``/``bias``); works for
    ``nn.BatchNorm1d/2d`` and ``FrozenStatsNormalization``."""
    weight = normalization.weight
    var = normalization.running_var
    mean = normalization.running_mean
    # Transfer only on a measured mismatch: this runs per perceptron per
    # optimizer step, and a blocking H2D copy per call throttles the QAT.
    if var.device != weight.device:
        var = var.to(weight.device)
    if mean.device != weight.device:
        mean = mean.to(weight.device)
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
        # Non-persistent buffers: move with .to()/offload, absent from
        # state_dict (cache/golden/parity layouts unchanged).
        self.register_buffer(
            "running_mean", normalization.running_mean.clone().detach(),
            persistent=False,
        )
        self.register_buffer(
            "running_var", normalization.running_var.clone().detach(),
            persistent=False,
        )
        self.weight = normalization.weight
        self.bias = normalization.bias
        self.eps = normalization.eps

        self.affine = normalization.affine

    def __setstate__(self, state):
        super().__setstate__(state)
        # Caches saved before the buffer registration carry the stats as
        # plain attrs that .to()/offload cannot move; migrate them.
        for name in ("running_mean", "running_var"):
            if name not in self._buffers:
                self._buffers[name] = self.__dict__.pop(name, None)

    def forward(self, x):
        if self.running_mean.device != x.device:
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
