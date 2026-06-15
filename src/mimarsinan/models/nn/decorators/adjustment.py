"""Rate-adjusted and nested decorators."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.decorators.rate_buffer import RateBuffer


class RandomMaskAdjustmentStrategy:
    # ``generator`` (default None) makes the per-element mask reproducible: an
    # axis that called ``set_decision_seed`` wires a seeded ``torch.Generator``
    # here. None preserves the legacy global-RNG draw bit-for-bit.
    def __init__(self, generator=None):
        self._generator = generator

    def adjust(self, base, target, rate):
        if self._generator is not None:
            random_mask = torch.rand(base.shape, device=base.device, generator=self._generator)
        else:
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

    # ``rate`` may be a plain float or a shared ``RateBuffer`` (in-place ramp).
    # The buffer carries one scalar ``alpha`` read live at transform time so the
    # rate advances without rebuilding the decorator stack; resolving it to a
    # float here keeps the 0/1 short-circuits and the adjust call bit-identical.
    def _resolved_rate(self):
        if isinstance(self.rate, RateBuffer):
            return float(self.rate.alpha)
        return self.rate

    # Rate=0 is the identity (all strategies reduce to "return base"), so skip
    # the inner decorator call and the adjustment tensor ops. Rate=1 with
    # Mix/RandomMask/Nested strategies collapses to "return target" (mask is
    # fully 1 everywhere: rand() < 1.0 is always True since rand() is [0,1)).
    # Note: rate=0 short-circuit with RandomMask-based strategies skips a
    # torch.rand() call that the un-optimized path would consume; this shifts
    # global RNG state slightly but has no semantic effect on outputs.
    def input_transform(self, x):
        rate = self._resolved_rate()
        if rate == 0.0:
            return x
        target = self.decorator.input_transform(x)
        if rate == 1.0:
            return target
        return self.adjustment_strategy.adjust(x, target, rate)

    def output_transform(self, x):
        rate = self._resolved_rate()
        if rate == 0.0:
            return x
        target = self.decorator.output_transform(x)
        if rate == 1.0:
            return target
        return self.adjustment_strategy.adjust(x, target, rate)


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

