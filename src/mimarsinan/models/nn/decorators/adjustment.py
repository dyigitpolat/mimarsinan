"""Rate-adjusted and nested decorators."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.decorators.rate_buffer import RateBuffer

_ACTIVATION_TREE_CHILD_ATTRS = (
    "base_activation", "decorator", "adjustment_strategy", "target_activation",
)
_ACTIVATION_TREE_SEQ_ATTRS = ("decorators", "strategies")


def iter_activation_tree(node, seen=None):
    """Yield every object reachable from an activation ``node`` through the
    decorator/strategy containers (the shared walk under the stochastic-decision
    and shift-neutralization traversals)."""
    if seen is None:
        seen = set()
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for attr in _ACTIVATION_TREE_CHILD_ATTRS:
        yield from iter_activation_tree(getattr(node, attr, None), seen)
    for attr in _ACTIVATION_TREE_SEQ_ATTRS:
        seq = getattr(node, attr, None)
        if isinstance(seq, (list, tuple)):
            for child in seq:
                yield from iter_activation_tree(child, seen)


class RandomMaskAdjustmentStrategy:
    # ``generator=None`` preserves the legacy global-RNG draw bit-for-bit; a seeded generator makes the mask reproducible.
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

    def _resolved_rate(self):
        if isinstance(self.rate, RateBuffer):
            return float(self.rate.alpha)
        return self.rate

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
    """Blends the base activation output with a target activation (e.g. ReLU):
    rate 0 is entirely the base activation, rate 1 entirely the target."""

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

