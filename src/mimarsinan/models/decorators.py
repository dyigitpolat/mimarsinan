"""Composable decorators and adjustment strategies for activations."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from mimarsinan.models.activations import DifferentiableClamp, StaircaseFunction


class SavedTensorDecorator(nn.Module):
    """Captures layer I/O for inspection.

    Default mode stores a **reference** to the tensor (fast, zero-copy),
    which is appropriate when the saved tensor feeds into a differentiable
    loss (see :class:`CustomClassificationLoss`). For statistics collection
    -- activation scale estimation, clamp saturation checks, input
    histograms -- use ``sample_to_cpu=True``. In that mode the decorator
    immediately detaches, subsamples (deterministic linspace), and moves
    the tensor to CPU inside the forward pass, so the GPU activation can
    be freed as soon as the forward continues.

    This is the only sustainable approach for large-input backbones (e.g.
    ViT-B/16 at 224x224): without sampling, every perceptron's full output
    tensor stays pinned in VRAM for the whole forward, which compounds to
    >20 GiB on CIFAR-10 + batch 128 and OOMs any step that attaches saved
    decorators to all perceptrons simultaneously.

    When ``sample_to_cpu`` is enabled, consumers receive a flat 1-D float32
    CPU tensor of length ``min(numel, max_samples)``. All distribution-level
    consumers (quantiles, histograms, saturation ratios) are accurate under
    subsampling.
    """

    def __init__(self, *, sample_to_cpu: bool = False, max_samples: int = 8192):
        super(SavedTensorDecorator, self).__init__()
        self.latest_input = None
        self.latest_output = None
        self._sample_to_cpu = bool(sample_to_cpu)
        self._max_samples = int(max_samples) if max_samples and max_samples > 0 else 0

    def _maybe_sample(self, x):
        if not self._sample_to_cpu:
            return x
        flat = x.detach().reshape(-1).to(torch.float32)
        if self._max_samples <= 0 or flat.numel() <= self._max_samples:
            return flat.cpu()
        idx = torch.linspace(
            0, flat.numel() - 1, steps=self._max_samples, device=flat.device
        ).round().long()
        # Belt-and-braces: linspace endpoint is theoretically in-bounds, but
        # clamp in case of any float32 rounding drift for very large flats.
        idx.clamp_(0, flat.numel() - 1)
        # Under cuda_debug, sync before the first .cpu() so an async failure
        # from a prior forward kernel surfaces at its own mapper, not here.
        if os.environ.get("MIMARSINAN_CUDA_DEBUG") == "1" and flat.is_cuda:
            torch.cuda.synchronize()
        return flat.index_select(0, idx).cpu()

    def input_transform(self, x):
        if len(x.shape) > 1:
            self.latest_input = self._maybe_sample(x)

        return x

    def output_transform(self, x):
        if len(x.shape) > 1:
            self.latest_output = self._maybe_sample(x)

        return x


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

        return x

    def output_transform(self, x):
        if len(x.shape) > 1:
            self.out_mean = torch.mean(x)
            self.out_var = torch.var(x)
            self.out_max = torch.max(x)
            self.out_min = torch.min(x)

            self.out_hist = torch.histc(x.flatten(), bins=100, min=self.out_min.item(), max=self.out_max.item())
            self.out_hist_bin_edges = torch.linspace(self.out_min.item(), self.out_max.item(), steps=101)

        return x


class ShiftDecorator:
    def __init__(self, shift):
        self.shift = shift

    def input_transform(self, x):
        self.shift = self.shift.to(x.device)
        return torch.sub(x, self.shift)

    def output_transform(self, x):
        return x


class ClampDecorator:
    def __init__(self, clamp_min, clamp_max):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.clamp_min = self.clamp_min.to(x.device)
        self.clamp_max = self.clamp_max.to(x.device)
        return DifferentiableClamp.apply(x, self.clamp_min, self.clamp_max)


class LearnableClampDecorator:
    """Clamp decorator whose ceiling is taken live from a ``Perceptron``'s
    ``effective_clamp_ceiling()`` each forward pass.  This is the Phase-B2
    variant that lets ``ClampTuner`` drive the ceiling with gradient
    descent: once ``perceptron.log_clamp_ceiling.requires_grad`` is True,
    the forward pass builds a fresh ``exp(log_clamp_ceiling)`` tensor per
    call so gradients flow back into the log-space parameter.  When
    ``requires_grad`` is False the behaviour is identical to the plain
    ``ClampDecorator`` -- the ceiling is just a frozen number.
    """

    def __init__(self, perceptron, clamp_min):
        self.perceptron = perceptron
        self.clamp_min = clamp_min

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.clamp_min = self.clamp_min.to(x.device)
        ceiling = self.perceptron.effective_clamp_ceiling().to(x.device)
        return DifferentiableClamp.apply(x, self.clamp_min, ceiling)


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


class RateAdjustedDecorator:
    def __init__(self, rate, decorator, adjustment_strategy):
        self.rate = rate
        self.decorator = decorator
        self.adjustment_strategy = adjustment_strategy

    # Rate=0 is the identity (all strategies reduce to "return base"), so skip
    # the inner decorator call and the adjustment tensor ops. Rate=1 with
    # Mix/RandomMask/Nested strategies collapses to "return target" (mask is
    # fully 1 everywhere: rand() < 1.0 is always True since rand() is [0,1)).
    # Note: rate=0 short-circuit with RandomMask-based strategies skips a
    # torch.rand() call that the un-optimized path would consume; this shifts
    # global RNG state slightly but has no semantic effect on outputs.
    def input_transform(self, x):
        if self.rate == 0.0:
            return x
        target = self.decorator.input_transform(x)
        if self.rate == 1.0:
            return target
        return self.adjustment_strategy.adjust(x, target, self.rate)

    def output_transform(self, x):
        if self.rate == 0.0:
            return x
        target = self.decorator.output_transform(x)
        if self.rate == 1.0:
            return target
        return self.adjustment_strategy.adjust(x, target, self.rate)


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
