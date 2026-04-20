"""Composable decorators and adjustment strategies for activations."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from mimarsinan.models.activations import DifferentiableClamp, StaircaseFunction


class NoisyDropout(nn.Module):
    def __init__(self, dropout_p, rate, noise_radius):
        super(NoisyDropout, self).__init__()
        self.dropout_p = dropout_p
        self.rate = rate
        self.noise_radius = noise_radius
        # Cache the Dropout module instead of instantiating one per forward.
        # dropout_p may be a 0-d tensor; nn.Dropout wants a float.
        p = float(dropout_p.item()) if isinstance(dropout_p, torch.Tensor) else float(dropout_p)
        self._dropout = nn.Dropout(p)

    def forward(self, x):
        # Rate=0 is the common case during non-noise tuning cycles; skip the
        # two torch.rand allocations and the dropout + mix ops entirely.
        if self.rate == 0.0:
            return x

        random_mask = torch.rand(x.shape, device=x.device)
        random_mask = (random_mask < self.rate).float()

        out = self._dropout(x)
        out = out + self.noise_radius * torch.rand_like(out) - 0.5 * self.noise_radius
        return random_mask * out + (1.0 - random_mask) * x


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


class ScaleDecorator:
    def __init__(self, scale):
        self.scale = scale

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.scale = self.scale.to(x.device)
        return self.scale * x


class ClampDecorator:
    """Clamps activations into ``[clamp_min, clamp_max]``.

    ``clamp_max`` is either:

    - a fixed tensor (legacy behaviour — unchanged), or
    - ``None``, with a learnable ``scale_param`` (an ``nn.Parameter``)
      supplied instead. The clamp ceiling is then the scalar value of
      ``scale_param`` on the forward pass, and gradients flow back into
      ``scale_param`` through ``DifferentiableClamp`` (which passes
      through the ceiling for saturating inputs).

    When ``scale_param`` is provided, the ``ClampTuner`` is responsible
    for (a) adding it to the optimiser's parameter group, (b) applying a
    regulariser (see :func:`clamp_tuner.clamp_scale_regulariser`), and
    (c) freezing the learned scalar back onto the perceptron via
    :func:`clamp_tuner.freeze_learnable_scale` before the step commits.
    """

    def __init__(self, clamp_min, clamp_max=None, *, scale_param=None):
        assert (clamp_max is None) ^ (scale_param is None), (
            "ClampDecorator requires exactly one of clamp_max or scale_param"
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale_param = scale_param

    def _effective_clamp_max(self, x):
        # ``scale_param`` was added after some pickles were produced. Old
        # pickled ``ClampDecorator`` instances won't have this attribute at
        # all — fall back gracefully to ``clamp_max`` (which those pickles
        # do populate) so loading a pre-refactor IR graph still works.
        scale_param = getattr(self, "scale_param", None)
        if scale_param is not None:
            return scale_param.to(x.device)
        return self.clamp_max.to(x.device)

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.clamp_min = self.clamp_min.to(x.device)
        clamp_max = self._effective_clamp_max(x)
        return DifferentiableClamp.apply(x, self.clamp_min, clamp_max)


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
