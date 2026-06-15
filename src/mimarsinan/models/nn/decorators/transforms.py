"""Basic input/output tensor decorators."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class NoisyDropout(nn.Module):
    # ``generator`` (default None) makes the mask + additive noise reproducible
    # for a stochastic axis that called ``set_decision_seed``; None preserves the
    # legacy global-RNG draws bit-for-bit. (nn.Dropout itself is identity in eval,
    # so the seeded generator fully determines the eval-mode forward.)
    def __init__(self, dropout_p, rate, noise_radius, generator=None):
        super(NoisyDropout, self).__init__()
        self.dropout_p = dropout_p
        self.rate = rate
        self.noise_radius = noise_radius
        self._generator = generator
        # Cache the Dropout module instead of instantiating one per forward.
        # dropout_p may be a 0-d tensor; nn.Dropout wants a float.
        p = float(dropout_p.item()) if isinstance(dropout_p, torch.Tensor) else float(dropout_p)
        self._dropout = nn.Dropout(p)

    def forward(self, x):
        # Rate=0 is the common case during non-noise tuning cycles; skip the
        # two torch.rand allocations and the dropout + mix ops entirely.
        if self.rate == 0.0:
            return x

        if self._generator is not None:
            random_mask = torch.rand(x.shape, device=x.device, generator=self._generator)
        else:
            random_mask = torch.rand(x.shape, device=x.device)
        random_mask = (random_mask < self.rate).float()

        out = self._dropout(x)
        if self._generator is not None:
            noise = torch.rand(out.shape, dtype=out.dtype, device=out.device, generator=self._generator)
        else:
            noise = torch.rand_like(out)
        out = out + self.noise_radius * noise - 0.5 * self.noise_radius
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


