"""Basic input/output tensor decorators."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class NoisyDropout(nn.Module):
    # ``generator=None`` preserves the legacy global-RNG draws bit-for-bit; a seeded generator makes the mask and noise reproducible.
    def __init__(self, dropout_p, rate, noise_radius, generator=None):
        super(NoisyDropout, self).__init__()
        self.dropout_p = dropout_p
        self.rate = rate
        self.noise_radius = noise_radius
        self._generator = generator
        p = float(dropout_p.item()) if isinstance(dropout_p, torch.Tensor) else float(dropout_p)
        self._dropout = nn.Dropout(p)

    def forward(self, x):
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
    """Captures layer I/O for inspection. Default stores a zero-copy reference;
    ``sample_to_cpu=True`` detaches, subsamples (deterministic linspace), and moves
    to CPU inside forward so large-backbone activations are freed promptly."""

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
        idx.clamp_(0, flat.numel() - 1)
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


