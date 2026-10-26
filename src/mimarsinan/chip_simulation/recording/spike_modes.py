"""Torch spike-mode encoders shared by spiking core flows."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.models.spiking.wire_semantics import ttfs_spike_time


def to_stochastic_spikes(tensor: torch.Tensor) -> torch.Tensor:
    return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()


def to_front_loaded_spikes(tensor: torch.Tensor, cycle: int, simulation_length: int) -> torch.Tensor:
    # Chip's FrontLoadedSpikeGenerator: llround(item*T) > cycle (same comb
    # count tie rule as Uniform — see comb_spike_count).
    return (comb_spike_count(tensor, simulation_length) > cycle).float()


def to_deterministic_spikes(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (tensor > threshold).float()


def to_ttfs_latched_spikes(tensor: torch.Tensor, cycle: int, simulation_length: int) -> torch.Tensor:
    """Latched time-to-first-spike: high from ``round(T*(1-rate))`` through ``T-1``.

    Matches ``ttfs_encoding.ttfs_latched_spike_train`` and nevresim
    ``TTFSSpikeGenerator``; rate 0 never fires."""
    T = simulation_length
    spike_time = ttfs_spike_time(tensor, T)
    return ((spike_time < T) & (cycle >= spike_time)).float()


# Golden-ratio channel phases: irrational stride maximally decorrelates comb
# alignment across channels while staying deterministic and index-stable.
_PHASE_GOLDEN = 0.6180339887498949


def comb_spike_count(rates: torch.Tensor, simulation_length: int) -> torch.Tensor:
    """[nevresim parity] the uniform comb's spike COUNT — the chip's tie rule.

    ``UniformSpikeGenerator`` computes ``llround(rate * T)`` in C++ double:
    exact half-integer products round AWAY from zero. ``torch.round`` is
    half-to-even and resolved those ties (rates on the odd 1/(2T) lattice —
    edges LSQ exact-QAT trains values ONTO) OPPOSITE to the chip. One rule,
    used by every count seam that mirrors the chip's comb."""
    x = rates.to(torch.float64) * float(simulation_length)
    f = torch.floor(x)
    return (f + ((x - f) >= 0.5).to(torch.float64)).to(torch.long)


def comb_spike_count_np(rates, simulation_length: int):
    """Numpy mirror of :func:`comb_spike_count` (same llround tie rule)."""
    x = np.asarray(rates, dtype=np.float64) * float(simulation_length)
    f = np.floor(x)
    return (f + (x - f >= 0.5)).astype(np.int64)


def channel_phase_offsets(
    channels: torch.Tensor, simulation_length: int, device: torch.device | None = None,
) -> torch.Tensor:
    """[E1] Comb rotation (whole cycles in [0, T)) of each channel id in
    ``channels`` for phase-dithered uniform encodes; rotation mod T preserves
    every channel's count exactly. A chip stage passes each stage-input
    channel's channel at unbounded geometry (``phase_channels``), so no
    cost-only decision moves a rotation."""
    idx = torch.as_tensor(channels, device=device).to(dtype=torch.float32)
    return torch.floor((idx * _PHASE_GOLDEN).frac() * simulation_length)


def uniform_phase_offsets(
    n_channels: int, simulation_length: int, device: torch.device | None = None,
) -> torch.Tensor:
    """Per-channel comb rotation of channels ``0 .. n_channels - 1`` (a
    producer's own output: the NF twin's and the host boundary's encode)."""
    return channel_phase_offsets(torch.arange(n_channels, device=device), simulation_length, device)


def to_uniform_spikes(
    tensor: torch.Tensor,
    cycle: int,
    simulation_length: int,
    phase_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    # [nevresim parity] comb arithmetic is canonically FLOAT64: the chip's
    # UniformSpikeGenerator computes in double, and f32 spacing flips spike
    # PLACEMENT (same count) at exact-division knife-edges — e.g. n=12,
    # T=32, spacing 8/3: double fires cycle 8, f32 fires cycle 9 (the
    # t0_04 ±1-window class across every discipline). The COUNT uses the
    # chip's llround tie rule (comb_spike_count), not torch.round.
    T = simulation_length
    t64 = tensor.to(torch.float64)
    n = comb_spike_count(tensor, T)
    mask = (n != 0) & (n != T) & (cycle < T)
    n_safe = torch.clamp(n, min=1)
    spacing = T / n_safe.to(torch.float64)
    if phase_offsets is None:
        result = mask & (torch.floor(cycle / spacing) < n_safe) & (torch.floor(cycle % spacing) == 0)
    else:
        e = (float(cycle) + phase_offsets.to(device=tensor.device, dtype=torch.float64)) % T
        result = mask & (torch.floor(e / spacing) < n_safe) & (torch.floor(e % spacing) == 0)
    result = result.to(tensor.dtype)
    result[n == T] = 1.0
    return result


def to_spikes(
    tensor: torch.Tensor,
    cycle: int,
    *,
    simulation_length: int,
    spike_mode: str,
) -> torch.Tensor:
    if spike_mode == "Stochastic":
        return to_stochastic_spikes(tensor)
    if spike_mode == "Deterministic":
        return to_deterministic_spikes(tensor)
    if spike_mode == "FrontLoaded":
        return to_front_loaded_spikes(tensor, cycle, simulation_length)
    if spike_mode == "Uniform":
        return to_uniform_spikes(tensor, cycle, simulation_length)
    if spike_mode == "TTFS":
        return to_ttfs_latched_spikes(tensor, cycle, simulation_length)
    raise ValueError("Invalid spike mode: " + str(spike_mode))
