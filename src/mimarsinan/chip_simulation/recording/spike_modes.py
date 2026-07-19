"""Torch spike-mode encoders shared by spiking core flows."""

from __future__ import annotations

import torch

from mimarsinan.models.spiking.wire_semantics import ttfs_spike_time


def to_stochastic_spikes(tensor: torch.Tensor) -> torch.Tensor:
    return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()


def to_front_loaded_spikes(tensor: torch.Tensor, cycle: int, simulation_length: int) -> torch.Tensor:
    return (torch.round(tensor * simulation_length) > cycle).float()


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


def uniform_phase_offsets(
    n_channels: int, simulation_length: int, device: torch.device | None = None,
) -> torch.Tensor:
    """Per-channel comb rotation (whole cycles in [0, T)) for phase-dithered
    uniform encodes; rotation mod T preserves every channel's count exactly."""
    idx = torch.arange(n_channels, device=device, dtype=torch.float32)
    return torch.floor((idx * _PHASE_GOLDEN).frac() * simulation_length)


def to_uniform_spikes(
    tensor: torch.Tensor,
    cycle: int,
    simulation_length: int,
    phase_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    T = simulation_length
    n = torch.round(tensor * T).to(torch.long)
    mask = (n != 0) & (n != T) & (cycle < T)
    n_safe = torch.clamp(n, min=1)
    spacing = T / n_safe.float()
    if phase_offsets is None:
        result = mask & (torch.floor(cycle / spacing) < n_safe) & (torch.floor(cycle % spacing) == 0)
    else:
        e = (float(cycle) + phase_offsets.to(tensor.device)) % T
        result = mask & (torch.floor(e / spacing) < n_safe) & (torch.floor(e % spacing) == 0)
    result = result.float()
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
