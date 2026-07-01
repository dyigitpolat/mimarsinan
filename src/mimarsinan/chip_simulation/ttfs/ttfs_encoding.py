"""TTFS latched spike encoding (nevresim ``TTFSSpikeGenerator`` semantics)."""

from __future__ import annotations

import numpy as np

from mimarsinan.models.spiking.wire_semantics import (
    ttfs_grid_quantize_np,
    ttfs_spike_time_np,
)


def ttfs_spike_time(rate: np.ndarray, simulation_length: int) -> np.ndarray:
    """Per-element spike time: ``round(S * (1 - clamp(rate, 0, 1)))``."""
    return ttfs_spike_time_np(rate, simulation_length)


def ttfs_input_grid_quantize(rates: np.ndarray, simulation_length: int) -> np.ndarray:
    """Encode→decode round-trip: the value a single-spike timing can carry.

    Mirrors the synchronized cycle soma's decode ``(S - k)/S`` of a spike
    encoded at ``k = ttfs_spike_time(rate)``; ``k >= S`` never fires (0.0).
    """
    return ttfs_grid_quantize_np(rates, simulation_length)


def ttfs_latched_spike_train(rates: np.ndarray, simulation_length: int) -> np.ndarray:
    """Latched TTFS train ``(N, D, S)``: high from ``spike_time`` through ``S-1``."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError(f"rates must be (N, D); got shape {rates.shape}")
    n, d = rates.shape
    s = int(simulation_length)
    out = np.zeros((n, d, s), dtype=np.float64)
    spike_times = ttfs_spike_time(rates, s)
    for cycle in range(s):
        out[:, :, cycle] = (spike_times < s) & (cycle >= spike_times)
    return out


def ttfs_single_spike_train(rates: np.ndarray, simulation_length: int) -> np.ndarray:
    """Single-shot TTFS train ``(N, D, S)``: exactly one spike at ``spike_time``.

    Genuine single-spike encoding for ``ttfs_cycle_based`` (vs the latched train
    used by ``ttfs_quantized``). A neuron with rate 0 never fires.
    """
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError(f"rates must be (N, D); got shape {rates.shape}")
    n, d = rates.shape
    s = int(simulation_length)
    out = np.zeros((n, d, s), dtype=np.float64)
    spike_times = ttfs_spike_time(rates, s)
    for cycle in range(s):
        out[:, :, cycle] = (spike_times < s) & (cycle == spike_times)
    return out


def ttfs_input_spike_times_1based(rate: float, simulation_length: int) -> list[int]:
    """1-based SANA-FE ``input`` soma spike list for one scalar rate in ``[0, 1]``."""
    s = int(simulation_length)
    clamped = float(np.clip(rate, 0.0, 1.0))
    t0 = int(np.rint(s * (1.0 - clamped)))
    if t0 >= s:
        return []
    return [t0 + 1 + c for c in range(s - t0)]


def ttfs_always_on_spike_times_1based(simulation_length: int) -> list[int]:
    """Always-on fires at cycle 0 only (HCM TTFS boundary)."""
    s = int(simulation_length)
    if s <= 0:
        return []
    return [1]


def ttfs_latched_spike_times_1based(rate: float, simulation_length: int) -> list[int]:
    """1-based spike times for a latched input neuron (one onset, then every cycle)."""
    return ttfs_input_spike_times_1based(rate, simulation_length)
