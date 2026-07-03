"""Synchronized genuine single-spike TTFS reference (the spec for the C++ backends)."""

from __future__ import annotations

import numpy as np

NO_SPIKE = -1


def encode_activation_to_spike_time(a: np.ndarray, S: int) -> np.ndarray:
    """Activation in [0,1] → single-spike fire cycle ``k = ⌈S·(1−a)⌉`` (NO_SPIKE if 0)."""
    a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
    k = np.ceil(S * (1.0 - a))
    return np.where(k < S, k.astype(np.int64), NO_SPIKE)


def decode_spike_time_to_activation(k: np.ndarray, S: int) -> np.ndarray:
    """Single-spike fire cycle → activation ``(S − k)/S`` (0 for NO_SPIKE)."""
    k = np.asarray(k)
    return np.where(k == NO_SPIKE, 0.0, (S - k.astype(np.float64)) / S)


def ttfs_cycle_fire_step(V: np.ndarray, threshold: float, S: int) -> np.ndarray:
    """Synchronized single-spike fire cycle for membrane ``V`` (NO_SPIKE if it never fires)."""
    theta = max(float(threshold), 1e-12)
    k_raw = np.ceil(S * (1.0 - np.asarray(V, dtype=np.float64) / theta))
    fires = k_raw < S
    k = np.clip(k_raw, 0, S - 1).astype(np.int64)
    return np.where(fires, k, NO_SPIKE)


def run_ttfs_cycle_genuine_layers(layers, input_activations: np.ndarray, S: int):
    """Synchronized genuine sim over a feedforward list of ``(W, bias, threshold)``.

    ``W`` is ``(out, in)``; group g consumes group g-1's single-spike outputs.
    Returns ``(output_activations, per_layer_spike_times)``."""
    a = np.asarray(input_activations, dtype=np.float64)
    per_layer_spike_times = []
    for (W, bias, threshold) in layers:
        V = a @ np.asarray(W, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)
        k = ttfs_cycle_fire_step(V, threshold, S)
        per_layer_spike_times.append(k)
        a = decode_spike_time_to_activation(k, S)
    return a, per_layer_spike_times


def genuine_total_cycles(num_latency_groups: int, simulation_steps: int) -> int:
    """Synchronized timeline length: ``S × groups`` (each group runs a full window)."""
    return int(simulation_steps) * int(num_latency_groups)


def latency_groups(latencies) -> tuple[int, list[int]]:
    """Map per-core latency to ascending topological groups (group 0 = input side).

    Cores sharing a latency form one group; unknown latency (``None``) goes last.
    Returns ``(num_groups, per_core_group)``."""
    vals = [int(l) for l in latencies if l is not None]
    distinct = sorted(set(vals))
    rank = {v: i for i, v in enumerate(distinct)}
    num_groups = max(len(distinct), 1)
    last = num_groups - 1
    per_core = [rank[int(l)] if l is not None else last for l in latencies]
    return num_groups, per_core


def synchronized_window(group: int, simulation_steps: int) -> tuple[int, int]:
    """Non-overlapping ``S``-cycle window for a latency group: ``[g·S, (g+1)·S)``."""
    s = int(simulation_steps)
    return group * s, s
