"""Synchronized genuine single-spike TTFS reference (the spec for the C++ backends).

``ttfs_cycle_based`` runs latency groups (topological depth) **sequentially**: each
group runs a full ``S``-cycle window to completion before the next starts, so by the
time a neuron's window runs all its inputs are complete (causal). Total simulation
time is ``S × num_groups`` (synchronized), not ``S + latency`` (pipelined).

Inter-core signals are **single spikes**: a neuron fires at most once, at cycle
``k_fire = ⌈S·(1 − V/θ)⌉`` within its group window, encoding its value
``a = (S − k_fire)/S``. Downstream groups decode incoming spike *timings* back to
activations — no real values on the wire, no preset membrane. Because ``V`` is
reconstructed exactly from the completed inputs, the decoded result equals
``ttfs_quantized_activation(V, θ, S)`` per neuron (parity with the analytical path).

This module is the numerical reference / spec; the nevresim & SANA-FE somas realize
the same dynamics event-driven over the ``S × num_groups`` cycle timeline.
"""

from __future__ import annotations

import numpy as np

NO_SPIKE = -1  # sentinel: neuron never fired (encodes activation 0)


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

    ``W`` is ``(out, in)``; group g consumes group g-1's single-spike outputs (decoded).
    Returns ``(output_activations, per_layer_spike_times)`` — the spike times are the
    genuine single-spike representation propagated between groups.
    """
    a = np.asarray(input_activations, dtype=np.float64)
    per_layer_spike_times = []
    for (W, bias, threshold) in layers:
        V = a @ np.asarray(W, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)
        k = ttfs_cycle_fire_step(V, threshold, S)  # single spike per neuron
        per_layer_spike_times.append(k)
        a = decode_spike_time_to_activation(k, S)  # next group decodes the timings
    return a, per_layer_spike_times


def genuine_total_cycles(num_latency_groups: int, simulation_steps: int) -> int:
    """Synchronized timeline length: ``S × groups`` (each group runs a full window)."""
    return int(simulation_steps) * int(num_latency_groups)


def latency_groups(latencies) -> tuple[int, list[int]]:
    """Map per-core latency to topological groups.

    Cores sharing a latency form one group (they are independent / same depth).
    Returns ``(num_groups, per_core_group)`` where group ranks ascend with latency
    (group 0 = latency 0 = closest to input, last group = output). Cores with
    unknown latency (``None``) are placed in the final group.
    """
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
