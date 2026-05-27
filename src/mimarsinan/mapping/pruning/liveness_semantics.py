"""Spiking-mode predicates for IR liveness / bias-only activation checks.

Single source of truth for whether ``hardware_bias`` can make a core
contributing when all axon weights are dead. Used by :mod:`ir_liveness` and
:mod:`ir_pruning` so pruning matches HCM / TTFS analytical semantics.
"""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.spiking_semantics import TTFS_MODES


def bias_can_activate(
    *,
    spiking_mode: str,
    bias,
    threshold: float,
    simulation_steps: int,
    zero_threshold: float = 1e-8,
    bias_only_emission_check: str = "conservative",
) -> bool:
    """Return whether bias alone can produce downstream activity for ``spiking_mode``."""
    mode = str(spiking_mode or "lif").lower()
    if mode == "ttfs":
        return ttfs_continuous_bias_can_activate(
            bias=bias, zero_threshold=zero_threshold,
        )
    if mode == "ttfs_quantized":
        return ttfs_quantized_bias_can_activate(
            bias=bias,
            threshold=threshold,
            simulation_steps=simulation_steps,
            zero_threshold=zero_threshold,
        )
    return lif_bias_can_fire(
        bias=bias,
        threshold=threshold,
        simulation_steps=simulation_steps,
        zero_threshold=zero_threshold,
        mode=bias_only_emission_check,
    )


def ttfs_continuous_bias_can_activate(
    *,
    bias,
    zero_threshold: float = 1e-8,
) -> bool:
    """Continuous TTFS: ``relu(V)/θ`` is positive iff ``V > 0`` (bias-only path)."""
    arr = _normalized_bias_array(bias)
    if arr is None:
        return False
    return float(np.max(arr)) > zero_threshold


def ttfs_quantized_bias_can_activate(
    *,
    bias,
    threshold: float,
    simulation_steps: int,
    zero_threshold: float = 1e-8,
) -> bool:
    """Quantized TTFS: neuron fires iff closed-form activation at ``S`` is > 0."""
    from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation_np

    arr = _normalized_bias_array(bias)
    if arr is None:
        return False
    s = max(int(simulation_steps), 1)
    th = np.maximum(np.asarray(threshold, dtype=np.float64), 1e-12)
    if th.ndim == 0:
        th = np.full(arr.shape, float(th))
    elif th.size != arr.size:
        th = np.broadcast_to(th, arr.shape)
    mask = np.abs(arr) >= zero_threshold
    if not np.any(mask):
        return False
    v = arr[mask]
    th_m = th[mask]
    act = ttfs_quantized_activation_np(v, th_m, s)
    return bool(np.any(act > 0))


def lif_bias_can_fire(
    *,
    bias,
    threshold: float,
    simulation_steps: int,
    zero_threshold: float = 1e-8,
    mode: str = "conservative",
) -> bool:
    """LIF / rate: conservative integration bound ``max(|b|)*T >= θ``."""
    arr = _normalized_bias_array(bias)
    if arr is None:
        return False
    abs_bias = np.abs(arr)
    if not np.any(abs_bias >= zero_threshold):
        return False
    if mode == "exact":
        pass
    return float(abs_bias.max()) * float(simulation_steps) >= float(threshold)


def _normalized_bias_array(bias) -> np.ndarray | None:
    if bias is None:
        return None
    arr = np.asarray(bias, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr.ravel() if arr.ndim > 1 else arr


def is_ttfs_mode(spiking_mode: str) -> bool:
    return str(spiking_mode or "lif").lower() in TTFS_MODES
