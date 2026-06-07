"""Wire-op kernel pairs: each TTFS wire op defined once, with torch+numpy twins.

Every pair keeps the exact same operation order in both frameworks so the twins
agree bit-for-bit in float64 (cross-twin tests sweep ±1 ULP around the S-grid).
C++ backends (nevresim, SANA-FE somas) cannot share code; they stay parity-locked
against these twins via the existing integration harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ── D1: quantized TTFS staircase (deployment ceil form, inclusive compare) ──
def ttfs_quantized_staircase_np(
    v: np.ndarray,
    threshold,
    simulation_length: int,
) -> np.ndarray:
    """``(S - clamp(ceil(S*(1-V/θ)), 0, S-1)) / S`` with fire mask (numpy)."""
    s = int(simulation_length)
    safe = np.maximum(np.asarray(threshold, dtype=np.float64), 1e-12)
    k_fire_raw = np.ceil(s * (1.0 - v / safe))
    fires = k_fire_raw < s
    k_fire = np.clip(k_fire_raw, 0, s - 1)
    out = np.where(fires, (s - k_fire) / s, 0.0)
    return out.astype(np.float64, copy=False)


def ttfs_quantized_staircase(
    V: torch.Tensor,
    threshold: torch.Tensor,
    simulation_length: int,
) -> torch.Tensor:
    """``(S - clamp(ceil(S*(1-V/θ)), 0, S-1)) / S`` with fire mask (torch)."""
    S = simulation_length
    safe_thresh = threshold.clamp(min=1e-12)
    k_fire_raw = torch.ceil(S * (1.0 - V / safe_thresh))
    fires = k_fire_raw < S
    k_fire = k_fire_raw.clamp(0, S - 1)
    return torch.where(fires, (S - k_fire) / S, torch.zeros_like(k_fire))


def _ttfs_strict_staircase_np(v, threshold, simulation_length: int) -> np.ndarray:
    """Strict-compare (``<``) variant: exact grid ties fire one cycle later."""
    s = int(simulation_length)
    safe = np.maximum(np.asarray(threshold, dtype=np.float64), 1e-12)
    k_fire_raw = np.floor(s * (1.0 - v / safe)) + 1.0
    fires = k_fire_raw < s
    k_fire = np.clip(k_fire_raw, 0, s - 1)
    out = np.where(fires, (s - k_fire) / s, 0.0)
    return out.astype(np.float64, copy=False)


def _ttfs_strict_staircase(V, threshold, simulation_length: int) -> torch.Tensor:
    S = simulation_length
    safe_thresh = threshold.clamp(min=1e-12)
    k_fire_raw = torch.floor(S * (1.0 - V / safe_thresh)) + 1.0
    fires = k_fire_raw < S
    k_fire = k_fire_raw.clamp(0, S - 1)
    return torch.where(fires, (S - k_fire) / S, torch.zeros_like(k_fire))


# ── Generic floor staircase (act-quant; supports non-integer levels) ────────
def floor_staircase(x: torch.Tensor, levels) -> torch.Tensor:
    return torch.floor(x * levels) / levels


def floor_staircase_np(x: np.ndarray, levels) -> np.ndarray:
    return np.floor(x * levels) / levels


# ── D2: value→spike-time encode and the grid-snap round trip ────────────────
def ttfs_spike_time_np(rate: np.ndarray, simulation_length: int) -> np.ndarray:
    """Per-element spike time ``round(S * (1 - clamp(rate, 0, 1)))`` (numpy)."""
    s = int(simulation_length)
    clamped = np.clip(rate, 0.0, 1.0)
    return np.rint(s * (1.0 - clamped)).astype(np.int64)


def ttfs_spike_time(rate: torch.Tensor, simulation_length: int) -> torch.Tensor:
    """Per-element spike time ``round(S * (1 - clamp(rate, 0, 1)))`` (torch)."""
    s = simulation_length
    clamped = rate.clamp(0.0, 1.0)
    return torch.round(s * (1.0 - clamped))


def ttfs_grid_quantize_np(rates: np.ndarray, simulation_length: int) -> np.ndarray:
    """Encode→decode round trip ``(S - spike_time)/S``; ``k >= S`` never fires."""
    s = int(simulation_length)
    spike_times = ttfs_spike_time_np(np.asarray(rates, dtype=np.float64), s)
    return np.where(spike_times < s, (s - spike_times) / float(s), 0.0)


def ttfs_grid_quantize(rates: torch.Tensor, simulation_length: int) -> torch.Tensor:
    """Encode→decode round trip ``(S - spike_time)/S`` (torch twin)."""
    s = simulation_length
    spike_times = ttfs_spike_time(rates, s)
    return torch.where(
        spike_times < s, (s - spike_times) / float(s), torch.zeros_like(spike_times)
    )


@dataclass(frozen=True)
class WireSemantics:
    """Wire-op bundle for one deployment: ``(S, compare_mode)``.

    ``compare_mode`` mirrors nevresim's Compare policies: ``"<="`` inclusive
    (the parity-locked default) vs ``"<"`` strict (exact grid ties fire one
    cycle later). C++ ``"<"`` parity is not yet harness-covered; the Python
    twins define the intended tie semantics.
    """

    simulation_steps: int
    compare_mode: str = "<="

    def __post_init__(self):
        if self.compare_mode not in ("<", "<="):
            raise ValueError(
                f"WireSemantics compare_mode must be '<' or '<='; "
                f"got {self.compare_mode!r}"
            )

    def quantized_staircase(self, V: torch.Tensor, threshold) -> torch.Tensor:
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(float(threshold), dtype=V.dtype, device=V.device)
        if self.compare_mode == "<":
            return _ttfs_strict_staircase(V, threshold, self.simulation_steps)
        return ttfs_quantized_staircase(V, threshold, self.simulation_steps)

    def quantized_staircase_np(self, v: np.ndarray, threshold) -> np.ndarray:
        if self.compare_mode == "<":
            return _ttfs_strict_staircase_np(v, threshold, self.simulation_steps)
        return ttfs_quantized_staircase_np(v, threshold, self.simulation_steps)

    def spike_time(self, rates: torch.Tensor) -> torch.Tensor:
        return ttfs_spike_time(rates, self.simulation_steps)

    def spike_time_np(self, rates: np.ndarray) -> np.ndarray:
        return ttfs_spike_time_np(rates, self.simulation_steps)

    def grid_quantize(self, rates: torch.Tensor) -> torch.Tensor:
        return ttfs_grid_quantize(rates, self.simulation_steps)

    def grid_quantize_np(self, rates: np.ndarray) -> np.ndarray:
        return ttfs_grid_quantize_np(rates, self.simulation_steps)
