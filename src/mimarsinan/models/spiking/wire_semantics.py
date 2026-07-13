"""Wire-op kernel pairs: each TTFS wire op defined once with parity-locked torch+numpy twins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


COMPARATOR_HALF_STEP_LADDER_SHIFT = 0.5
"""[E3] comparator-side half-step: every per-cycle compare level θ·(S-k)/S drops
by θ/(2S) — ``ceil(S·(1-v/θ) - 1/2)``, the exact zero-bit-cost twin of the
+θ/(2S) bias fold, carried in the threshold ladder (never a θ rescale)."""


def ttfs_quantized_staircase_np(
    v: np.ndarray,
    threshold,
    simulation_length: int,
    *,
    comparator_half_step: bool = False,
) -> np.ndarray:
    """``(S - clamp(ceil(S*(1-V/θ)), 0, S-1)) / S`` with fire mask (numpy)."""
    s = int(simulation_length)
    safe = np.maximum(np.asarray(threshold, dtype=np.float64), 1e-12)
    ladder = s * (1.0 - v / safe)
    if comparator_half_step:
        ladder = ladder - COMPARATOR_HALF_STEP_LADDER_SHIFT
    k_fire_raw = np.ceil(ladder)
    fires = k_fire_raw < s
    k_fire = np.clip(k_fire_raw, 0, s - 1)
    out = np.where(fires, (s - k_fire) / s, 0.0)
    return out.astype(np.float64, copy=False)


def ttfs_quantized_staircase(
    V: torch.Tensor,
    threshold: torch.Tensor,
    simulation_length: int,
    *,
    comparator_half_step: bool = False,
) -> torch.Tensor:
    """``(S - clamp(ceil(S*(1-V/θ)), 0, S-1)) / S`` with fire mask (torch)."""
    S = simulation_length
    safe_thresh = threshold.clamp(min=1e-12)
    ladder = S * (1.0 - V / safe_thresh)
    if comparator_half_step:
        ladder = ladder - COMPARATOR_HALF_STEP_LADDER_SHIFT
    k_fire_raw = torch.ceil(ladder)
    fires = k_fire_raw < S
    k_fire = k_fire_raw.clamp(0, S - 1)
    return torch.where(fires, (S - k_fire) / S, torch.zeros_like(k_fire))


def _ttfs_strict_staircase_np(
    v, threshold, simulation_length: int, *, comparator_half_step: bool = False,
) -> np.ndarray:
    """Strict-compare (``<``) variant: exact grid ties fire one cycle later."""
    s = int(simulation_length)
    safe = np.maximum(np.asarray(threshold, dtype=np.float64), 1e-12)
    ladder = s * (1.0 - v / safe)
    if comparator_half_step:
        ladder = ladder - COMPARATOR_HALF_STEP_LADDER_SHIFT
    k_fire_raw = np.floor(ladder) + 1.0
    fires = k_fire_raw < s
    k_fire = np.clip(k_fire_raw, 0, s - 1)
    out = np.where(fires, (s - k_fire) / s, 0.0)
    return out.astype(np.float64, copy=False)


def _ttfs_strict_staircase(
    V, threshold, simulation_length: int, *, comparator_half_step: bool = False,
) -> torch.Tensor:
    S = simulation_length
    safe_thresh = threshold.clamp(min=1e-12)
    ladder = S * (1.0 - V / safe_thresh)
    if comparator_half_step:
        ladder = ladder - COMPARATOR_HALF_STEP_LADDER_SHIFT
    k_fire_raw = torch.floor(ladder) + 1.0
    fires = k_fire_raw < S
    k_fire = k_fire_raw.clamp(0, S - 1)
    return torch.where(fires, (S - k_fire) / S, torch.zeros_like(k_fire))


_LIF_COMPARE_MODES = ("<", "<=")


def _require_lif_compare_mode(compare_mode: str) -> None:
    if compare_mode not in _LIF_COMPARE_MODES:
        raise ValueError(
            f"lif_count_staircase compare_mode must be '<' or '<='; "
            f"got {compare_mode!r}"
        )


def lif_count_staircase_np(
    z: np.ndarray, threshold, window: int, *, compare_mode: str = "<=",
) -> np.ndarray:
    """LIF window-count commutation staircase ``θ·clamp(F(T·z/θ), 0, T)/T``
    (numpy); ``F = floor`` for inclusive ``'<='``, ``ceil − 1`` for strict
    ``'<'`` — exact-integer charge fires one fewer (Theorems 2/A2)."""
    _require_lif_compare_mode(compare_mode)
    t = int(window)
    safe = np.maximum(np.asarray(threshold, dtype=np.float64), 1e-12)
    r = np.asarray(z, dtype=np.float64) / safe
    c = np.ceil(t * r) - 1.0 if compare_mode == "<" else np.floor(t * r)
    c = np.clip(c, 0.0, float(t))
    return (safe * c / t).astype(np.float64, copy=False)


def lif_count_staircase(
    z: torch.Tensor, threshold: torch.Tensor, window: int, *, compare_mode: str = "<=",
) -> torch.Tensor:
    """LIF window-count commutation staircase ``θ·clamp(F(T·z/θ), 0, T)/T`` (torch twin)."""
    _require_lif_compare_mode(compare_mode)
    t = int(window)
    safe = threshold.clamp(min=1e-12)
    r = z / safe
    c = torch.ceil(t * r) - 1.0 if compare_mode == "<" else torch.floor(t * r)
    return safe * c.clamp(0.0, float(t)) / t


def floor_staircase(x: torch.Tensor, levels) -> torch.Tensor:
    return torch.floor(x * levels) / levels


def floor_staircase_np(x: np.ndarray, levels) -> np.ndarray:
    return np.floor(x * levels) / levels


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
    """Wire-op bundle for one deployment: ``(S, compare_mode, comparator_half_step)``.

    ``compare_mode`` mirrors nevresim Compare: ``"<="`` inclusive (parity default)
    vs ``"<"`` strict (exact grid ties fire one cycle later).
    ``comparator_half_step`` carries the [E3] mid-tread offset in the compare
    ladder instead of the bias lattice; both parity twins shift together.
    """

    simulation_steps: int
    compare_mode: str = "<="
    comparator_half_step: bool = False

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
            return _ttfs_strict_staircase(
                V, threshold, self.simulation_steps,
                comparator_half_step=self.comparator_half_step,
            )
        return ttfs_quantized_staircase(
            V, threshold, self.simulation_steps,
            comparator_half_step=self.comparator_half_step,
        )

    def quantized_staircase_np(self, v: np.ndarray, threshold) -> np.ndarray:
        if self.compare_mode == "<":
            return _ttfs_strict_staircase_np(
                v, threshold, self.simulation_steps,
                comparator_half_step=self.comparator_half_step,
            )
        return ttfs_quantized_staircase_np(
            v, threshold, self.simulation_steps,
            comparator_half_step=self.comparator_half_step,
        )

    def spike_time(self, rates: torch.Tensor) -> torch.Tensor:
        return ttfs_spike_time(rates, self.simulation_steps)

    def spike_time_np(self, rates: np.ndarray) -> np.ndarray:
        return ttfs_spike_time_np(rates, self.simulation_steps)

    def grid_quantize(self, rates: torch.Tensor) -> torch.Tensor:
        return ttfs_grid_quantize(rates, self.simulation_steps)

    def grid_quantize_np(self, rates: np.ndarray) -> np.ndarray:
        return ttfs_grid_quantize_np(rates, self.simulation_steps)
