"""Greedy cascaded TTFS fire-once step (single-spike, hardware-faithful)."""

from __future__ import annotations

import torch

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


def ttfs_cycle_fire_once(
    memb: torch.Tensor,
    threshold: torch.Tensor,
    has_fired: torch.Tensor,
    *,
    thresholding_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Emit a single spike on the cycle ``memb`` first crosses ``threshold``;
    ``has_fired`` is the persistent per-neuron latch updated in place. No membrane
    reset — later input is greedily ignored."""
    crossed = _THRESHOLD_OPS[thresholding_mode](threshold, memb)
    newly_fired = crossed & (~has_fired)
    has_fired |= crossed
    if output_dtype is not None:
        return newly_fired.to(output_dtype)
    return newly_fired.float()
