"""Shared TTFS-cycle fire-once-latch step (genuine binary-spike TTFS).

Counterpart to ``lif_kernels.lif_fire_and_reset``: where LIF fires repeatedly and
resets, a TTFS-cycle neuron fires at most once and **latches** its output high for
the rest of the window. There is no membrane reset.
"""

from __future__ import annotations

import torch

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


def ttfs_cycle_fire_and_latch(
    memb: torch.Tensor,
    threshold: torch.Tensor,
    has_fired: torch.Tensor,
    *,
    thresholding_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Latch ``has_fired`` once ``memb`` crosses ``threshold``; return latched spikes.

    ``has_fired`` (bool, same shape as ``memb``) is updated in place and is the
    persistent per-neuron state across cycles. The returned tensor is the latched
    output: high from the firing cycle onward. No reset is applied to ``memb``.
    """
    crossed = _THRESHOLD_OPS[thresholding_mode](threshold, memb)
    has_fired |= crossed
    if output_dtype is not None:
        return has_fired.to(output_dtype)
    return has_fired.float()
