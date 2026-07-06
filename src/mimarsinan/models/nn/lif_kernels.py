"""Shared LIF integrate-and-fire step (unified and hybrid core flows)."""

from __future__ import annotations

import torch

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


def lif_fire_and_reset(
    memb: torch.Tensor,
    threshold: torch.Tensor,
    *,
    thresholding_mode: str,
    firing_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Threshold ``memb``, return spike tensor, apply Novena/Default reset in-place.

    Masked arithmetic instead of boolean fancy indexing: ``memb[fired]`` lowers
    through ``nonzero`` and forces a host-device sync per call (the dominant
    cycle-loop wall cost); ``fired * threshold`` is bit-exact for finite
    thresholds since ``fired`` is exactly 0 or 1.
    """
    fired = _THRESHOLD_OPS[thresholding_mode](threshold, memb)
    if firing_mode == "Novena":
        memb.masked_fill_(fired, 0.0)
    elif firing_mode == "Default":
        fired_typed = fired.to(memb.dtype)
        memb.sub_(fired_typed * threshold)
        # The reset conversion doubles as the output when dtypes agree (the
        # hybrid cycle loop's hot path) — same values, one fewer kernel.
        if output_dtype == memb.dtype or (output_dtype is None and memb.dtype == torch.float32):
            return fired_typed
    if output_dtype is not None:
        return fired.to(output_dtype)
    return fired.float()
