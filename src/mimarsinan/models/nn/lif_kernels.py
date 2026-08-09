"""Shared LIF integrate-and-fire step (unified and hybrid core flows)."""

from __future__ import annotations

import torch

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


def snap_membrane_to_lattice(memb: torch.Tensor, lattice_scale: float) -> None:
    """Project the membrane onto its exact arithmetic lattice, in place.

    On the integer chip every true membrane value is a multiple of the
    lattice quantum (half a chip unit covers the half-step init); float
    summation-order noise (~1e-7) pushed values ACROSS threshold ties —
    which the integer-theta lattice makes common (the 2026-08-09 NF↔SCM
    13/788 catch: a true tie fired on one twin and not the other). This is
    an EXACT projection, not a tolerance: noise ≪ half a quantum."""
    memb.mul_(lattice_scale).round_().div_(lattice_scale)


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
