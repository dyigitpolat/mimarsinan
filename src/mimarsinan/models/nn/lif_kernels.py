"""Shared LIF integrate-and-fire step (unified and hybrid core flows)."""

from __future__ import annotations

import contextlib
import contextvars

import torch

_MEASUREMENT_PLANE = contextvars.ContextVar("lif_measurement_plane", default=False)


@contextlib.contextmanager
def measurement_plane():
    """The chip-lattice MEASUREMENT plane: inside, armed LIF nodes snap
    membranes to the integer-chip lattice so exact ties are decided by
    exact values (parity twins, certificates). Outside — training AND
    tuning-plane telemetry — the continuous membrane and the fused kernels
    are untouched: snapped telemetry deterministically steered keep-best /
    floor decisions onto different trajectories (t0_01 fresh 0.9771 vs
    0.9799, n8d 2026-08-10)."""
    token = _MEASUREMENT_PLANE.set(True)
    try:
        yield
    finally:
        _MEASUREMENT_PLANE.reset(token)


def in_measurement_plane() -> bool:
    return bool(_MEASUREMENT_PLANE.get())

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


class MembraneRailTouchedError(ValueError):
    """A membrane reached a rail of a register whose law says it cannot.

    ``membrane_arithmetic='saturating_signed'`` claims to hold exactly the
    number the unbounded accumulator holds, which is true only strictly inside
    the register's interval. The deployment carries a static no-saturation
    bound proving the rails are unreachable, so touching one is a broken proof,
    not this law's physics — refused everywhere (torch, nevresim, RTL), never
    clamped, because a clamp reports a number the contract denies.
    """


def enforce_membrane_rails(
    memb: torch.Tensor, membrane_bounds: tuple[float, float]
) -> None:
    """Refuse a membrane that reached either rail of its declared register.

    Armed only where the law asserts the rails are unreachable
    (``SomaLaw.asserts_no_saturation``); a register whose saturation IS the
    modelled substrate never calls this. Deliberately conservative: a value
    that merely EQUALS a rail without having been clamped there also refuses,
    because from the membrane alone the two are indistinguishable and the
    honest answer is to widen the register rather than to guess.
    """
    low, high = membrane_bounds
    if not bool(torch.any((memb <= low) | (memb >= high))):
        return
    raise MembraneRailTouchedError(
        f"the membrane reached a rail of its declared register "
        f"[{low}, {high}]: this law holds the same number the unbounded "
        f"accumulator holds only strictly inside the interval, and the "
        f"deployment's no-saturation bound said the rails were unreachable. "
        f"Refused, never clamped — a clamp would report a number the "
        f"deployment's own contract denies. Widen membrane_bits, lower the "
        f"fan-in, or raise theta."
    )


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
