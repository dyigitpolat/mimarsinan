"""Typed feasibility constraints: an infeasible candidate is scored, never a crash.

A deployment constraint (the on-chip parameter floor, say) shapes the FEASIBLE REGION
of the search. Discovering one as a pipeline exception after the search picked a
winner is the wrong seam — the optimizer already has a constraint channel, so a
violation belongs there, typed, with the measurement that explains it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

#: The on-chip parameter floor (``onchip_min_fraction``): below it, a mapping is not
#: a genuine on-chip deployment.
ONCHIP_FLOOR_CONSTRAINT = "onchip_min_fraction"


@dataclass(frozen=True)
class ConstraintReport:
    """One violated constraint: which, what was measured, and against what limit."""

    constraint: str
    measured: float
    limit: float
    detail: str

    @property
    def violation(self) -> float:
        """How far past the limit — a gradient the optimizer can descend."""
        return max(0.0, self.limit - self.measured)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint": self.constraint,
            "measured": float(self.measured),
            "limit": float(self.limit),
            "violation": float(self.violation),
            "detail": self.detail,
        }


def onchip_floor_violation(
    *, fraction: float, floor: float
) -> Optional[ConstraintReport]:
    """The on-chip floor report for a candidate, or ``None`` when it is satisfied.

    A floor of zero is no constraint at all (the operator disabled it), not a
    zero-width one every candidate trivially satisfies.
    """
    if floor <= 0.0 or fraction >= floor:
        return None
    return ConstraintReport(
        constraint=ONCHIP_FLOOR_CONSTRAINT,
        measured=float(fraction),
        limit=float(floor),
        detail=(
            f"only {fraction:.2%} of the deployed parameters would sit on chip "
            f"cores, below the required {floor:.0%} floor — this candidate is not "
            f"a genuine on-chip deployment"
        ),
    )


__all__ = [
    "ONCHIP_FLOOR_CONSTRAINT",
    "ConstraintReport",
    "onchip_floor_violation",
]
