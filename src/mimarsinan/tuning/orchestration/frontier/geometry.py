"""The frontier geometry SSOT: the rate↔position mapping and the unit ladder."""

from __future__ import annotations

import math

_LADDER_RATE_EPSILON = 1e-9
"""Absorbs IEEE upward rounding of ``(i/n) * n`` so ladder rates pin exactly."""


def frontier_position(rate: float, n_units: int) -> int:
    """Installed frontier position at ``rate``: ``ceil(rate * n)``, clamped to [0, n].

    Ladder rates ``i/n`` map exactly to position ``i``. CEILING is load-bearing:
    a gate midpoint retry ``(committed + rate)/2`` must retrain the TARGET
    frontier from the restored snapshot — a frontier cannot bisect below the
    unit being converted.
    """
    n = max(0, int(n_units))
    k = math.ceil(float(rate) * n - _LADDER_RATE_EPSILON)
    return max(0, min(n, k))


def frontier_ladder(n_units: int) -> list:
    """The fast-ladder rates walking the frontier one unit per rung: ``[i/n]``."""
    n = max(1, int(n_units))
    return [i / n for i in range(1, n + 1)]
