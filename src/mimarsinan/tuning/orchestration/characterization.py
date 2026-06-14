"""Characterization phase (spec §10 / report V9): profile an axis before search.

Sweeps the paired drop on a grid of α and derives a ``Profile``: a monotonicity
verdict (A1 — a significant non-monotonicity downgrades the controller to dense-
grid safe mode), the maximum local slope (A3 — a near-vertical cliff calls for a
smaller ``epsilon``), and the highest α that stayed within budget. The profile
configures the controller per axis instead of trusting global assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Profile:
    """Characterization output, archived with the run for reproducibility."""

    monotonic: bool
    max_slope: float
    epsilon_hint: float
    feasible_max: float
    drops: list


def characterize(drop_fn, grid, *, budget=0.02, epsilon_floor=2 ** -8, epsilon_cap=2 ** -4):
    """Profile ``drop_fn(alpha) -> drop`` over an increasing ``grid`` of α.

    ``monotonic`` is False if any later grid point's drop falls below an earlier
    one's by more than the noise budget (A1). ``epsilon_hint`` shrinks inversely
    with the steepest local slope so bisection never steps over a cliff (A3).
    ``feasible_max`` is the highest α whose drop stayed within ``budget``.
    """
    grid = [float(a) for a in grid]
    drops = [float(drop_fn(a)) for a in grid]

    monotonic = all(
        drops[i + 1] >= drops[i] - budget for i in range(len(drops) - 1)
    )

    slopes = [
        abs(drops[i + 1] - drops[i]) / max(grid[i + 1] - grid[i], 1e-12)
        for i in range(len(grid) - 1)
    ]
    max_slope = max(slopes) if slopes else 0.0

    # Steeper cliff → smaller admissible increment, clamped to [floor, cap].
    raw = budget / max_slope if max_slope > 0 else epsilon_cap
    epsilon_hint = min(max(raw, epsilon_floor), epsilon_cap)

    feasible = [a for a, d in zip(grid, drops) if d <= budget]
    feasible_max = max(feasible) if feasible else 0.0

    return Profile(
        monotonic=monotonic,
        max_slope=max_slope,
        epsilon_hint=epsilon_hint,
        feasible_max=feasible_max,
        drops=drops,
    )
