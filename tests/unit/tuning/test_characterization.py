"""Characterization phase: monotonicity verdict + slope→epsilon (P5c)."""

import pytest

from mimarsinan.tuning.orchestration.characterization import characterize, Profile

GRID = [i / 10 for i in range(11)]  # 0.0 .. 1.0


def test_smooth_monotone_curve_is_monotonic_with_large_epsilon():
    prof = characterize(lambda a: 0.1 * a, GRID, budget=0.02)
    assert isinstance(prof, Profile)
    assert prof.monotonic is True
    # gentle slope (0.1) → epsilon hint at the cap
    assert prof.epsilon_hint == pytest.approx(2 ** -4)


def test_cliff_curve_yields_small_epsilon_hint():
    # near-vertical jump at 0.5 → large slope → small epsilon hint
    prof = characterize(lambda a: 0.0 if a < 0.5 else 1.0, GRID, budget=0.02)
    assert prof.max_slope > 5.0
    assert prof.epsilon_hint < 2 ** -4
    assert prof.epsilon_hint >= 2 ** -8


def test_non_monotone_curve_is_flagged():
    # drop dips back down in the middle (a re-aligning quant grid)
    curve = {0.0: 0.0, 0.3: 0.05, 0.5: 0.01, 0.7: 0.06, 1.0: 0.1}
    prof = characterize(lambda a: curve.get(round(a, 1), 0.0), GRID, budget=0.02)
    assert prof.monotonic is False  # 0.05 → 0.01 dip exceeds the budget


def test_feasible_max_tracks_budget():
    # drop == alpha; budget 0.35 → feasible through 0.3, not 0.4
    prof = characterize(lambda a: a, GRID, budget=0.35)
    assert prof.feasible_max == pytest.approx(0.3)
