"""The conversion-frontier geometry SSOT.

One rate→position mapping and one ladder for every monotone frontier walk
(P4 converted-prefix segments, [5v B1/B2] cascade hops): ladder rates ``i/n``
pin EXACTLY to position ``i``, ceiling is load-bearing for gate midpoint
retries, and the three walkers (hop staging, prefix ramp, the prefix forward)
share the same function rather than re-deriving it.
"""

from __future__ import annotations

import pytest

from mimarsinan.models.spiking.training.prefix_genuine_forward import (
    prefix_length_for_rate,
)
from mimarsinan.tuning.orchestration.adaptation_manager import hop_frontier
from mimarsinan.tuning.orchestration.frontier import (
    frontier_ladder,
    frontier_position,
)


class TestFrontierPosition:
    def test_ladder_rates_pin_exactly_to_their_position(self):
        for n in range(1, 129):
            for i in range(0, n + 1):
                assert frontier_position(i / n, n) == i, (i, n)

    def test_float_rounding_never_overshoots_a_ladder_rate(self):
        # (i/n) * n rounds ABOVE i for many (i, n) in IEEE double (e.g. 7/25);
        # the epsilon guard keeps the documented "i/n -> i" contract exact.
        for i, n in [(7, 25), (14, 25), (15, 29), (29, 35), (21, 38), (7, 41)]:
            assert frontier_position(i / n, n) == i

    def test_clamped_to_the_unit_range(self):
        assert frontier_position(-0.5, 9) == 0
        assert frontier_position(0.0, 9) == 0
        assert frontier_position(1.2, 9) == 9

    def test_gate_midpoint_retry_stays_on_the_target_unit(self):
        # A midpoint retry (committed + target)/2 between rungs i/n and
        # (i+1)/n must retrain the TARGET frontier, never bisect below it.
        for n in (4, 9, 25):
            for i in range(0, n):
                midpoint = ((i / n) + ((i + 1) / n)) / 2.0
                assert frontier_position(midpoint, n) == i + 1

    def test_zero_units_is_always_position_zero(self):
        assert frontier_position(0.7, 0) == 0


class TestFrontierLadder:
    def test_ladder_walks_one_unit_per_rung_and_ends_at_one(self):
        for n in (1, 5, 9, 25):
            rates = frontier_ladder(n)
            assert len(rates) == n
            assert rates[-1] == 1.0
            assert [frontier_position(r, n) for r in rates] == list(range(1, n + 1))

    def test_degenerate_unit_count_yields_the_single_full_rung(self):
        assert frontier_ladder(0) == [1.0]
        assert frontier_ladder(-3) == [1.0]


class TestWalkersShareTheGeometry:
    def test_hop_frontier_is_the_ssot_mapping(self):
        for n in (9, 25, 41):
            for i in range(0, n + 1):
                assert hop_frontier(i / n, n) == frontier_position(i / n, n)

    def test_prefix_length_is_the_ssot_mapping(self):
        for n in (3, 25):
            for i in range(0, n + 1):
                assert prefix_length_for_rate(i / n, n) == frontier_position(i / n, n)

    def test_hop_and_prefix_agree_everywhere(self):
        rates = [k / 97 for k in range(98)] + [0.001, 0.499, 0.501, 0.999]
        for n in (1, 6, 9, 25, 49):
            for rate in rates:
                assert hop_frontier(rate, n) == prefix_length_for_rate(rate, n)
