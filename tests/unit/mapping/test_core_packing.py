"""Tests for the greedy_pack_softcores algorithm and helpers."""

import pytest

from mimarsinan.mapping.core_packing import (
    greedy_pack_softcores,
    pick_best_softcore,
    _placement_waste,
    _remaining_capacity,
)


# ---------------------------------------------------------------------------
# Minimal SoftCore/HardCore stubs that satisfy the Protocol
# ---------------------------------------------------------------------------

class FakeSoftCore:
    def __init__(self, axons, neurons):
        self._a = axons
        self._n = neurons

    def get_input_count(self):
        return self._a

    def get_output_count(self):
        return self._n


class FakeHardCore:
    def __init__(self, axons, neurons):
        self.axons_per_core = axons
        self.neurons_per_core = neurons
        self.available_axons = axons
        self.available_neurons = neurons

    def get_input_count(self):
        return self.axons_per_core

    def get_output_count(self):
        return self.neurons_per_core


# ---------------------------------------------------------------------------
# Tests: pick_best_softcore
# ---------------------------------------------------------------------------

class TestPickBestSoftcore:
    def test_single(self):
        cores = [FakeSoftCore(4, 8)]
        assert pick_best_softcore(cores) is cores[0]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pick_best_softcore([])

    def test_picks_most_extreme(self):
        a = FakeSoftCore(100, 2)
        b = FakeSoftCore(2, 100)
        c = FakeSoftCore(50, 50)
        assert pick_best_softcore([a, b, c]) in (a, b)

    def test_tie_breaking(self):
        a = FakeSoftCore(10, 10)
        b = FakeSoftCore(10, 10)
        result = pick_best_softcore([a, b])
        assert result is a or result is b


# ---------------------------------------------------------------------------
# Tests: placement_waste and remaining_capacity
# ---------------------------------------------------------------------------

class TestWasteAndCapacity:
    def test_perfect_fit_zero_waste(self):
        s = FakeSoftCore(8, 4)
        h = FakeHardCore(8, 4)
        assert _placement_waste(s, h) == 0

    def test_waste_increases_with_mismatch(self):
        s = FakeSoftCore(4, 4)
        h_good = FakeHardCore(4, 4)
        h_bad = FakeHardCore(64, 4)
        assert _placement_waste(s, h_good) < _placement_waste(s, h_bad)

    def test_remaining_capacity_perfect_fit(self):
        s = FakeSoftCore(8, 4)
        h = FakeHardCore(8, 4)
        assert _remaining_capacity(s, h) == 0

    def test_remaining_capacity_with_slack(self):
        s = FakeSoftCore(4, 2)
        h = FakeHardCore(8, 4)
        assert _remaining_capacity(s, h) == (8 - 4) * (4 - 2)


# ---------------------------------------------------------------------------
# Tests: greedy_pack_softcores
# ---------------------------------------------------------------------------

class TestGreedyPackSoftcores:
    def _is_mapping_possible(self, soft, hard):
        return (soft.get_input_count() <= hard.available_axons
                and soft.get_output_count() <= hard.available_neurons)

    def _place(self, idx, hard, soft):
        hard.available_axons -= soft.get_input_count()
        hard.available_neurons -= soft.get_output_count()

    def test_single_core_packing(self):
        softs = [FakeSoftCore(4, 4)]
        unused = [FakeHardCore(8, 8)]
        used = []
        greedy_pack_softcores(
            softcores=softs, used_hardcores=used,
            unused_hardcores=unused,
            is_mapping_possible=self._is_mapping_possible,
            place=self._place,
        )
        assert len(softs) == 0
        assert len(used) == 1

    def test_multiple_cores_packed(self):
        softs = [FakeSoftCore(4, 4), FakeSoftCore(2, 2)]
        unused = [FakeHardCore(8, 8), FakeHardCore(8, 8)]
        used = []
        greedy_pack_softcores(
            softcores=softs, used_hardcores=used,
            unused_hardcores=unused,
            is_mapping_possible=self._is_mapping_possible,
            place=self._place,
        )
        assert len(softs) == 0

    def test_no_fit_raises(self):
        softs = [FakeSoftCore(100, 100)]
        unused = [FakeHardCore(4, 4)]
        used = []
        with pytest.raises(RuntimeError, match="does not fit"):
            greedy_pack_softcores(
                softcores=softs, used_hardcores=used,
                unused_hardcores=unused,
                is_mapping_possible=self._is_mapping_possible,
                place=self._place,
            )

    def test_exhausted_cores_raises(self):
        softs = [FakeSoftCore(4, 4), FakeSoftCore(4, 4), FakeSoftCore(4, 4)]
        unused = [FakeHardCore(4, 4)]
        used = []
        with pytest.raises(RuntimeError):
            greedy_pack_softcores(
                softcores=softs, used_hardcores=used,
                unused_hardcores=unused,
                is_mapping_possible=self._is_mapping_possible,
                place=self._place,
            )

    def test_heterogeneous_core_types(self):
        """Scarcity-aware placement should preserve large cores for big softcores."""
        softs = [FakeSoftCore(16, 16), FakeSoftCore(4, 4), FakeSoftCore(4, 4)]
        unused = [
            FakeHardCore(8, 8), FakeHardCore(8, 8), FakeHardCore(8, 8),
            FakeHardCore(32, 32),
        ]
        used = []
        greedy_pack_softcores(
            softcores=softs, used_hardcores=used,
            unused_hardcores=unused,
            is_mapping_possible=self._is_mapping_possible,
            place=self._place,
        )
        assert len(softs) == 0

    def test_prefers_tight_used_cores(self):
        """When a used core has room, pack there instead of opening a new one."""
        s1 = FakeSoftCore(4, 4)
        s2 = FakeSoftCore(2, 2)
        unused = [FakeHardCore(8, 8), FakeHardCore(8, 8)]
        used = []

        greedy_pack_softcores(
            softcores=[s1], used_hardcores=used,
            unused_hardcores=unused,
            is_mapping_possible=self._is_mapping_possible,
            place=self._place,
        )
        assert len(used) == 1
        assert len(unused) == 1

        greedy_pack_softcores(
            softcores=[s2], used_hardcores=used,
            unused_hardcores=unused,
            is_mapping_possible=self._is_mapping_possible,
            place=self._place,
        )
        assert len(used) == 1
        assert len(unused) == 1
