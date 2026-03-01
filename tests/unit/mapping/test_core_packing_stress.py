"""
Stress tests for greedy_pack_softcores algorithm.

Tests degenerate inputs, single-element cores, and boundary conditions.
"""

import pytest

from mimarsinan.mapping.core_packing import (
    greedy_pack_softcores,
    pick_best_softcore,
    _placement_waste,
    _remaining_capacity,
)


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


def _is_mapping_possible(soft, hard):
    return (soft.get_input_count() <= hard.available_axons
            and soft.get_output_count() <= hard.available_neurons)


def _place(idx, hard, soft):
    hard.available_axons -= soft.get_input_count()
    hard.available_neurons -= soft.get_output_count()


class TestPickBestSoftcoreStress:
    def test_single_axon_single_neuron(self):
        """Minimal core dimensions."""
        cores = [FakeSoftCore(1, 1)]
        assert pick_best_softcore(cores) is cores[0]

    def test_highly_asymmetric_cores(self):
        """One very tall, one very wide — should pick the most extreme."""
        tall = FakeSoftCore(1, 1000)
        wide = FakeSoftCore(1000, 1)
        square = FakeSoftCore(31, 31)
        best = pick_best_softcore([tall, wide, square])
        assert best in (tall, wide), \
            "Should prefer the most asymmetric core"

    def test_zero_dimension_core(self):
        """Core with 0 axons or 0 neurons."""
        zero_a = FakeSoftCore(0, 10)
        normal = FakeSoftCore(5, 5)
        # This should still work (though degenerate)
        result = pick_best_softcore([zero_a, normal])
        assert result is not None


class TestPlacementWasteStress:
    def test_waste_is_zero_for_identical(self):
        s = FakeSoftCore(10, 10)
        h = FakeHardCore(10, 10)
        assert _placement_waste(s, h) == 0

    def test_waste_formula_hand_computed(self):
        """
        waste = h_a * s_n + s_a * h_n - 2 * s_a * s_n
        s = (3, 2), h = (8, 4)
        waste = 8*2 + 3*4 - 2*3*2 = 16 + 12 - 12 = 16
        """
        s = FakeSoftCore(3, 2)
        h = FakeHardCore(8, 4)
        assert _placement_waste(s, h) == 16

    def test_waste_for_single_element(self):
        """s = (1, 1), h = (N, M) → waste = N + M - 2."""
        s = FakeSoftCore(1, 1)
        h = FakeHardCore(256, 256)
        expected = 256 + 256 - 2
        assert _placement_waste(s, h) == expected

    def test_remaining_capacity_hand_computed(self):
        """
        s = (4, 2), h available = (8, 4)
        remaining = (8-4) * (4-2) = 4 * 2 = 8
        """
        s = FakeSoftCore(4, 2)
        h = FakeHardCore(8, 4)
        assert _remaining_capacity(s, h) == 8


class TestGreedyPackStress:
    def test_exact_fit_packing(self):
        """Cores exactly fill available hardware."""
        softs = [FakeSoftCore(4, 4), FakeSoftCore(4, 4)]
        unused = [FakeHardCore(4, 8)]  # Each HC can hold one softcore (4 axons, 4 of 8 neurons)
        used = []

        # After packing first, HC has 4 axons used, 4 neurons remaining (but 0 axons available)
        # Actually, our _place subtracts axons and neurons
        # First pack: HC goes from (4,8) to (0,4) — second core needs 4 axons but 0 available
        # So this should need 2 hardware cores
        unused2 = [FakeHardCore(4, 4), FakeHardCore(4, 4)]
        greedy_pack_softcores(
            softcores=softs, used_hardcores=used,
            unused_hardcores=unused2,
            is_mapping_possible=_is_mapping_possible,
            place=_place,
        )
        assert len(softs) == 0
        assert len(used) == 2

    def test_many_small_cores_into_large_hardware(self):
        """100 small softcores packed into large hardware."""
        softs = [FakeSoftCore(1, 1) for _ in range(100)]
        unused = [FakeHardCore(100, 100)]
        used = []

        greedy_pack_softcores(
            softcores=softs, used_hardcores=used,
            unused_hardcores=unused,
            is_mapping_possible=_is_mapping_possible,
            place=_place,
        )
        assert len(softs) == 0

    def test_packing_modifies_softcores_list_inplace(self):
        """Verify the function modifies the caller's list (side effect)."""
        original = [FakeSoftCore(4, 4)]
        softs = original  # Same reference
        unused = [FakeHardCore(8, 8)]
        used = []

        greedy_pack_softcores(
            softcores=softs, used_hardcores=used,
            unused_hardcores=unused,
            is_mapping_possible=_is_mapping_possible,
            place=_place,
        )
        # The caller's list is mutated
        assert len(original) == 0
        assert len(softs) == 0

    def test_softcore_larger_than_any_hardware(self):
        """A softcore that doesn't fit in any hardware core."""
        softs = [FakeSoftCore(500, 500)]
        unused = [FakeHardCore(256, 256)]
        used = []

        with pytest.raises(RuntimeError, match="does not fit"):
            greedy_pack_softcores(
                softcores=softs, used_hardcores=used,
                unused_hardcores=unused,
                is_mapping_possible=_is_mapping_possible,
                place=_place,
            )

    def test_empty_unused_and_used_raises(self):
        """No hardware at all."""
        softs = [FakeSoftCore(4, 4)]
        with pytest.raises(RuntimeError):
            greedy_pack_softcores(
                softcores=softs, used_hardcores=[],
                unused_hardcores=[],
                is_mapping_possible=_is_mapping_possible,
                place=_place,
            )
