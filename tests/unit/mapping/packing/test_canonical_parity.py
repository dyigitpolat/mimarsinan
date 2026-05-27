"""Layout and runtime packers share canonical feasibility predicates."""

from __future__ import annotations

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreInstance, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.canonical import canonical_is_mapping_possible
from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores


class _RuntimeSoft:
    def __init__(self, axons: int, neurons: int):
        self._a, self._n = axons, neurons

    def get_input_count(self):
        return self._a

    def get_output_count(self):
        return self._n


class _RuntimeHard:
    def __init__(self, axons: int, neurons: int):
        self.axons_per_core = axons
        self.neurons_per_core = neurons
        self.available_axons = axons
        self.available_neurons = neurons

    def get_input_count(self):
        return self.available_axons

    def get_output_count(self):
        return self.available_neurons


def test_canonical_feasibility_matches_layout_and_runtime_types():
    soft_layout = LayoutSoftCoreSpec(input_count=4, output_count=8, threshold_group_id=0)
    hard_layout = LayoutHardCoreInstance(axons_per_core=8, neurons_per_core=16)
    soft_runtime = _RuntimeSoft(4, 8)
    hard_runtime = _RuntimeHard(8, 16)

    assert canonical_is_mapping_possible(soft_layout, hard_layout)
    assert canonical_is_mapping_possible(soft_runtime, hard_runtime)

    tight = LayoutHardCoreInstance(axons_per_core=2, neurons_per_core=8)
    assert not canonical_is_mapping_possible(soft_layout, tight)
    assert not canonical_is_mapping_possible(soft_runtime, _RuntimeHard(2, 8))


def test_greedy_pack_places_single_softcore():
    soft = _RuntimeSoft(2, 2)
    hard = _RuntimeHard(4, 4)
    used: list = []
    unused = [hard]
    placed: list = []

    def place(idx, hc, sc):
        placed.append((idx, sc))

    greedy_pack_softcores(
        softcores=[soft],
        used_hardcores=used,
        unused_hardcores=unused,
        is_mapping_possible=canonical_is_mapping_possible,
        place=place,
    )
    assert len(placed) == 1
    assert len(used) == 1
