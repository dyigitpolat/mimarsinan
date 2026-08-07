"""Bank affinity is the packer's STATED preference, replacing an accidental one.

Before this, placement was ranked by geometry alone (`_placement_waste` / `_remaining_capacity`)
while `weight_programming` measured programming cost. Nothing connected them: `residency_class_id`
happened to equal the weight-bank id, so bank affinity was preserved BY ACCIDENT. Stating the cost
makes the affinity deliberate, and lets the residency key stop carrying that job.
"""

import pytest

from mimarsinan.mapping.packing.placement_cost import (
    placement_cost,
    bank_affinity_cost,
    note_resident_bank,
    resident_bank_ids,
)


class _Soft:
    def __init__(self, axons=4, neurons=4, bank=None):
        self._a, self._n = axons, neurons
        self.weight_bank_id = bank

    def get_input_count(self):
        return self._a

    def get_output_count(self):
        return self._n


class _Hard:
    pass


class TestBankAffinityCost:
    def test_a_resident_bank_costs_nothing_extra(self):
        hard = _Hard()
        note_resident_bank(hard, _Soft(bank=3))
        assert bank_affinity_cost(_Soft(bank=3), hard) == 0

    def test_a_new_bank_costs_its_area(self):
        hard = _Hard()
        note_resident_bank(hard, _Soft(bank=3))
        assert bank_affinity_cost(_Soft(axons=4, neurons=4, bank=9), hard) == 16

    def test_an_owned_matrix_always_costs(self):
        """Owned weights are programmed per placement; there is nothing to reuse."""
        hard = _Hard()
        note_resident_bank(hard, _Soft(bank=None))
        assert resident_bank_ids(hard) == frozenset()
        assert bank_affinity_cost(_Soft(axons=2, neurons=3, bank=None), hard) == 6

    def test_an_empty_core_has_nothing_resident(self):
        assert bank_affinity_cost(_Soft(bank=1), _Hard()) == 4 * 4


class TestTheCostOrdersProgrammingBeforeFit:
    def test_reuse_beats_a_tighter_fit(self):
        """The ordering that preserves reuse once residency classes stop matching banks."""
        reusing, fresh = _Hard(), _Hard()
        note_resident_bank(reusing, _Soft(bank=5))
        core = _Soft(bank=5)
        cost_reuse = placement_cost(core, reusing, remaining_capacity=900)
        cost_fresh = placement_cost(core, fresh, remaining_capacity=1)
        assert cost_reuse < cost_fresh, "a loose fit that reprograms nothing must win"

    def test_fit_still_decides_between_equal_programming(self):
        a, b = _Hard(), _Hard()
        for h in (a, b):
            note_resident_bank(h, _Soft(bank=2))
        core = _Soft(bank=2)
        assert placement_cost(core, a, remaining_capacity=10) < placement_cost(
            core, b, remaining_capacity=99
        )

    def test_the_cost_is_a_plain_comparable_tuple(self):
        """Extra terms (routing, energy) extend it without the packer learning about them."""
        cost = placement_cost(_Soft(bank=1), _Hard(), remaining_capacity=7)
        assert isinstance(cost, tuple) and len(cost) == 2
        assert cost[1] == 7


class TestHardCoreRecordsWhatItHosts:
    def test_add_softcore_notes_the_bank(self):
        import numpy as np
        import torch

        from mimarsinan.mapping.packing.softcore.hard_core import HardCore
        from mimarsinan.mapping.packing.softcore.soft_core import SoftCore

        hc = HardCore(8, 8)
        sc = SoftCore(core_matrix=np.zeros((4, 4)), axon_sources=[None] * 4, id=0)
        sc.threshold = 1.0
        sc.activation_scale = torch.tensor(1.0)
        sc.parameter_scale = torch.tensor(1.0)
        sc.input_activation_scale = torch.tensor(1.0)
        sc.weight_bank_id = 11
        hc.add_softcore(sc)
        assert 11 in resident_bank_ids(hc)
        assert bank_affinity_cost(sc, hc) == 0
