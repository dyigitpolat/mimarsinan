"""E85's priced plan (CL3): the pricer chooses the residency step whose period census prices the least write energy
plus carry energy per inference at the declared batch K (``x9_0_v3.price``'s argmin, ties to the first step, i.e.
the smaller r); the compute term is the same for every step of one program, so it never enters the choice."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute.step_choice import (
    StepChoiceError,
    carry_energy,
    cheapest_step,
    price_step,
)
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus


def _physics(**constants):
    return profile_from_dict({
        "format_version": 1, "name": "step_choice_test", "display_name": "Step choice test",
        "description_file": "t.md", "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "unit": "J", "evidence_kind": "estimated", "note": "test"}
                      for k, v in constants.items()},
    })


def _census(payloads, carry=()):
    passes = tuple(PassCensus(position=q, segment_index=0, resident=not b, cores=1, resident_cores=0 if b else 1,
                              written_cells_backing_store=b, written_cells_producer=0, payload_bytes=b,
                              compute_steps=0) for q, b in enumerate(payloads))
    return PassPeriodCensus(passes=passes, boundaries=tuple(PassBoundaryCensus(q, c, c) for q, c in carry))


TECH = _physics(e_program_per_byte=3.0, e_dma_per_byte=0.0, e_readout_per_byte=2.0)


class TestThePrice:
    def test_writes_and_carry_are_the_per_byte_constants_times_the_census(self):
        census = _census([100, 60], carry=[(0, 7)])
        price = price_step(census, TECH, WritePolicy(batch_k=16))
        assert price.write.nominal == float(Fraction(3) * 160 * 1000 / 16)
        assert price.carry.nominal == float(Fraction(2) * 7 * 1000)       # readout only: inject is priced at 0
        assert price.nominal == price.write.nominal + price.carry.nominal

    def test_a_resident_census_prices_zero(self):
        price = price_step(_census([0]), TECH, WritePolicy(batch_k=1))
        assert (price.write.nominal, price.carry.nominal) == (0.0, 0.0)

    def test_a_target_without_a_commit_energy_refuses_by_name(self):
        with pytest.raises(StepChoiceError) as info:
            price_step(_census([10, 10]), _physics(e_readout_per_byte=1.0), WritePolicy(batch_k=1))
        assert isinstance(info.value, ToolchainRefusal) and "commit energy" in str(info.value)

    def test_carried_bytes_without_their_constants_refuse_by_name(self):
        with pytest.raises(StepChoiceError) as info:
            carry_energy(_census([10, 10], carry=[(0, 5)]), _physics(e_program_per_byte=1.0))
        assert "e_readout_per_byte" in str(info.value)

    def test_no_carry_needs_no_carry_constant(self):
        assert carry_energy(_census([10, 10]), _physics(e_program_per_byte=1.0)).nominal == 0.0


class TestTheChoice:
    def test_the_least_writes_plus_carry_wins(self):
        steps = [_census([100, 100], carry=[(0, 1)]), _census([60, 60], carry=[(0, 40)]), _census([80, 80])]
        # K = 1: writes (600, 360, 480) x 1e3 plus carry (2, 80, 0) x 1e3: the middle step
        assert cheapest_step(steps, TECH, WritePolicy(batch_k=1)) == 1
        # K = 64: writes shrink by 64, carry does not: the least carry wins
        assert cheapest_step(steps, TECH, WritePolicy(batch_k=64)) == 2
        assert cheapest_step(steps[:2], TECH, WritePolicy(batch_k=64)) == 0

    def test_ties_go_to_the_first_step(self):
        steps = [_census([50]), _census([50]), _census([10, 40])]
        assert cheapest_step(steps, TECH, WritePolicy(batch_k=1)) == 0

    def test_steps_of_equal_exact_price_tie_although_their_float_prices_differ(self):
        """An E85 fixture witness (R18 R-P (4, 4, 4, 2) at C 174, K 1, the COMPASS reference row, c_byte = e_byte / 2):
        301 written tiles with 393,216 carried bytes and 303 with 360,448 price exactly the same, while their float
        prices differ in the last place; the earlier step (the smaller r) wins in either order."""
        e, tb = 0.06795304220119135, 8192
        tech = _physics(e_program_per_byte=e, e_dma_per_byte=0.0, e_readout_per_byte=e)
        first = _census([150 * tb, 151 * tb], carry=[(0, 393216 // 2)])
        later = _census([151 * tb, 152 * tb], carry=[(0, 360448 // 2)])
        exact = [Fraction(e) * 301 * tb + Fraction(e) / 2 * 393216, Fraction(e) * 303 * tb + Fraction(e) / 2 * 360448]
        prices = [price_step(c, tech, WritePolicy(batch_k=1)).nominal for c in (first, later)]
        assert exact[0] == exact[1] and prices[0] != prices[1]
        assert cheapest_step([first, later], tech, WritePolicy(batch_k=1)) == 0
        assert cheapest_step([later, first], tech, WritePolicy(batch_k=1)) == 0

    def test_a_corner_selects_the_band_end(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "banded", "display_name": "Banded", "description_file": "t.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {"e_program_per_byte": {"low": 1.0, "nominal": 1.0, "high": 1.0, "unit": "J",
                                                 "evidence_kind": "estimated", "note": "t"},
                          "e_dma_per_byte": {"nominal": 0.0, "unit": "J", "evidence_kind": "estimated", "note": "t"},
                          "e_readout_per_byte": {"low": 0.1, "nominal": 1.0, "high": 10.0, "unit": "J",
                                                 "evidence_kind": "estimated", "note": "t"}},
        })
        steps = [_census([100], carry=()), _census([20, 20], carry=[(0, 10)])]
        # writes 100 vs 40; carry 0 vs 10 x (0.1 | 1 | 10)
        assert cheapest_step(steps, physics, WritePolicy(batch_k=1), corner="low") == 1
        assert cheapest_step(steps, physics, WritePolicy(batch_k=1), corner="high") == 0

    def test_no_step_refuses_by_name(self):
        with pytest.raises(StepChoiceError):
            cheapest_step([], TECH, WritePolicy(batch_k=1))
