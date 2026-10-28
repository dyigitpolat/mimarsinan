"""[U-F4] Each carried position's value between the passes of a wavefront program is held as the chip declares it.

Undeclared, the carry is off chip (the dense census: read out after its producing pass, re-injected before its
reader). An on-chip carry buffer is a separate declared component (``carry_buffer_bytes``): a value produced and read
inside one wavefront group then stays in the buffer (written once, read once, ``carry_onchip_bytes``), priced at the
buffer's own access energy ``e_carry_buffer_per_byte`` (a physics constant: declared with its source, or the energy
headline refuses by name), and a group whose live carry exceeds the buffer refuses by name. A value read by a later
group crosses the boundary either way. Both planes state the same on-chip census. [U-F4 targeted pass, M1] The live
carry is sized in the run's domain before it meets the buffer's bytes: the census states its peak in carried values
too, a value-domain buffer holds them at the value grid (activation_bits; refused by name without it), and a spiking
buffer holds what the pass transfer buffers (the raster under verbatim, the window counts under collapse)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import ADMISSIBILITY_BEARING
from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.pass_carry import candidate_carry_census
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.quantities.value_domain import (
    ONCHIP_CARRY,
    ONCHIP_CENSUS,
    onchip_carry,
    record_carry_census,
    record_value_facts,
    value_bytes,
)
from mimarsinan.mapping.support.schedule.pass_carry import carried_wire_bytes, carry_census_from_spans
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from unit.deployment_record.g3_fixtures import constant, physics
from unit.mapping.pipelined_multiplexing.test_planes import _ladder_noc, _pipelined
from unit.mapping.lanes.lane_planes import STEPS, TRANSFER

KEY = "carry_buffer_bytes"


@pytest.fixture(scope="module", params=["together", "groups"])
def planes(request):
    platform, program = _pipelined("residual", request.param)
    scheduled, specs = _ladder_noc(program, platform, 10)
    mapping = build_hybrid_mapping_for_pipeline(program.ir, platform)
    return specs, scheduled, mapping


class TestTheDeclaration:
    def test_it_is_an_admissibility_bearing_platform_component_with_no_default(self):
        from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS

        entry = REGISTRY[KEY]
        assert str(entry.decision_class) == ADMISSIBILITY_BEARING and entry.section == "platform_constraints"
        assert KEY not in DEFAULT_PLATFORM_CONSTRAINTS
        assert entry.relevant.evaluate({"pipelined_multiplexing": True})
        assert not entry.relevant.evaluate({"position_multiplexing": True})

    def test_its_access_energy_is_a_physics_constant(self):
        from mimarsinan.deployment_record.platform_physics.constants import spec_for

        assert spec_for("e_carry_buffer_per_byte").display_unit == "pJ"


class TestBothPlanesStateTheOnChipCensus:
    def test_the_candidate_and_the_record_agree_group_by_group(self, planes):
        specs, scheduled, mapping = planes
        candidate = candidate_carry_census(specs, scheduled.noc, timesteps=STEPS, pass_transfer=TRANSFER)
        schedule = schedule_record_from_mapping(mapping, weight_bits=4, params_reloaded=0, timesteps=STEPS,
                                                pass_transfer=TRANSFER)
        assert schedule.wavefronts
        assert candidate["onchip_out_bytes"] == sum(g.onchip_out_bytes for g in schedule.wavefronts) > 0
        assert candidate["onchip_in_bytes"] == sum(g.onchip_in_bytes for g in schedule.wavefronts) > 0
        assert candidate["onchip_peak_live_bytes"] == max(g.onchip_peak_live_bytes for g in schedule.wavefronts)
        assert candidate["onchip_out_bytes"] <= candidate["boundary_out_bytes"]
        assert candidate["onchip_peak_live_elements"] == max(g.onchip_peak_live_elements for g in schedule.wavefronts)

    @pytest.mark.parametrize("bits", [4, 8, 16])
    def test_both_planes_size_the_live_values_by_one_rule(self, planes, bits):
        specs, scheduled, mapping = planes
        candidate = candidate_carry_census(specs, scheduled.noc, timesteps=STEPS, pass_transfer=TRANSFER)
        schedule = schedule_record_from_mapping(mapping, weight_bits=4, params_reloaded=0, timesteps=STEPS,
                                                pass_transfer=TRANSFER)
        need = value_bytes(candidate["onchip_peak_live_elements"], bits)
        for capacity, fits in ((need, True), (need - 1, False)):
            platform = _value_platform(bits, capacity)
            context = candidate_context_from_platform({**platform, "simulation_steps": STEPS}, carry=candidate)
            facts = record_value_facts(SimpleNamespace(schedule=schedule, identity=SimpleNamespace(platform=platform)))
            assert (onchip_carry(record_carry_census(schedule), platform)[1] is None) is fits
            assert (context.carry_onchip_refusal is None) is fits
            assert (ONCHIP_CARRY not in dict(facts.withheld)) is fits


CENSUS = {"carried_wires": 3, "carried_bytes": 48, "peak_live_bytes": 32, "boundary_out_bytes": 48,
          "boundary_in_bytes": 56, "onchip_out_bytes": 32, "onchip_in_bytes": 32, "onchip_peak_live_bytes": 24}


class TestTheDeclarationDecidesWhereTheCarryIsHeld:
    def test_undeclared_every_carried_value_crosses_the_boundary(self):
        carry, refusal = onchip_carry(CENSUS, {})
        assert (carry["boundary_out_bytes"], carry["boundary_in_bytes"]) == (48, 56)
        assert ONCHIP_CENSUS not in carry and refusal is None

    def test_declared_the_group_s_values_stay_in_the_buffer(self):
        carry, refusal = onchip_carry(CENSUS, {KEY: 24})
        assert (carry["boundary_out_bytes"], carry["boundary_in_bytes"], carry[ONCHIP_CENSUS]) == (16, 24, 64)
        assert refusal is None

    def test_a_group_whose_live_carry_exceeds_the_buffer_is_refused_by_name(self):
        _carry, refusal = onchip_carry(CENSUS, {KEY: 23})
        assert refusal is not None and "24" in refusal and "carry_buffer_bytes 23" in refusal

    def test_a_census_without_wavefront_groups_is_unchanged(self):
        plain = {k: v for k, v in CENSUS.items() if not k.startswith("onchip")}
        assert onchip_carry(plain, {KEY: 1}) == (plain, None)


#: Three carried wires inside one wavefront group of three passes: 130 values live over its first boundary, 90 over
#: its second.
SPANS = ((1, 100, 0, 1), (2, 60, 1, 2), (3, 30, 0, 2))
PEAK_VALUES = 130


def _census(transfer: str, steps: int = 32):
    return carry_census_from_spans(SPANS, boundary_count=3, timesteps=steps, transfer=transfer, groups=(0, 0, 0))


def _value_platform(bits, capacity):
    return {"core_semantics": "mvm", "activation_bits": bits, KEY: capacity}


class TestTheLiveCarryIsSizedInTheRunsDomain:
    @pytest.mark.parametrize("transfer", [COLLAPSE, VERBATIM])
    def test_the_census_states_the_live_peak_in_values_beside_its_bytes(self, transfer):
        census = _census(transfer)
        assert census["onchip_peak_live_elements"] == PEAK_VALUES
        assert census["onchip_peak_live_bytes"] == PEAK_VALUES * carried_wire_bytes(1, 32, transfer)

    @pytest.mark.parametrize("transfer", [COLLAPSE, VERBATIM])
    @pytest.mark.parametrize("bits", [4, 8, 16])
    def test_a_value_domain_buffer_holds_the_live_values_at_the_value_grid(self, transfer, bits):
        need = value_bytes(PEAK_VALUES, bits)
        assert onchip_carry(_census(transfer), _value_platform(bits, need))[1] is None
        refusal = onchip_carry(_census(transfer), _value_platform(bits, need - 1))[1]
        assert refusal is not None and f"{need} B" in refusal and f"activation_bits {bits}" in refusal
        assert f"{KEY} {need - 1}" in refusal

    @pytest.mark.parametrize("transfer", [COLLAPSE, VERBATIM])
    def test_a_spiking_buffer_holds_what_the_pass_transfer_buffers(self, transfer):
        need = _census(transfer)["onchip_peak_live_bytes"]
        assert onchip_carry(_census(transfer), {KEY: need})[1] is None
        assert f"{KEY} {need - 1}" in onchip_carry(_census(transfer), {KEY: need - 1})[1]

    def test_a_value_domain_buffer_without_the_value_grid_is_refused_by_name(self):
        _held, refusal = onchip_carry(_census(COLLAPSE), {"core_semantics": "mvm", KEY: 1 << 20})
        assert refusal is not None and "activation_bits" in refusal and KEY in refusal


def _energy(quantities, phys):
    pricing = price_absolute(quantities, phys)
    terms = {t.name: t for t in pricing.terms}
    refusals = {r.name: r.reason for r in pricing.refusals}
    return terms, refusals


def _quantities(onchip: float):
    values = {"synaptic_events": QuantityValue(1e6, "static"), "carry_out_bytes": QuantityValue(16.0, "static"),
              "carry_in_bytes": QuantityValue(24.0, "static"), ONCHIP_CARRY: QuantityValue(onchip, "static")}
    return Quantities(values)


class TestThePricer:
    def test_a_buffer_whose_access_energy_is_declared_prices_it(self):
        phys = physics("generic_estimated_22nm", e_carry_buffer_per_byte=constant(4.0, unit="pJ",
                                                                                 note="test: a cited SRAM access"))
        terms, refusals = _energy(_quantities(64.0), phys)
        assert "energy_dynamic_mj" in terms and "carry_buffer" in terms["energy_dynamic_mj"].source, refusals

    def test_a_buffer_without_a_declared_access_energy_is_refused_by_name(self):
        terms, refusals = _energy(_quantities(64.0), physics("generic_estimated_22nm"))
        assert "energy_dynamic_mj" not in terms
        assert "e_carry_buffer_per_byte" in refusals.get("energy_dynamic_mj", ""), refusals
