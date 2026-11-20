"""[U-ENERGY V-M1] The static sum's one leg form: every energy figure over a wall holds the chip's static power over
that wall, or is refused with the leg's reason, never a number without its leg.

The verify's four cases, each on the candidate plane (the scaling ladder's priced program, its load figures and Pi)
and on the record plane (a sealed record's pricing, its share variants and the measured model): a declared carry
buffer whose static row the description does not state, and a description that declares no cycle time (the
per-inference wall refused while the loads' walls price). Then the structure: the sum's power field is read nowhere
in the toolchain but in the leg form, and a ratio of two energy costs is printed only over like legs. Every constant
here is a test constant."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import replace
from pathlib import Path

import pytest

import mimarsinan
import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.schema import ComputePartitionRecord
from unit.deployment_record.introspection.test_scaling_ladder import CONSTANTS, READING, _facts, _network, _platform

ROOT = Path(mimarsinan.__file__).resolve().parent
J_TO_MJ = 1e3
STATIC = {"p_static_per_core": 0.25, "p_static_global": 4.0}  # mW
BUFFER = {"e_carry_buffer_per_byte": 1.0, "area_carry_buffer_per_byte": 1.0}  # the buffer's two other rows
BUFFER_ROW = "p_carry_buffer_static_per_byte"
RECORD_BASE = {"e_mac": 0.5, "t_cycle": 100.0, "e_dma_per_byte": 2.0, "e_program_per_byte": 3.0,
               "t_program_per_byte": 7.0}
LOADS = (("latency_load_s", "energy_load_mj", "energy_static_load_mj"),
         ("latency_programming_s", "energy_programming_mj", "energy_static_programming_mj"))
ENERGY_AXIS = ("energy_dynamic_mj", "energy_static_mj")
PARTITION = ComputePartitionRecord(onchip_params=900, host_params=0, total_params=900, onchip_macs=68096, host_macs=0,
                                   total_macs=68096)


def physics(name, constants):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "test"} for k, v in constants.items()}})


def _ends(band):
    return [band.low, band.nominal, band.high]


def _watts(ph, cores):
    return [getattr(ph.band("p_static_per_core"), c) * cores + getattr(ph.band("p_static_global"), c)
            for c in ("low", "nominal", "high")]


_SCHEDULES = {}


def ladder_point(constants, **declared):
    """The scaling ladder's hand point (8 cores, K = 16) priced on a test description: (priced program, row)."""
    network = _network(4)
    point = {**_platform(8), **declared}
    key = tuple(sorted(declared.items()))
    if key not in _SCHEDULES:
        census = ladder.shape_census(network, point)
        _SCHEDULES[key] = (census, ladder.schedule_program(network, census, point, _facts))
    census, scheduled = _SCHEDULES[key]
    priced_point = {**point, "platform_physics_resolved": physics("leg_form", constants).to_dict()}
    priced = ladder.price_scheduled(network, census, scheduled, priced_point)
    row = ladder.ladder_row(network, census, priced, {**priced_point, "inference_batch_K": 16}, READING)
    return priced, row


def record_with(constants, **platform):
    from unit.deployment_record.record_fixtures import make_rewritten_record
    record = make_rewritten_record()
    return replace(record, identity=replace(record.identity, platform={
        **record.identity.platform, **platform, "platform_physics_resolved": physics("leg_form", constants).to_dict()}),
        energy=replace(record.energy, synaptic_events=1e5),  # a sealed event census and a partition census
        utilization=replace(record.utilization, partition=PARTITION))  # (no host work), so the headline prices


def _by_name(pricing):
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


# ------------------------------------------------------------------------------------------- the candidate plane
class TestTheCandidatePlane:
    def test_a_carry_buffer_without_its_static_row_refuses_every_load_energy_and_pi_by_its_name(self):
        priced, row = ladder_point({**CONSTANTS, **STATIC, **BUFFER}, carry_buffer_bytes=4096)
        program, refused = priced.terms["program"], priced.refusals["program"]
        assert BUFFER_ROW in refused["energy_per_inference_mj"]
        for wall, energy, leg in LOADS:
            assert wall in program, wall  # the wall prices: only the buffer's row is refused
            assert energy not in program and leg not in program, energy
            assert BUFFER_ROW in refused[energy] and BUFFER_ROW in refused[leg], energy
        axis = row["energy"]
        assert axis["pi"] is None and axis["terms"]["compute_only"] is None
        assert BUFFER_ROW in axis["absent"]

    def test_with_its_static_row_the_buffer_joins_every_leg(self):
        constants = {**CONSTANTS, **STATIC, **BUFFER, BUFFER_ROW: 50.0}
        priced, row = ladder_point(constants, carry_buffer_bytes=4096)
        program = priced.terms["program"]
        watts = [w + 50e-9 * 4096 for w in _watts(physics("leg_form", constants), 8)]
        for wall, energy, leg in LOADS:
            assert _ends(program[leg]) == pytest.approx([w * t * J_TO_MJ for w, t in zip(watts, _ends(program[wall]))],
                                                        rel=1e-12)
        assert row["energy"]["pi"] is not None

    def test_no_cycle_time_refuses_the_compute_only_cost_and_pi_by_the_walls_reason(self):
        priced, row = ladder_point({**{k: v for k, v in CONSTANTS.items() if k != "t_cycle"}, **STATIC})
        program, refused = priced.terms["program"], priced.refusals["program"]
        assert "e2e_latency_s" in refused and "energy_per_inference_mj" in refused
        assert "e2e_latency_s" in refused["energy_static_mj"]  # the leg is refused by name, beside the headline
        for wall, energy, leg in LOADS:  # the loads' walls price, so their legs do
            assert energy in program and leg in program, energy
        compute = priced.refusals["compute_only"]
        assert "energy_dynamic_mj" in priced.terms["compute_only"] and "e2e_latency_s" in compute["energy_static_mj"]
        axis = row["energy"]
        assert axis["terms"]["compute_only"] is None and axis["pi"] is None and axis["overhead"] is None
        assert "e2e_latency_s" in axis["absent"]


# ---------------------------------------------------------------------------------------------- the record plane
class TestTheRecordPlane:
    def test_a_carry_buffer_without_its_static_row_refuses_the_load_energies_and_the_measured_total(self):
        from mimarsinan.deployment_record.cost import DeploymentCostModel
        record = record_with({**RECORD_BASE, **STATIC, **BUFFER}, carry_buffer_bytes=4096)
        terms, refused = _by_name(absolute_pricing_for_record(record))
        assert BUFFER_ROW in refused["energy_per_inference_mj"]
        for wall, energy, leg in LOADS:
            assert wall in terms and energy not in terms and leg not in terms, energy
            assert BUFFER_ROW in refused[energy] and BUFFER_ROW in refused[leg], energy
        model = DeploymentCostModel()
        names = {t.name for t in model.energy(record)}
        assert "total_mj" not in names and "modeled_static_mj" not in names and "measured_total_mj" in names
        notes = " | ".join(model.evaluate(record).notes)
        assert "total_mj withheld" in notes and BUFFER_ROW in notes

    def test_with_its_static_row_the_measured_total_holds_the_buffer(self):
        from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
        constants = {**RECORD_BASE, **STATIC, **BUFFER, BUFFER_ROW: 50.0}
        record = record_with(constants, carry_buffer_bytes=4096)
        model = DeploymentCostModel()
        cores = sum(int(c.get("count", 1)) for c in record.identity.platform["cores"])
        watts = [w + 50e-9 * 4096 for w in _watts(physics("leg_form", constants), cores)]
        total_s = find_term(model.latency(record), "total_s").band
        leg = find_term(model.energy(record), "modeled_static_mj").band
        assert _ends(leg) == pytest.approx([w * t * J_TO_MJ for w, t in zip(watts, _ends(total_s))], rel=1e-12)

    def test_no_cycle_time_refuses_the_per_inference_figures_and_the_compute_only_share(self):
        from mimarsinan.deployment_record.cost.absolute.host_split import price_shares, shares
        from mimarsinan.deployment_record.cost.absolute.report import (
            declared_physics_of, record_compute_split, record_geometry, record_readout_bits)
        from mimarsinan.deployment_record.cost.absolute.validity import UNJUDGED
        from mimarsinan.deployment_record.quantities.from_record import from_record
        constants = {**{k: v for k, v in RECORD_BASE.items() if k != "t_cycle"}, **STATIC}
        record = record_with(constants)
        geometry = record_geometry(record)
        split = shares(*price_shares(from_record(record), declared_physics_of(record),
                                     compute=record_compute_split(record), readout_bits=record_readout_bits(record),
                                     geometry=UNJUDGED if geometry is None else geometry),
                       ENERGY_AXIS, "energy_per_inference_mj")
        assert split["compute_only"] is None and "e2e_latency_s" in split["absent"]
        terms, refused = _by_name(absolute_pricing_for_record(record))
        assert "energy_per_inference_mj" not in terms and "energy_static_mj" in refused
        assert "e2e_latency_s" in refused["energy_per_inference_mj"] and "e2e_latency_s" in refused["energy_static_mj"]
        for wall, energy, leg in LOADS:
            assert energy in terms and leg in terms, energy

    def test_the_measured_total_over_its_own_wall_still_holds_its_leg(self):
        from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
        constants = {**{k: v for k, v in RECORD_BASE.items() if k != "t_cycle"}, **STATIC}
        record = record_with(constants)
        model = DeploymentCostModel()
        cores = sum(int(c.get("count", 1)) for c in record.identity.platform["cores"])
        total_s = find_term(model.latency(record), "total_s").band
        leg = find_term(model.energy(record), "modeled_static_mj").band
        watts = _watts(physics("leg_form", constants), cores)
        assert _ends(leg) == pytest.approx([w * t * J_TO_MJ for w, t in zip(watts, _ends(total_s))], rel=1e-12)


# ----------------------------------------------------------------------------------------------------- structure
def _power_fields():
    """The static sum's power fields: its fields that carry a band of watts."""
    from mimarsinan.deployment_record.cost.absolute.walls.static_power import StaticSum
    return {f.name for f in dataclasses.fields(StaticSum) if "Band" in str(f.type)}


LEG_FORM = ("deployment_record/cost/absolute/walls/static_power.py", "static_leg")


def power_field_reads(extra=()):
    """Every read of the sum's power fields in the toolchain's source (an attribute, or its name handed to
    ``getattr``), outside the leg form: (module, line, what)."""
    fields = _power_fields()
    sources = [(str(p.relative_to(ROOT)), p.read_text()) for p in sorted(ROOT.rglob("*.py"))] + list(extra)
    found = []
    for module, text in sources:
        tree = ast.parse(text)
        allowed = set()
        if module == LEG_FORM[0]:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == LEG_FORM[1]:
                    allowed.update(id(n) for n in ast.walk(node))
        for node in ast.walk(tree):
            if id(node) in allowed:
                continue
            if isinstance(node, ast.Attribute) and node.attr in fields:
                found.append((module, node.lineno, node.attr))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
                  and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in fields):
                found.append((module, node.lineno, f"getattr {node.args[1].value}"))
    return found


class TestTheOneLegForm:
    def test_the_sums_power_fields_are_read_nowhere_but_in_the_leg_form(self):
        assert _power_fields(), "the static sum carries its power in a field of its own"
        assert power_field_reads() == []

    def test_a_planted_reader_of_the_sum_is_red(self):
        field = sorted(_power_fields())[0]
        planted = ("deployment_record/cost/absolute/planted.py",
                   f"def planted(pricing, wall):\n    return pricing.static_power.{field}.nominal * wall\n")
        assert power_field_reads([planted]) == [("deployment_record/cost/absolute/planted.py", 2, field)]

    def test_no_module_multiplies_a_sum_by_a_wall_but_the_leg_form(self):
        from mimarsinan.deployment_record.cost.absolute.walls import static_power
        assert hasattr(static_power, "static_leg") and not hasattr(static_power, "static_over")
        for path in ROOT.rglob("*.py"):
            assert "static_over" not in path.read_text(), path

    def test_the_leg_form_refuses_a_refused_part_before_it_prices(self):
        from mimarsinan.deployment_record.cost.absolute.walls.static_power import StaticSum, static_leg
        from mimarsinan.deployment_record.schema.provenance import Band
        watts, wall = Band(1.0, 2.0, 3.0, "static power"), Band(1.0, 1.0, 1.0, "a wall")
        refused = StaticSum(watts=watts, declared=True, refused=((BUFFER_ROW, "the row is unstated (test)"),))
        assert static_leg(refused, wall, "a_wall_s").refused == "the row is unstated (test)"
        assert static_leg(refused, wall, "a_wall_s").band is None
        priced = StaticSum(watts=watts, declared=True, parts=("p_static_global",))
        assert _ends(static_leg(priced, wall, "a_wall_s").band) == [1e3, 2e3, 3e3]
        over_a_refused_wall = static_leg(priced, None, "a_wall_s", "the wall's reason (test)")
        assert over_a_refused_wall.band is None and "the wall's reason (test)" in over_a_refused_wall.refused
        undeclared = static_leg(StaticSum(), wall, "a_wall_s")
        assert undeclared.band is None and not undeclared.refused and undeclared.named


class TestLikeLegs:
    def test_pi_is_not_printed_over_an_aggregate_held_in_its_divisor_alone(self):
        """A chip whose per-event aggregate holds its static power prices it over its events (the divisor's wall) and
        names it unpriced over the load's wall (the numerator's), so the two costs hold unlike legs."""
        constants = {**CONSTANTS, "p_static_global": 4.0, "e_synaptic_event_total": 3.0}
        priced, row = ladder_point(constants)
        assert "energy_load_mj" in priced.terms["program"]
        axis = row["energy"]
        assert axis["pi"] is None and axis["pi_host_included"] is None and axis["overhead"] is None
        assert "unlike static legs" in axis["absent"]

    def test_pi_prints_where_both_costs_hold_the_same_parts(self):
        priced, row = ladder_point({**CONSTANTS, **STATIC})
        assert row["energy"]["pi"] is not None and row["energy"]["absent"] is None
        legs = priced.static_legs
        assert legs["once"] == legs["compute_only"] == legs["period"] and legs["once"]


class TestTheStepPrice:
    """m-3: the step price's static sum is required, or absent by name; a refused leg refuses the step."""

    def _step(self):
        from mimarsinan.deployment_record.quantities.write_law import WritePolicy
        from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus
        census = PassPeriodCensus(passes=(PassCensus(position=0, segment_index=0, resident=False, cores=2,
                                                     resident_cores=0, written_cells_backing_store=1000,
                                                     written_cells_producer=0, payload_bytes=1000,
                                                     compute_steps=0),), boundaries=())
        return census, WritePolicy(batch_k=4)

    def test_the_sum_is_required(self):
        from mimarsinan.deployment_record.cost.absolute.step_choice import cheapest_step, price_step
        census, policy = self._step()
        ph = physics("step", {**RECORD_BASE, **STATIC})
        with pytest.raises(TypeError):
            price_step(census, ph, policy)
        with pytest.raises(TypeError):
            cheapest_step([census], ph, policy)

    def test_absent_by_name_it_prices_no_leg_and_a_refused_part_refuses_the_step(self):
        from mimarsinan.deployment_record.cost.absolute.step_choice import StepChoiceError, price_step
        from mimarsinan.deployment_record.cost.absolute.walls.static_power import NO_STATIC, StaticSum
        from mimarsinan.deployment_record.schema.provenance import Band
        census, policy = self._step()
        ph = physics("step", {**RECORD_BASE, **STATIC})
        bare = price_step(census, ph, policy, static=NO_STATIC)
        watts = Band(1e-3, 2e-3, 3e-3, "static power")
        priced = price_step(census, ph, policy, static=StaticSum(watts=watts, declared=True, parts=("p",)))
        assert priced.write.nominal > bare.write.nominal
        with pytest.raises(StepChoiceError, match="unstated"):
            price_step(census, ph, policy, static=StaticSum(watts=watts, declared=True,
                                                            refused=((BUFFER_ROW, "the row is unstated (test)"),)))
