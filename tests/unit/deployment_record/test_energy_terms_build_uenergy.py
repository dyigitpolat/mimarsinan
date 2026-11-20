"""[U-ENERGY] The build's own cases: the design check's three (FT/uenergy/design_r3_check.md c-1 to c-3) and one per
item the build adds (the legs printed apart, the host's idle power over the load walls, the record plane's table of
what it does not produce, row drives on a bit-sliced dataflow, the record view's load quantities and the measured
model's programming off the period census, each pass's wall priced once per program, the static rows' walls).

Every constant is a test constant; the shipped descriptions are read as they ship. New names are imported inside the
tests, so a tree without them fails test by test."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

J_TO_MJ = 1e3
CORNERS = ("low", "nominal", "high")
BITS = 4
PLATFORM = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 8}], "weight_bits": BITS}
BASE = {"e_mac": 0.5, "t_cycle": 100.0, "e_dma_per_byte": 2.0, "e_program_per_byte": 3.0, "e_core_program": 5.0,
        "t_program_per_byte": 7.0, "t_core_program": 50.0}
STATIC = {"p_static_per_core": 0.25, "p_static_global": 4.0}  # mW
WIDTH = 8.0  # bytes per connectivity entry
CORE_ENTRIES = 12  # the record fixture's entries per core
REPO = Path(__file__).resolve().parents[3]


def _physics(name, constants, **extra):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: ({**v, "evidence_kind": "estimated", "note": "test"} if isinstance(v, dict) else
                          {"nominal": v, "evidence_kind": "estimated", "note": "test"}) for k, v in constants.items()},
        **extra})


def _plane_loads():
    from mimarsinan.deployment_record.quantities.planes import loads
    return loads


def _hand_passes(marks=frozenset()):
    def row(i, resident, cells, ext, kept=frozenset()):
        return PassRow(segment_index=i, resident=resident, core_cells=cells, marks=kept, core_extents=ext,
                       compute_steps=3)
    return ProgramPasses(rows=(row(0, False, (40, 20), ((8, 5), (4, 5)), marks), row(1, False, (30,), ((6, 5),)),
                               row(2, False, (16, 24), ((4, 4), (6, 4)))))


def _q(**extra):
    values = {"synaptic_events": 1e6, "cores_physical": 16, "latency_steps": 40.0, "host_macs": 0.0,
              "compute_op_count": 0.0, **extra}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _terms(pricing):
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _ends(band):
    return [band.low, band.nominal, band.high]


def _record():
    from unit.deployment_record.record_fixtures import make_full_record
    return make_full_record()


def _declared(record, physics, **platform):
    return replace(record, identity=replace(record.identity, platform={
        **record.identity.platform, **platform, "platform_physics_resolved": physics.to_dict()}))


def _record_pricing(record, constants, **platform):
    from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
    return absolute_pricing_for_record(_declared(record, _physics("record", constants), **platform))


def _kept_first_core_and_thirty_entries():
    """The record fixture with both segments written: segment 1 keeps its core 0 (a residency mark) and its core 1
    states 30 entries; segment 2 (one core of 12 entries) is rewritten. Two passes write, so the period rewrites core 1
    of segment 1 and core 0 of segment 2 (42 entries); the first load writes all three cores (54)."""
    record = _record()
    stages = list(record.schedule.stages)
    cores = (stages[1].cores[0], replace(stages[1].cores[1], connectivity_entries=30))
    s1 = replace(stages[1], resident_core_indices=(0,), cores=cores, connectivity_entries=CORE_ENTRIES + 30)
    s2 = replace(stages[2], programming="reprogram", connectivity_entries=CORE_ENTRIES, params_bytes=5000,
                 params_programmed=5000, cores=tuple(replace(c, params_bytes=5000) for c in stages[2].cores))
    return replace(record, schedule=replace(record.schedule, stages=(stages[0], s1, s2)))


class TestTheChecksCases:
    """FT/uenergy/design_r3_check.md c-1, c-2 and c-3."""

    def test_a_load_counts_the_entries_of_the_cores_its_census_writes(self):
        """c-1 (its mutant K1 takes the right number of entries from the wrong cores): 42 on the period, 54 on the
        first load, and the period's wall carries exactly those entries' bytes."""
        record = _kept_first_core_and_thirty_entries()
        loads = _plane_loads().record_load_inputs(record)
        assert loads.period.entries == 30 + CORE_ENTRIES
        assert loads.first.entries == CORE_ENTRIES + 30 + CORE_ENTRIES
        bare, _ = _terms(_record_pricing(record, BASE))
        counted, _ = _terms(_record_pricing(record, {**BASE, "bytes_per_connectivity_entry": WIDTH}))
        assert counted["latency_programming_s"].value - bare["latency_programming_s"].value == pytest.approx(
            7e-9 * WIDTH * (30 + CORE_ENTRIES), rel=1e-9)

    def test_at_a_declared_batch_the_writes_per_inference_and_the_reprogramming_agree_on_the_entries(self):
        """c-2: at a declared batch K the write law per inference and the reprogramming per round read one period
        census, its connectivity entries included: K x the exposed write time is the reprogramming wall (serial), and
        K x the write energy is the reprogramming energy less its static leg."""
        record = _kept_first_core_and_thirty_entries()
        terms, _ = _terms(_record_pricing(record, {**BASE, **STATIC, "bytes_per_connectivity_entry": WIDTH},
                                          inference_batch_K=4))
        assert 4 * terms["latency_write_exposed_s"].value == pytest.approx(terms["latency_programming_s"].value,
                                                                           rel=1e-12)
        assert 4 * terms["energy_write_mj"].value == pytest.approx(
            terms["energy_programming_mj"].value - terms["energy_static_programming_mj"].value, rel=1e-12)

    def test_the_write_law_names_the_entries_a_plane_does_not_count(self):
        """c-2's other half: where the census states no entries per core and the description declares their width,
        the per-inference write terms name the width with the plane's reason, never a silent zero."""
        from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
        loads = _plane_loads().load_inputs(_hand_passes(), PLATFORM, weight_bits=BITS, label="test")
        writes = WriteInputs.of(replace(loads.period.writes, policy=replace(loads.period.writes.policy, batch_k=4)))
        pricing = price_write_inputs(writes, _physics("width", {**BASE, "bytes_per_connectivity_entry": WIDTH}))
        assert "bytes_per_connectivity_entry(connectivity; not produced on this plane" in pricing.energy.basis

    def test_a_census_that_writes_nothing_names_no_entry(self):
        """A period that rewrites nothing writes no entry, so neither its wall nor its energy names the entries' width,
        even on a plane that counts no entries (one written pass, never displaced)."""
        lone = ProgramPasses(rows=tuple(replace(row, resident=i != 1) for i, row in enumerate(_hand_passes().rows)))
        loads = _plane_loads().load_inputs(lone, PLATFORM, weight_bits=BITS, label="test",
                                           entries_absent=_plane_loads().CANDIDATE_NO_ENTRIES)
        terms, _ = _terms(price_absolute(_q(), _physics("lone", {**BASE, "bytes_per_connectivity_entry": WIDTH}),
                                         loads=loads))
        assert terms["latency_programming_s"].value == 0.0
        for name in ("latency_programming_s", "energy_programming_mj"):
            assert "bytes_per_connectivity_entry" not in terms[name].source, name
        assert "bytes_per_connectivity_entry(connectivity; not produced" in terms["energy_load_mj"].source

    def test_no_text_names_the_removed_second_load_pricer(self):
        """c-3: the ladder's tile-write pricer is gone, and no source or document text names it."""
        stale = [str(path.relative_to(REPO)) for root in ("src", "docs") for path in sorted((REPO / root).rglob("*"))
                 if path.is_file() and path.suffix in (".py", ".md") and "tile_writes" in path.read_text("utf-8")]
        assert not stale, stale


class TestTheLegsAndTheHost:
    def _loads(self):
        return _plane_loads().load_inputs(_hand_passes(), PLATFORM, weight_bits=BITS, label="test")

    def test_each_load_prints_its_static_leg_apart(self):
        """T-E1c: each load's static leg is a term of its own beside the load energy that holds it."""
        static, _ = _terms(price_absolute(_q(), _physics("legs", {**BASE, **STATIC}), loads=self._loads()))
        bare, refused = _terms(price_absolute(_q(), _physics("bare", BASE), loads=self._loads()))
        watts = 0.25e-3 * 16 + 4e-3
        for wall, energy, leg in (("latency_load_s", "energy_load_mj", "energy_static_load_mj"),
                                  ("latency_programming_s", "energy_programming_mj", "energy_static_programming_mj")):
            assert static[leg].value == pytest.approx(watts * static[wall].value * J_TO_MJ, rel=1e-12), leg
            assert static[energy].value - bare[energy].value == pytest.approx(static[leg].value, rel=1e-12), leg
            assert leg not in bare and "undeclared" in refused[leg], leg

    def test_the_hosts_idle_power_is_named_over_the_load_walls(self):
        """T-E1d: an edge host's idle power over a load's wall is named, not priced (its whole-system line is per
        inference)."""
        from mimarsinan.deployment_record.platform_physics.hosts import store as ST
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
        from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
        physics = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(
            {"host": host_description_block("jetson_orin_nano_8gb"), "store": ST.store_block(ST.CHIP, "test")}))
        terms, _ = _terms(price_absolute(_q(), physics, loads=self._loads()))
        for wall, energy in (("latency_load_s", "energy_load_mj"), ("latency_programming_s", "energy_programming_mj")):
            assert f"p_host_idle(the host's idle power over {wall}" in terms[energy].source, energy


class TestThePlanesStateTheirCounts:
    def test_the_record_plane_states_what_it_does_not_produce(self):
        """m-r2-9: a record that seals no NoC census states it in its table, and a declared hop energy over it is
        named with that reason beside the estimate."""
        from mimarsinan.deployment_record.quantities.from_record import from_record
        from mimarsinan.deployment_record.quantities.planes.states import RECORD_UNPRODUCED
        from unit.deployment_record.record_fixtures import make_traffic
        record = replace(_record(), traffic=make_traffic(with_noc=False))
        record = replace(record, energy=replace(record.energy, synaptic_events=1000))  # the headline prices
        assert from_record(record).unproduced_reason("noc_total_hops") == RECORD_UNPRODUCED["noc_total_hops"]
        terms, _ = _terms(_record_pricing(record, {**BASE, "e_inter_tile_hop": 1.0}))
        assert (f"e_inter_tile_hop(inter_tile; not produced on this plane: {RECORD_UNPRODUCED['noc_total_hops']})"
                in terms["energy_dynamic_mj"].source)

    def test_a_spiking_platforms_value_bytes_are_a_zero_by_declaration(self):
        """A platform that names no value domain sends spikes, never values: on the record plane its value NoC bytes
        are a zero by that declaration (its reason printed), never a count the plane does not produce."""
        from mimarsinan.deployment_record.quantities.from_record import from_record
        record = replace(_record(), energy=replace(_record().energy, synaptic_events=1000))
        view = from_record(record)
        assert view.known_zero_reason("noc_value_bytes") and not view.unproduced_reason("noc_value_bytes")
        terms, _ = _terms(_record_pricing(record, {**BASE, "e_noc_value_per_byte_hop": 1.0}))
        assert (f"zero by declaration: e_noc_value_per_byte_hop ({view.known_zero_reason('noc_value_bytes')})"
                in terms["energy_dynamic_mj"].source)

    def test_a_bit_sliced_dataflow_derives_its_row_drives(self):
        """m-r2-7: a bit-sliced crossbar drives every used row once per input slice, so its row drives are
        ceil(macs x cells_per_weight / array_cols) x (input_bits / dac_bits), and the row-drive energy prices them."""
        from mimarsinan.deployment_record.cost.absolute.context import PricingContext
        physics = get_platform_physics("isaac_like")
        q = _q(macs=1000.0, timesteps=1.0)
        derived = PricingContext.create(q, physics).quantities
        assert derived.has("boundary_events")
        assert derived.get("boundary_events").value == math.ceil(1000 * 8 / 128) * 16
        dynamic = next(t for t in price_absolute(q, physics).terms if t.name == "energy_dynamic_mj")
        assert "row_drive: e_row_drive" in dynamic.source

    def test_the_record_views_load_quantities_state_their_rule(self):
        """m-r3-3: the record view's scalar load quantities keep one rule on both planes (the segments marked
        reprogram, the candidate's non-resident passes: the mapping's census and the record agree field by field),
        and their text states that no pricer reads them, the load law pricing its own census; the scalar payload
        pricer is gone."""
        from mimarsinan.deployment_record.cost.absolute import formulas
        from mimarsinan.deployment_record.quantities.from_record import from_record
        from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS
        lone = from_record(_record())
        assert [lone.get(k).value for k in ("reprogrammed_bytes", "connectivity_entries", "reprogrammed_cores")] == [
            100.0, 24.0, 2.0]
        for key in ("reprogrammed_bytes", "connectivity_entries", "reprogrammed_cores"):
            assert "no pricer reads" in QUANTITY_SPECS[key].doc, key
        assert "keeps a lone written pass loaded across rounds" in QUANTITY_SPECS["reprogrammed_bytes"].doc
        assert not hasattr(formulas, "programming_payload_bytes")

    def test_a_record_whose_period_census_cannot_be_stated_refuses_the_measured_model_by_name(self, tmp_path):
        """Ruling (d): the measured model prices what the period census writes, so a record whose census cannot be
        stated (no complete core geometry) has no measured surface, named in the emitted report, and the objective
        view never raises."""
        import json
        from mimarsinan.deployment_record.cost.absolute.report import emit_physics_report
        from mimarsinan.deployment_record.cost.segments import programming_refusal
        from mimarsinan.deployment_record.objectives import OBJECTIVES
        from mimarsinan.deployment_record.objectives.views import DeploymentRecordView
        record = _record()
        assert programming_refusal(record) == ""
        bare = replace(record, identity=replace(record.identity, platform={"weight_bits": 8}))
        assert "that census cannot be stated" in programming_refusal(bare)
        view = DeploymentRecordView(record=bare)
        assert not view.costable() and "programming_energy_mj" not in {s.key for s in OBJECTIVES.available_for(view)}
        path = emit_physics_report(_declared(bare, _physics("record", BASE)), str(tmp_path))
        notes = json.loads(Path(str(path)).read_text())["notes"]
        assert any(n.startswith("refused measured_cost_model: ") for n in notes), notes

    def test_the_measured_model_programs_what_the_period_census_writes(self):
        """The ruling (d) of 2026-11-12: the measured model's per-segment programming prices the cores the record's
        period census writes in each segment (their payload and their entries), one rule with every load."""
        from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
        model = DeploymentCostModel()
        lone = model.segment_initialization(_record())
        assert [_ends(find_term(s.terms, "payload_bytes").band) for s in lone] == [[0.0, 0.0, 0.0]] * 2
        rotating = model.segment_initialization(_kept_first_core_and_thirty_entries())
        width = model.bytes_per_connectivity_entry
        for segment, (payload, entries) in zip(rotating, ((5000, 30), (5000, CORE_ENTRIES))):
            assert _ends(find_term(segment.terms, "payload_bytes").band) == pytest.approx(
                [payload + entries * getattr(width, c) for c in CORNERS], rel=1e-12)


class TestOneWallPerPassPerProgram:
    def test_the_period_reads_the_every_core_walls_again_and_prices_the_same(self):
        """The cost rule of the build: each pass's write wall is priced once per program for both loads (the
        every-core census's walls read again for the period's passes that write the same cores), and the period's
        wall equals the write law's own over the period census at every corner; a pass that keeps a core is priced
        again, never read from the every-core census."""
        from mimarsinan.deployment_record.cost.absolute.walls.write_walls import PassWalls
        from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
        physics = _physics("walls", {**BASE, "t_program_per_byte": {"low": 5.0, "nominal": 7.0, "high": 11.0}})
        loads = _plane_loads().load_inputs(_hand_passes(marks=frozenset({1})), PLATFORM, weight_bits=BITS,
                                           label="test")
        memo = PassWalls()
        price_write_inputs(WriteInputs.of(loads.first.writes), physics, lifetime=False, walls=memo)
        assert memo.priced == 3
        period = price_write_inputs(WriteInputs.of(loads.period.writes), physics, lifetime=False, walls=memo)
        assert memo.priced == 4  # the pass that keeps its core 1 is priced again; the other two are read back
        alone = price_write_inputs(WriteInputs.of(loads.period.writes), physics, lifetime=False)
        assert _ends(period.exposed) == pytest.approx(_ends(alone.exposed), rel=1e-12)
        assert _ends(period.energy) == pytest.approx(_ends(alone.energy), rel=1e-12)


class TestTheAggregateHoldsTheLocalMemories:
    def test_a_whole_system_event_energy_supersedes_the_local_memory_rows(self):
        """A per-event aggregate holds every on-chip energy of its events, a core's local memories among them, so a
        description that declares both never sums the two."""
        rows = {"e_local_mem_read_per_byte": 0.6, "e_local_mem_write_per_byte": 1.1}
        q = _q(local_mem_read_bytes=5000.0, local_mem_write_bytes=700.0)
        alone = price_absolute(q, _physics("rows", {"e_mac": 0.5, "t_cycle": 100.0, **rows}))
        held = price_absolute(q, _physics("aggregate", {"e_synaptic_event_total": 3.0, "t_cycle": 100.0, **rows}))
        dynamic = {name: next(t for t in p.terms if t.name == "energy_dynamic_mj") for name, p in
                   (("alone", alone), ("held", held))}
        assert "local_memory_read: e_local_mem_read_per_byte" in dynamic["alone"].source
        assert "e_local_mem" not in dynamic["held"].source
        assert dynamic["held"].value == pytest.approx(3.0e-12 * 1e6 * J_TO_MJ, rel=1e-12)


class TestTheStaticRowsStateTheirWalls:
    def test_the_static_rows_state_every_wall_they_multiply(self):
        """§4.9: each static row's multiplicand names every chip time its power is charged over."""
        from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS
        for row in ("p_static_per_core", "p_static_global", "p_carry_buffer_static_per_byte"):
            walls = {alt.split(" x ")[-1].strip() for alt in PHYSICS_CONSTANTS[row].multiplicand.split(" | ")}
            assert walls == {"e2e_latency_s", "latency_load_s", "latency_programming_s", "total_s"}, row
