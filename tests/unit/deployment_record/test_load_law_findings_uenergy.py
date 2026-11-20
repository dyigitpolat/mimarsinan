"""[U-ENERGY] The load law's cases the design reviews named (FT/uenergy/design_r2_review.md 35780502).

N-M1: a load's connectivity entries are those of exactly the cores its census writes, so a period that rewrites
nothing prices nothing, and a rotating period counts only its rewritten cores' entries.
N-M2: one test per surviving mutant of the review (Q1 to Q6, ``review_r2/steps/w10_mutants.py``, set ``mine``), and the
link's ends named on a program load's transfer (U-J5's behaviour, here with a load census).
m-1: a declared static power over a refused load wall refuses the load energy by the wall's reason (the headline's
rule), with the measured reason: where the wall prices, the static leg is the load energy's dominant part.
m-6: the static sum is a required argument of the load law, the ladder keeps no second load pricer, and two quantity
views that differ only in a zero by declaration are unequal.

Every constant is a test constant; the shipped descriptions are read as they ship. New names are imported inside the
tests, so a tree without them fails test by test."""

from __future__ import annotations

from dataclasses import replace

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
#: the hand passes (one resident pass of two cores, two written passes of one and two cores, 4-bit weights)
FIRST_BYTES, FIRST_CORES = 20 + 10 + 15 + 8 + 12, 5
#: the record fixture: 20 declared cores, segment 1 written (2 cores, 12 entries each, 5,000 payload bytes each),
#: segment 2 resident (1 core, 12 entries, 5,000 bytes on its first load)
RECORD_CORES, RECORD_FIRST_BYTES, RECORD_FIRST_CORES, CORE_ENTRIES = 20, 15000, 3, 12


def _physics(name, constants, **extra):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: ({**v, "evidence_kind": "estimated", "note": "test"} if isinstance(v, dict) else
                          {"nominal": v, "evidence_kind": "estimated", "note": "test"}) for k, v in constants.items()},
        **extra})


def _plane_loads():
    from mimarsinan.deployment_record.quantities.planes import loads as plane_loads
    return plane_loads


def _passes():
    def row(i, resident, cells, ext):
        return PassRow(segment_index=i, resident=resident, core_cells=cells, marks=frozenset(), core_extents=ext,
                       compute_steps=3)
    return ProgramPasses(rows=(row(0, True, (40, 20), ((8, 5), (4, 5))), row(1, False, (30,), ((6, 5),)),
                               row(2, False, (16, 24), ((4, 4), (6, 4)))))


def _loads(platform=PLATFORM):
    return _plane_loads().load_inputs(_passes(), platform, weight_bits=BITS, label="test")


def _q(**extra):
    values = {"synaptic_events": 1e6, "cores_physical": 16, "latency_steps": 40.0, "host_macs": 0.0,
              "compute_op_count": 0.0, **extra}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _terms(pricing):
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _ends(band):
    return [band.low, band.nominal, band.high]


def _declared(record, physics, **platform):
    return replace(record, identity=replace(record.identity, platform={
        **record.identity.platform, **platform, "platform_physics_resolved": physics.to_dict()}))


def _record():
    from unit.deployment_record.record_fixtures import make_full_record
    return make_full_record()


def _record_pricing(record, constants, **platform):
    from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
    return absolute_pricing_for_record(_declared(record, _physics("record", constants), **platform))


def _two_written_passes_with_a_kept_core():
    """The record fixture with both segments written: segment 1 keeps its core 1 across periods (a residency mark),
    segment 2 (one core, its segment total equal to its core's 12 entries) is rewritten. Two passes write, so the
    period rotates and rewrites core 0 of each: 24 entries; the first load writes all three cores: 36 entries."""
    record = _record()
    stages = list(record.schedule.stages)
    s1 = replace(stages[1], resident_core_indices=(1,))
    s2 = replace(stages[2], programming="reprogram", connectivity_entries=CORE_ENTRIES, params_bytes=5000,
                 params_programmed=5000, cores=tuple(replace(c, params_bytes=5000) for c in stages[2].cores))
    return replace(record, schedule=replace(record.schedule, stages=(stages[0], s1, s2)))


class TestTheEntriesComeFromTheCensus:
    """N-M1: one census decides a load's writes and its entries."""

    def test_a_period_that_rewrites_nothing_prices_zero_with_the_census_reason(self):
        """The record fixture's period census rewrites nothing (one written pass is never displaced): with a declared
        entry width and static rows the reprogramming per round is 0 in time and energy, by the census's reason, and
        no connectivity entry is priced in it (the second prototype priced 1.3056e-05 mJ and 1.344e-06 s)."""
        from mimarsinan.deployment_record.quantities.planes.loads import sealed_total_disagreements
        terms, _ = _terms(_record_pricing(_record(), {**BASE, **STATIC, "bytes_per_connectivity_entry": WIDTH}))
        assert terms["latency_programming_s"].value == 0.0
        assert _ends(terms["energy_programming_mj"].band) == [0.0, 0.0, 0.0]

        def priced(text):  # [m-2] the fixture's sealed totals differ from its cores', named apart: no entry priced
            for note in sealed_total_disagreements(_record()):
                text = text.replace(note, "")
            return text
        source = priced(terms["energy_programming_mj"].source)
        assert "nothing is rewritten per period" in source and "connectivity" not in source, source
        assert "connectivity" not in priced(terms["latency_programming_s"].source)

    def test_a_rotating_period_counts_the_entries_of_its_rewritten_cores_only(self):
        loads = _plane_loads().record_load_inputs(_two_written_passes_with_a_kept_core())
        assert loads.first.entries == 3 * CORE_ENTRIES
        assert loads.period.entries == 2 * CORE_ENTRIES  # core 0 of each pass; the kept core's entries are not

    def test_a_core_that_fuses_several_declared_tiles_counts_its_entries_once(self):
        """A chip that declares a 128 x 256 core: each 256 x 256 core of the record fuses two declared tiles, and its
        12 entries are counted once (the first load writes 3 cores, 6 tiles: 36 entries, never 72)."""
        record = _record()
        platform = {**record.identity.platform, "cores": [{"max_axons": 128, "max_neurons": 256, "count": 80}]}
        fused = replace(record, identity=replace(record.identity, platform=platform))
        loads = _plane_loads().record_load_inputs(fused)
        assert [p.cores for p in loads.first.writes.census.census.passes] == [4, 2]
        assert loads.first.entries == 3 * CORE_ENTRIES

    def test_the_rotating_periods_entries_are_priced_once_with_their_wall(self):
        record = _two_written_passes_with_a_kept_core()
        bare, _ = _terms(_record_pricing(record, {**BASE, **STATIC}))
        counted, _ = _terms(_record_pricing(record, {**BASE, **STATIC, "bytes_per_connectivity_entry": WIDTH}))
        entry_bytes = 2 * CORE_ENTRIES * WIDTH
        wall = 7e-9 * entry_bytes
        watts = 0.25e-3 * RECORD_CORES + 4e-3
        assert counted["latency_programming_s"].value - bare["latency_programming_s"].value == pytest.approx(
            wall, rel=1e-9)
        assert counted["energy_programming_mj"].value - bare["energy_programming_mj"].value == pytest.approx(
            ((2.0 + 3.0) * 1e-12 * entry_bytes + watts * wall) * J_TO_MJ, rel=1e-9)


class TestTheReviewsSurvivingMutants:
    """N-M2: one named test per mutant of the delta review (Q1 to Q6), each red on its mutant."""

    def test_q1_the_static_leg_covers_the_law_wall_and_the_entries_wall(self):
        """The record fixture's first load: its wall is the write law's wall plus the entries' wall, and its static
        leg is S x that whole wall, computed here from the rows (Q1 left the entries' wall out)."""
        constants = {**BASE, "bytes_per_connectivity_entry": WIDTH}
        bare, _ = _terms(_record_pricing(_record(), constants))
        static, _ = _terms(_record_pricing(_record(), {**constants, **STATIC}))
        law_wall = 7e-9 * RECORD_FIRST_BYTES + 50e-6 * RECORD_FIRST_CORES
        entries_wall = 7e-9 * WIDTH * 3 * CORE_ENTRIES
        assert static["latency_load_s"].value == pytest.approx(law_wall + entries_wall, rel=1e-12)
        watts = 0.25e-3 * RECORD_CORES + 4e-3
        assert static["energy_load_mj"].value - bare["energy_load_mj"].value == pytest.approx(
            watts * (law_wall + entries_wall) * J_TO_MJ, rel=1e-9)

    def test_q2_an_undeclared_transfer_row_of_a_host_owned_store_is_named(self, tmp_path):
        from mimarsinan.deployment_record.platform_physics.hosts import store as ST
        from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
        from unit.deployment_record.registered_hosts import declared_rows, registered
        rows = {"e_link_per_byte": {"nominal": 10.0, "unit": "pJ", "evidence_kind": "estimated", "note": "test"}}
        with registered(tmp_path, "test_q2_link", rows):  # [composition: U-HOST] a host row is a description's own
            physics = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(
                {"host": declared_rows("test_q2_link", rows), "store": ST.store_block(ST.HOST, "test")}))
            terms, _ = _terms(price_absolute(_q(), physics, loads=_loads()))
        for name in ("energy_load_mj", "energy_programming_mj"):
            source = terms[name].source
            assert "store transfer (link): e_link_per_byte" in source, name
            assert "e_host_mem_per_byte(host_memory; the host-owned store's transfer of the load)" in source, name

    def test_q3_the_measured_model_reads_the_resolved_sum_with_the_tabled_buffer(self):
        """The measured model's static leg is the record pricing's resolved sum x total_s: on a record whose
        description tables its carry buffer's static power per capacity row, 0.25 mW x 20 + 4 mW + 50 nW x 65,536
        bytes = 12.2768 mW (Q3, the raw rows, gave 9 mW)."""
        from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
        from unit.deployment_record.record_fixtures import with_partition
        tabled = _physics("p4", {**BASE, **STATIC, "e_carry_buffer_per_byte": 1.0, "area_carry_buffer_per_byte": 1.0},
                          carry_buffer={"rows": [{"bytes": 65536, "p_carry_buffer_static_per_byte":
                                                  {"nominal": 50.0, "unit": "nW", "evidence_kind": "estimated",
                                                   "note": "test"}}]})
        record = _declared(with_partition(_record()), tabled, carry_buffer_bytes=65536)  # [composition: U-HOST]
        model = DeploymentCostModel()
        total_s = find_term(model.latency(record), "total_s").band
        leg = find_term(model.energy(record), "modeled_static_mj").band
        watts = 0.25e-3 * RECORD_CORES + 4e-3 + 50e-9 * 65536
        assert _ends(leg) == pytest.approx([watts * t * J_TO_MJ for t in _ends(total_s)], rel=1e-12)

    def test_q4_an_overlapped_platform_loads_at_the_serial_wall(self):
        overlapped = {**PLATFORM, "write_overlap": {"mode": "overlapped", "shadow_capacity_fraction": 0.5}}
        terms, _ = _terms(price_absolute(_q(), _physics("overlap", {**BASE, **STATIC}), loads=_loads(overlapped)))
        assert terms["latency_load_s"].value == pytest.approx(7e-9 * FIRST_BYTES + 50e-6 * FIRST_CORES, rel=1e-12)

    def test_q5_counted_entries_without_a_width_name_the_width(self):
        terms, _ = _terms(_record_pricing(_record(), {**BASE, **STATIC}))
        assert ("bytes_per_connectivity_entry(connectivity entries; undeclared, so their bytes are unpriced)"
                in terms["energy_load_mj"].source)

    def test_q6_an_aggregate_beside_a_global_static_row_prices_no_static_leg(self):
        physics = _physics("aggregate", {**BASE, "p_static_global": 4.0, "e_synaptic_event_total": 3.0})
        pricing = price_absolute(_q(), physics, loads=_loads())
        terms, _ = _terms(pricing)
        assert pricing.static_power.watts is None and pricing.static_power.in_aggregate
        for wall, energy in (("latency_load_s", "energy_load_mj"), ("latency_programming_s", "energy_programming_mj")):
            assert "static over" not in terms[energy].source, energy
            assert f"held in e_synaptic_event_total over its synaptic events only; not priced over {wall}" in \
                terms[energy].source, energy


class TestTheLinkEndsOfAProgramLoad:
    """N-M2: the link-end naming R5's program load carried (U-J5 L-2), now on the one law's transfer evidence."""

    END = "the chip's own end of the host-chip link"
    EVERY_CORNER = ("boundary: the chip's own end of the host-chip link is unpriced (no row of the vocabulary prices "
                    "it: a crossing byte pays the link and host memory, never the chip's backing-store rows)")

    def _host_owned(self, host="jetson_orin_nano_8gb"):
        from mimarsinan.deployment_record.platform_physics.hosts import store as ST
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
        from mimarsinan.deployment_record.platform_physics.resolve import (
            merge_override_blocks, resolve_platform_physics)
        block = {key: dict(row) for key, row in host_description_block(host).items()}
        return resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(
            {"host": block, "store": ST.store_block(ST.HOST, "test")}))

    def test_a_program_load_over_the_link_names_the_same_corners(self):
        terms, _ = _terms(price_absolute(_q(), self._host_owned(), loads=_loads()))
        for name in ("energy_load_mj", "energy_programming_mj"):
            source = terms[name].source
            assert f"{self.END} is unpriced at the low and nominal corners" in source, source
            assert self.EVERY_CORNER not in source, source

    def test_a_description_without_the_declaration_keeps_the_text_at_every_corner(self, tmp_path):
        from unit.deployment_record.registered_hosts import registered_copy

        def edit(data):  # [composition: U-HOST] other link ends are another description's statement
            data["link_ends"]["rows"].pop("e_link_per_byte", None)
            if not data["link_ends"]["rows"]:
                del data["link_ends"]

        with registered_copy(tmp_path, "jetson_orin_nano_8gb", edit) as host:
            terms, _ = _terms(price_absolute(_q(), self._host_owned(host), loads=_loads()))
        for name in ("energy_load_mj", "energy_programming_mj"):
            assert self.EVERY_CORNER in terms[name].source, name


class TestAStaticPowerOverARefusedLoadWall:
    """m-1: isaac_like as shipped declares a per-unit write wall and no write granularity, so the law refuses its
    load walls; its 10.4 W global static power then refuses both load energies by the wall's reason (R5 priced their
    write part). With a declared cell granularity the walls price, and the static leg is the energy's dominant part."""

    def test_the_load_energies_refuse_by_the_walls_reason(self):
        terms, refused = _terms(price_absolute(_q(), get_platform_physics("isaac_like"), loads=_loads()))
        for wall, energy in (("latency_load_s", "energy_load_mj"), ("latency_programming_s", "energy_programming_mj")):
            assert wall not in terms and energy not in terms
            assert "t_write_per_unit needs the write granularity's units" in refused[wall]
            assert refused[energy].startswith(f"static power is declared but the wall it multiplies ({wall}) is "
                                              "refused, "
                                              "because t_write_per_unit needs the write granularity's units")

    def test_where_the_wall_prices_the_static_leg_dominates_the_load_energy(self):
        from mimarsinan.deployment_record.cost.absolute.walls import loads as L
        physics = get_platform_physics("isaac_like")
        cells = {**PLATFORM, "cores": [{**PLATFORM["cores"][0], "write_granularity": "cell"}]}
        pricing = price_absolute(_q(), physics, loads=_loads(cells))
        load = L.price_load(_loads(cells).first, physics, pricing.static_power, wall_name="latency_load_s")
        assert load.static_leg is not None and load.write_energy is not None
        assert load.static_leg.nominal > 1e3 * load.write_energy.nominal


class TestTheReadersStateTheirSum:
    """m-6."""

    def test_the_load_law_requires_the_static_sum(self):
        from mimarsinan.deployment_record.cost.absolute.walls import loads as L
        with pytest.raises(TypeError):
            L.price_load(_loads().first, _physics("s", BASE), wall_name="latency_load_s")  # pylint: disable=E1120

    def test_the_ladder_keeps_no_second_load_pricer(self):
        from mimarsinan.deployment_record.introspection import scaling_ladder_reading as reading
        assert not hasattr(reading, "tile_writes")

    def test_views_that_differ_only_in_a_zero_by_declaration_are_unequal(self):
        values = {"synaptic_events": QuantityValue(1.0, "static")}
        a = Quantities(values, known_zero={"neurons_used": "the platform names no value domain"})
        assert a != Quantities(values)
        assert a == Quantities(values, known_zero={"neurons_used": "the platform names no value domain"})
