"""[U-ENERGY] One load law: every load is the write law at K = 1, serial, over its census, with the chip's static power
over the load's own wall exactly once; the pricer's load terms and the ladder's once and period columns are one result.

Every constant here is a test constant (the shipped descriptions are read as they ship). The unit's new names are
imported where a test calls them, so a tree without them fails test by test, never at collection."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue


def _loads_module():
    from mimarsinan.deployment_record.cost.absolute.walls import loads
    return loads


def _plane_loads():
    from mimarsinan.deployment_record.quantities.planes import loads
    return loads


def load_inputs(*args, **kwargs):
    return _plane_loads().load_inputs(*args, **kwargs)


def load_census(*args, **kwargs):
    return _plane_loads().load_census(*args, **kwargs)


def price_load(*args, **kwargs):
    return _loads_module().price_load(*args, **kwargs)


def StaticSum(*args, **kwargs):  # noqa: N802
    from mimarsinan.deployment_record.cost.absolute.walls.static_power import StaticSum as static_sum_type
    return static_sum_type(*args, **kwargs)


J_TO_MJ = 1e3
CORNERS = ("low", "nominal", "high")
BITS = 4
PLATFORM = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 8}], "weight_bits": BITS}
BASE = {"e_mac": 0.5, "t_cycle": 100.0, "e_dma_per_byte": 2.0, "e_program_per_byte": 3.0, "e_core_program": 5.0,
        "t_program_per_byte": 7.0, "t_core_program": 50.0}
STATIC = {"p_static_per_core": 0.25, "p_static_global": 4.0}
CORES = 16
#: the hand passes: one resident pass (two cores), two written passes (one and two cores); 4-bit weights
FIRST_BYTES, FIRST_CORES = 20 + 10 + 15 + 8 + 12, 5
PERIOD_BYTES, PERIOD_CORES = 15 + 8 + 12, 3
SHIPPED = ("generic_estimated_22nm", "hermes_64", "isaac_like", "loihi", "neurram", "odin", "sram_cim_macro",
           "study_compass", "study_isaac_like", "study_vr_like", "truenorth")


def _physics(name, constants, **extra):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: ({**v, "evidence_kind": "estimated", "note": "test"} if isinstance(v, dict) else
                          {"nominal": v, "evidence_kind": "estimated", "note": "test"}) for k, v in constants.items()},
        **extra})


def _passes(extents=True, entries=False):
    # ``entries``: each core's connectivity entries on the rows (a sealed record states them per core): 6 on the
    # resident pass, 2 and 2 on the written passes, so the first load counts 10 and the period (two written passes
    # rotate) counts 4
    def row(i, resident, cells, ext, ent):
        extra = {"core_entries": ent} if entries else {}
        return PassRow(segment_index=i, resident=resident, core_cells=cells, marks=frozenset(),
                       core_extents=ext if extents else None, compute_steps=3, **extra)
    return ProgramPasses(rows=(row(0, True, (40, 20), ((8, 5), (4, 5)), (3, 3)),
                               row(1, False, (30,), ((6, 5),), (2,)),
                               row(2, False, (16, 24), ((4, 4), (6, 4)), (1, 1))))


def _loads(entries=False, **kw):
    return load_inputs(_passes(entries=entries), PLATFORM, weight_bits=BITS, label="test", **kw)


def _q(**extra):
    values = {"synaptic_events": 1e6, "cores_physical": CORES, "latency_steps": 40.0, "host_macs": 0.0,
              "compute_op_count": 0.0, **extra}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _priced(physics, quantities=None, **kw):
    pricing = price_absolute(quantities or _q(), physics, **kw)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _ends(band):
    return [band.low, band.nominal, band.high]


def _watts(physics, cores=CORES):
    return [getattr(physics.band("p_static_per_core"), c) * cores + getattr(physics.band("p_static_global"), c)
            for c in CORNERS]


class TestOneLawOverItsCensus:
    def test_the_first_load_is_the_write_law_over_every_core_with_one_static_leg(self):
        physics = _physics("first", {**BASE, **STATIC})
        terms, _ = _priced(physics, loads=_loads())
        wall = 7e-9 * FIRST_BYTES + 50e-6 * FIRST_CORES  # t_core_program in us
        assert terms["latency_load_s"].value == pytest.approx(wall, rel=1e-12)
        dynamic = (2.0 + 3.0) * 1e-12 * FIRST_BYTES + 5.0 * 1e-12 * FIRST_CORES
        assert terms["energy_load_mj"].value == pytest.approx((dynamic + _watts(physics)[1] * wall) * J_TO_MJ,
                                                              rel=1e-12)
        assert terms["energy_load_mj"].source.count("static over latency_load_s") == 1

    def test_the_reprogramming_per_round_is_the_same_law_over_the_period_census(self):
        physics = _physics("period", {**BASE, **STATIC})
        terms, _ = _priced(physics, loads=_loads())
        wall = 7e-9 * PERIOD_BYTES + 50e-6 * PERIOD_CORES
        assert terms["latency_programming_s"].value == pytest.approx(wall, rel=1e-12)
        dynamic = (2.0 + 3.0) * 1e-12 * PERIOD_BYTES + 5.0 * 1e-12 * PERIOD_CORES
        assert terms["energy_programming_mj"].value == pytest.approx(
            (dynamic + _watts(physics)[1] * wall) * J_TO_MJ, rel=1e-12)

    def test_the_load_reads_the_cell_commit_where_the_chip_commits_by_cell(self):
        cells = {**BASE, "e_write_per_cell": 0.2, "write_verify_iterations": 3.0}
        terms, _ = _priced(_physics("cells", cells), loads=_loads())
        first_cells = 40 + 20 + 30 + 16 + 24
        dynamic = 2.0e-12 * FIRST_BYTES + 0.2e-12 * 3.0 * first_cells + 5.0e-12 * FIRST_CORES
        assert terms["energy_load_mj"].value == pytest.approx(dynamic * J_TO_MJ, rel=1e-12)

    def test_the_static_leg_is_priced_at_each_corner(self):
        banded = _physics("banded", {**BASE, "p_static_per_core": {"low": 0.1, "nominal": 0.25, "high": 1.0},
                                     "p_static_global": {"low": 1.0, "nominal": 4.0, "high": 9.0},
                                     "t_program_per_byte": {"low": 5.0, "nominal": 7.0, "high": 11.0}})
        bare = _physics("bare", {**BASE, "t_program_per_byte": {"low": 5.0, "nominal": 7.0, "high": 11.0}})
        a, _ = _priced(bare, loads=_loads())
        b, _ = _priced(banded, loads=_loads())
        for wall, energy in (("latency_load_s", "energy_load_mj"), ("latency_programming_s", "energy_programming_mj")):
            leg = [w * t * J_TO_MJ for w, t in zip(_watts(banded), _ends(b[wall].band))]
            got = [y - x for x, y in zip(_ends(a[energy].band), _ends(b[energy].band))]
            assert got == pytest.approx(leg, rel=1e-12), energy
            assert leg[0] < leg[1] < leg[2]

    def test_without_static_rows_the_load_names_its_static_leg(self):
        terms, _ = _priced(_physics("no_static", BASE), loads=_loads())
        for wall, energy in (("latency_load_s", "energy_load_mj"), ("latency_programming_s", "energy_programming_mj")):
            assert f"p_static_per_core, p_static_global(static power over {wall}; undeclared)" in terms[energy].source

    def test_a_cold_round_and_a_warm_round_are_two_figures(self):
        terms, _ = _priced(_physics("rounds", {**BASE, **STATIC}), loads=_loads())
        assert terms["energy_load_mj"].value > terms["energy_programming_mj"].value
        assert "energy_load_mj" not in terms["energy_per_inference_mj"].source


class TestThePricerAndTheLadderReadOneResult:
    """On every shipped description the first load (and the reprogramming per round) priced by price_absolute equal
    the ladder's once (and period) column, or both refuse by one reason, and both equal the write law called alone
    over the one census rule plus the description's own static rows times the law's wall (an independent value)."""

    @pytest.mark.parametrize("profile", SHIPPED)
    def test_the_first_load_and_the_once_column_are_one_result(self, profile):
        import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
        from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
        from unit.deployment_record.introspection.test_scaling_ladder import _facts, _network, _platform
        network = _network(4)
        point = {**_platform(8), "inference_batch_K": 16,
                 "platform_physics_resolved": get_platform_physics(profile).to_dict()}
        census = ladder.shape_census(network, point)
        scheduled = ladder.schedule_program(network, census, point, _facts)
        priced = ladder.price_scheduled(network, census, scheduled, point)
        program, refused = priced.terms["program"], priced.refusals["program"]
        physics = get_platform_physics(profile)
        cores = sum(int(c.get("count", 1)) for c in point["cores"])  # the point's declared physical cores
        static = price_absolute(_q(cores_physical=cores), physics).static_power
        rows = [] if physics.has("e_synaptic_event_total") else (
            [[getattr(physics.band("p_static_per_core"), c) * cores for c in CORNERS]]
            if physics.has("p_static_per_core") else []) + (
            [[getattr(physics.band("p_static_global"), c) for c in CORNERS]]
            if physics.has("p_static_global") else [])
        watts = [sum(r[i] for r in rows) for i in range(3)] if rows else None
        for column, census_kind, wall, energy in (("once", True, "latency_load_s", "energy_load_mj"),
                                                  ("period", False, "latency_programming_s",
                                                   "energy_programming_mj")):
            census = load_census(scheduled.facts.passes, point, weight_bits=point["weight_bits"],
                                 every_core=census_kind, label="candidate")
            alone = price_load(census, physics, static if static is not None else StaticSum(), wall_name=wall)
            law = price_write_inputs(WriteInputs.of(census.writes), physics)
            if law.energy is None or (watts is not None and law.exposed is None):
                independent = None
            else:
                independent = [getattr(law.energy, c) + (0.0 if watts is None else
                                                         watts[i] * getattr(law.exposed, c) * J_TO_MJ)
                               for i, c in enumerate(CORNERS)]
            got, got_refused = priced.terms[column], priced.refusals[column]
            for key, name, mine in (("energy_load_mj", energy, alone.energy),
                                    ("latency_write_exposed_s", wall, alone.wall)):
                if name in program:
                    assert _ends(got[key]) == _ends(program[name]), (column, name)
                    assert mine is not None, (column, name)
                    if key == "latency_write_exposed_s":
                        assert _ends(mine) == pytest.approx(_ends(program[name]), rel=1e-12)
                        assert _ends(program[name]) == pytest.approx(_ends(law.exposed), rel=1e-12)
                    else:  # the energy against the independent value, corner by corner, never against itself
                        assert independent is not None, (column, name)
                        assert _ends(program[name]) == pytest.approx(independent, rel=1e-9), (column, name)
                        assert _ends(mine) == pytest.approx(independent, rel=1e-9), (column, name)
                else:
                    assert key not in got and got_refused[key] == refused[name], (column, name)
                    assert mine is None, (column, name)
                    if key == "energy_load_mj":
                        assert independent is None, (column, name)


class TestEveryLoadRowOverAnUnproducedCensus:
    """One test per load row over a census the plane does not produce: named with the plane's reason beside the
    figure, or the figure refused by that reason where every declared part of it is unpriced."""

    @pytest.mark.parametrize("row", ("e_dma_per_byte", "e_program_per_byte", "e_core_program", "t_program_per_byte",
                                     "t_core_program"))
    def test_a_plane_without_passes_refuses_every_load_by_its_reason(self, row):
        physics = _physics("rows", {"e_mac": 0.5, "t_cycle": 100.0, row: 3.0})
        terms, refused = _priced(physics, _q(reprogrammed_bytes=4096.0, reprogrammed_cores=6.0),
                                 loads=load_inputs(None, PLATFORM, weight_bits=BITS, label="scalar"))
        for name in ("latency_load_s", "energy_load_mj", "latency_programming_s", "energy_programming_mj"):
            assert name not in terms
            assert "the scalar plane states no pass structure" in refused[name]

    def test_no_load_census_given_refuses_by_name_never_zero(self):
        terms, refused = _priced(_physics("none", {**BASE, **STATIC}), _q(reprogrammed_bytes=4096.0,
                                                                          reprogrammed_cores=6.0))
        for name in ("latency_load_s", "energy_load_mj", "latency_programming_s", "energy_programming_mj"):
            assert name not in terms and refused[name] == _loads_module().NO_LOAD_CENSUS

    def test_t_write_per_unit_over_a_census_without_extents_refuses_the_wall_by_the_laws_reason(self):
        physics = _physics("units", {**BASE, "t_write_per_unit": 9.0, "write_verify_iterations": 1.0,
                                     **STATIC})
        platform = {**PLATFORM, "cores": [{**PLATFORM["cores"][0], "write_granularity": "row"}]}
        loads = load_inputs(_passes(extents=False), platform, weight_bits=BITS, label="test")
        terms, refused = _priced(physics, loads=loads)
        assert "latency_load_s" not in terms and "extents" in refused["latency_load_s"]
        assert "energy_load_mj" not in terms
        assert ("static power is declared but the wall it multiplies (latency_load_s) is refused"
                in refused["energy_load_mj"])

    def test_bytes_per_connectivity_entry_over_a_plane_that_counts_no_entries_is_named(self):
        physics = _physics("entries", {**BASE, "bytes_per_connectivity_entry": 8.0})
        reason = _plane_loads().CANDIDATE_NO_ENTRIES
        terms, _ = _priced(physics, loads=_loads(entries_absent=reason))
        assert ("bytes_per_connectivity_entry(connectivity; not produced on this plane: " + reason
                in terms["energy_load_mj"].source)

    def test_connectivity_entries_the_plane_counts_are_priced_on_the_byte_rows(self):
        # the entries ride the rows; each census counts those of the cores it writes (10 and 4)
        physics = _physics("entries", {**BASE, "bytes_per_connectivity_entry": 8.0})
        bare, _ = _priced(physics, loads=_loads(entries_absent="test: no entries stated"))
        counted, _ = _priced(physics, loads=_loads(entries=True))
        assert counted["energy_load_mj"].value - bare["energy_load_mj"].value == pytest.approx(
            (2.0 + 3.0) * 1e-12 * 80.0 * J_TO_MJ, rel=1e-9)
        assert counted["latency_programming_s"].value - bare["latency_programming_s"].value == pytest.approx(
            7e-9 * 32.0, rel=1e-9)

    def test_p_static_per_core_over_unproduced_cores_is_named_with_the_plane_reason(self):
        values = {"synaptic_events": 1e6, "latency_steps": 40.0, "host_macs": 0.0, "compute_op_count": 0.0}
        q = Quantities({k: QuantityValue(v, "static") for k, v in values.items()},
                       unproduced={"cores_physical": "the platform declares no cores"})
        terms, _ = _priced(_physics("percore", {**BASE, **STATIC}), q, loads=_loads())
        assert ("p_static_per_core(static; not produced on this plane: the platform declares no cores)"
                in terms["energy_load_mj"].source)

    def test_a_cell_commit_without_its_iterations_refuses_the_load_energy_by_the_laws_reason(self):
        physics = _physics("iter", {"e_mac": 0.5, "t_cycle": 100.0, "e_write_per_cell": 0.2, "t_program_per_byte": 7.0})
        terms, refused = _priced(physics, loads=_loads())
        assert "energy_load_mj" not in terms and "write_verify_iterations" in refused["energy_load_mj"]
        assert "latency_load_s" in terms


class TestAHostOwnedStore:
    def test_the_load_pays_the_transfer_of_its_bytes_on_the_host_owned_store(self):
        from mimarsinan.deployment_record.platform_physics.hosts import store as ST
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
        from mimarsinan.deployment_record.platform_physics.resolve import (
            merge_override_blocks, resolve_platform_physics)
        blocks = {"host": host_description_block("jetson_orin_nano_8gb"), "store": ST.store_block(ST.HOST, "test")}
        physics = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(blocks))
        load = price_load(_loads().first, physics, StaticSum(), wall_name="latency_load_s")
        per_byte = physics.band("e_link_per_byte").nominal + physics.band("e_host_mem_per_byte").nominal
        assert load.energy.nominal - load.write_energy.nominal == pytest.approx(per_byte * FIRST_BYTES * J_TO_MJ,
                                                                                rel=1e-9)


class TestThePlanesPassTheirCensus:
    """The planes that state passes hand the one law their census: a sealed record (with the connectivity entries it
    seals per core) and a candidate view's report."""

    def test_the_record_plane_prices_its_loads_with_its_entries(self):
        from dataclasses import replace
        from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
        from unit.deployment_record.record_fixtures import make_full_record
        record = make_full_record()
        physics = _physics("record", {**BASE, **STATIC, "bytes_per_connectivity_entry": 8.0})
        declared = replace(record, identity=replace(record.identity, platform={
            **record.identity.platform, "platform_physics_resolved": physics.to_dict()}))
        terms = {t.name: t for t in absolute_pricing_for_record(declared).terms}
        for name in ("latency_load_s", "energy_load_mj", "latency_programming_s", "energy_programming_mj"):
            assert name in terms, name
        assert "not produced on this plane" not in terms["energy_load_mj"].source
        assert terms["energy_load_mj"].source.count("static over latency_load_s") == 1

    def test_a_candidate_views_report_prices_its_loads(self):
        import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
        from mimarsinan.deployment_record.introspection.scaling_ladder_lanes import host_census
        from mimarsinan.deployment_record.objectives import CandidateStaticView, declared_core_capacity
        from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
        from mimarsinan.deployment_record.platform_physics import PlatformPhysics
        from unit.deployment_record.introspection.test_scaling_ladder import _facts, _network, _platform
        network = _network(4)
        point = _platform(8)
        census = ladder.shape_census(network, point)
        scheduled = ladder.schedule_program(network, census, point, _facts)
        facts, host = scheduled.facts, host_census(network, census)
        context = candidate_context_from_platform(
            point, latency_steps=facts.latency_steps, programming=facts.programming, carry=facts.carry,
            passes=facts.passes, **{k: host.get(k) for k in ("host_macs", "onchip_macs", "host_params",
                                                              "onchip_params", "compute_op_count")})
        view = CandidateStaticView(layout=scheduled.stats, chip_param_capacity=declared_core_capacity(dict(point)),
                                   total_params=network.total_params, host_side_segment_count=census.host_segments,
                                   physics=PlatformPhysics.from_dict(point["platform_physics_resolved"]),
                                   quantity_context=context, noc_fragments=scheduled.noc)
        names = {t.name for t in view.cost_report().energy} | {t.name for t in view.cost_report().latency}
        assert {"energy_load_mj", "latency_load_s", "energy_programming_mj", "latency_programming_s"} <= names
