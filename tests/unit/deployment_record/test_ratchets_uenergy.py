"""[U-ENERGY W2] The unit's two ratchets, each over its whole class and each shown red on a planted defect of its class.

T-E1a, the time ratchet keyed on the time dimension over every package: every site of the toolchain's source that
produces a time (``time_sites``) has exactly one static rule in ``TIME_RULES``, keyed by its module and what it
produces; a new site fails until it is classified and a removed one fails until its entry goes. At run time, every
priced time of every pricing (the eleven shipped descriptions on the record plane with its loads and its writes at a
declared batch, and on the candidate plane's hand point) is the per-inference wall, a summand of it under its label
(``metrics.E2E_SUMMANDS``), or the wall of exactly one load's static leg (``walls.loads.LOAD_FIGURES``).

T-E2a, the four-state ratchet over views of every plane: every site that builds a view a pricer reads
(``view_sites``) is classified in ``VIEW_SITES``; on the views every plane builds (candidate, record, level), every
absent factor of every component table is in exactly one absent state with its reason and every declared row over it
is named or printed as a zero by declaration (``test_no_silent_zero_uenergy._check_view``); the views derived from a
plane's view keep its states; a view a caller states by hand names every declared row over what it does not state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.metrics import E2E_SUMMANDS, LATENCY_HEADLINE
from mimarsinan.deployment_record.cost.absolute.walls.loads import LOAD_FIGURES
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
from mimarsinan.deployment_record.cost.host_terms import LATENCY_LINK
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS

from unit.deployment_record.time_sites import catalog_sites, source_sites, stale, unclassified
from unit.deployment_record.view_sites import view_sites

ROOT = Path(mimarsinan.__file__).resolve().parent

# ---------------------------------------------------------------------------------------------------- T-E1a, static
E2E = "the per-inference wall: energy_static_mj is the static leg over it"
SUMMAND = "a summand of e2e_latency_s (metrics.E2E_SUMMANDS): e2e's leg covers it, no leg of its own"
LOAD_WALL = "the wall of exactly one load's static leg (walls.loads.LOAD_FIGURES)"
MEASURED_WALL = "the measured model's whole latency: modeled_static_mj is the static leg over it (walls.measured)"
MEASURED_PART = "a part of the measured model's total_s: total_s's leg covers it, no leg of its own"
WRITER = ("a term writer given its name and unit: each priced time it writes is held to its rule at run time "
          "(the runtime ratchet over pricings)")
READING = ("a reading of a priced wall (the wall in another record, a per-inference period, a bound or an inverse): "
           "no chip time of its own, its wall's leg covers it")
NOT_CHIP = ("not a chip time: a process wall clock, a rate, a declared configuration value, a host calibration value, "
            "a simulator's preset or output, or another tool's measured run; never priced as chip time")
RULES = (E2E, SUMMAND, LOAD_WALL, MEASURED_WALL, MEASURED_PART, WRITER, READING, NOT_CHIP)

DR = "deployment_record/"
_PRESETS = ("axon_in", "axon_out", "dendrite", "soma_access", "soma_spike_out", "soma_update", "synapse", "tile_hop")
#: Every time site of the source, by key, with its one rule (the 52 sites of design revision 2's section 4.8 among
#: them, keyed by module now that the scan covers every package, and the end-to-end summands).
TIME_RULES = {
    **{f"chip_simulation/sanafe/presets.py:{p}_latency_s": NOT_CHIP for p in _PRESETS},
    "certification/program_check/report.py:compile_wall_s": NOT_CHIP,
    "certification/program_check/report.py:wall_s": NOT_CHIP,
    "certification/randomized/screen.py:median_wall_s": NOT_CHIP,
    "chip_simulation/cost_extraction.py:wall_s": NOT_CHIP,
    "chip_simulation/ledger_schema.py:wall_s": NOT_CHIP,
    "chip_simulation/odin_fpga/records.py:execution_s": NOT_CHIP,
    "chip_simulation/odin_fpga/records.py:programming_s": NOT_CHIP,
    "chip_simulation/odin_fpga/records.py:total_s": NOT_CHIP,
    "chip_simulation/sanafe/stats.py:max_sim_time_s": NOT_CHIP,
    "chip_simulation/sanafe/stats.py:sim_time_s": NOT_CHIP,
    "config_schema/defaults.py:allow_per_layer_s": NOT_CHIP,
    "config_schema/registry/decision_classes.py:allow_per_layer_s": NOT_CHIP,
    DR + "cost/absolute/context.py:PricingContext.term:name [unit unit]": WRITER,
    DR + "cost/absolute/metrics.py:latency_host_s": SUMMAND,
    DR + "cost/absolute/metrics.py:price_latency:LATENCY_HEADLINE": E2E,
    DR + "cost/absolute/metrics.py:price_latency:LATENCY_LINK": SUMMAND,
    DR + "cost/absolute/walls/loads.py:_put:name [unit unit]": WRITER,
    DR + "cost/absolute/walls/loads.py:price_loads:wall_name": LOAD_WALL,
    DR + "cost/absolute/writes.py:_price_technologies:LATENCY_WRITE": SUMMAND,
    DR + "cost/absolute/writes.py:price_write_terms:LATENCY_WRITE": SUMMAND,
    DR + "cost/combine.py:derived:name [unit unit]": WRITER,
    DR + "cost/combine.py:measured:name [unit unit]": WRITER,
    DR + "cost/combine.py:modeled:name [unit unit]": WRITER,
    DR + "cost/combine.py:host_ops_term:HOST_WALL_DIAGNOSTIC": NOT_CHIP,  # [composition: U-HOST] a diagnostic
    DR + "cost/legacy_projection.py:wall_s": NOT_CHIP,
    DR + "cost/model.py:compute_s": MEASURED_PART,
    DR + "cost/model.py:core_init_s": MEASURED_PART,
    DR + "cost/model.py:programming_s": MEASURED_PART,
    DR + "cost/model.py:sync_s": MEASURED_PART,
    DR + "cost/model.py:total_s": MEASURED_WALL,
    DR + "cost/segments.py:programming_time_s": MEASURED_PART,
    DR + "cost/segments.py:reset_time_s": MEASURED_PART,
    DR + "cost/write_lifetime.py:WritePricing.term:name [unit unit]": WRITER,
    DR + "cost/write_lifetime.py:join_lifetimes:name [unit bands[0][1]]": WRITER,
    **{f"{DR}design_point/pricing.py:{f}": READING for f in (
        "e2e_latency_s", "latency_s", "latency_without_writes_s", "request_latency_s", "throughput_inferences_per_s",
        "time_per_inference_s")},
    **{f"{DR}design_point/schema.py:{f}": READING for f in (
        "latency_s", "latency_without_writes_s", "request_latency_s", "throughput_inferences_per_s",
        "time_per_inference_s")},
    DR + "design_point/schema.py:latency_write_exposed_s": SUMMAND,
    DR + "design_point/schema.py:build_wall_s": NOT_CHIP,
    DR + "design_point/schema.py:wall_s": NOT_CHIP,
    DR + "fidelity_build.py:e2e_latency_s": READING,
    DR + "fidelity_build.py:throughput_inferences_s": READING,
    DR + "introspection/scaling_ladder.py:layout_wall_s": NOT_CHIP,
    DR + "introspection/scaling_ladder.py:point_wall_s": NOT_CHIP,
    DR + "introspection/scaling_ladder_law.py:layer_pipelined_bound_s": READING,
    DR + "introspection/scaling_ladder_reading.py:time_per_inference_s": READING,
    DR + "platform_physics/constants.py:_latency_s": NOT_CHIP,
    DR + "platform_physics/host_calibration.py:host_macs_per_s": NOT_CHIP,
    DR + "platform_physics/host_calibration.py:t_host_op_overhead_s": NOT_CHIP,
    DR + "quantities/from_record.py:host_ops_s": NOT_CHIP,  # [composition: U-HOST] its withheld diagnostic
    DR + "quantities/planes/states.py:host_ops_s": NOT_CHIP,
    "gui/viewmodel/gantt_vm.py:artifact_wall_s": NOT_CHIP,
    "gui/viewmodel/gantt_vm.py:offset_s": NOT_CHIP,
    "gui/viewmodel/gantt_vm.py:total_wall_s": NOT_CHIP,
    "gui/viewmodel/gantt_vm.py:wall_s": NOT_CHIP,
    "gui/wizard/physics_panel.py:e2e_latency_s": READING,
    "gui/wizard/physics_panel.py:throughput_inferences_s": READING,
    "gui/wizard/schema.py:e2e_latency_s": READING,
    "gui/wizard/schema.py:host_op_wall_s": NOT_CHIP,
    "gui/wizard/schema.py:throughput_inferences_s": READING,
    "gui/wizard/schema.py:throughput_samples_per_s": NOT_CHIP,
    "pipelining/core/engine/step_instrumentation.py:wall_s": NOT_CHIP,
    **{f"pipelining/pipeline_steps/verification/odin/odin_fpga_simulation_step.py:{f}": NOT_CHIP for f in (
        "execution_s", "host_ops_s", "programming_s", "segment_boundary_init_s")},
    "pipelining/pipeline_steps/verification/sanafe_simulation_step.py:max_sim_time_s": NOT_CHIP,
    "quantity:host_ops_s": NOT_CHIP,  # [composition: U-HOST] a diagnostic no pricer reads
    "search/optimizers/budget.py:wall_s": NOT_CHIP,
    "tuning/orchestration/ft_pass_wall.py:wall_s": NOT_CHIP,
    "tuning/orchestration/run_instrumentation.py:max_ft_pass_wall_s": NOT_CHIP,
    "tuning/orchestration/run_instrumentation.py:wall_s": NOT_CHIP,
    "tuning/orchestration/smooth_adaptation_cycle.py:max_ft_pass_wall_s": NOT_CHIP,
}

#: Each site whose rule names a table of the pricer, with the name it produces (checked against that table).
_NAMED = {DR + "cost/absolute/metrics.py:latency_host_s": "latency_host_s",
          DR + "cost/absolute/metrics.py:price_latency:LATENCY_LINK": LATENCY_LINK,
          DR + "cost/absolute/writes.py:_price_technologies:LATENCY_WRITE": "latency_write_exposed_s",
          DR + "cost/absolute/writes.py:price_write_terms:LATENCY_WRITE": "latency_write_exposed_s",
          DR + "design_point/schema.py:latency_write_exposed_s": "latency_write_exposed_s"}

#: [m-3] The write law's wall (``latency_write_exposed_s``) is one site read by three callers, each with its own rule:
#: a summand of the per-inference wall, the wall of a load's leg, and the step price's own wall.
STEP_WALL = "the step price's exposed write time: its own static leg by the one leg form (step_choice.price_step)"
WALL_CALLERS = {key: (("cost/absolute/metrics.py", "price_latency", SUMMAND),
                      ("cost/absolute/walls/loads.py", "price_load", LOAD_WALL),
                      ("cost/absolute/step_choice.py", "price_step", STEP_WALL))
                for key in (DR + "cost/absolute/writes.py:_price_technologies:LATENCY_WRITE",
                            DR + "cost/absolute/writes.py:price_write_terms:LATENCY_WRITE")}


def _time_sites(extra=()):
    return source_sites(ROOT, extra) + catalog_sites(QUANTITY_SPECS)


class TestTheTimeRatchetOverEveryPackage:
    def test_every_time_site_has_exactly_one_static_rule(self):
        sites = _time_sites()
        assert not unclassified(sites, {k: (v, "") for k, v in TIME_RULES.items()}), [
            (s.key, s.where) for s in unclassified(sites, {k: (v, "") for k, v in TIME_RULES.items()})]
        assert not stale(sites, {k: (v, "") for k, v in TIME_RULES.items()})
        assert set(TIME_RULES.values()) <= set(RULES)

    def test_the_summands_are_the_pricers_own_table(self):
        summands = dict(E2E_SUMMANDS)
        assert LATENCY_HEADLINE == "e2e_latency_s"
        for key, name in _NAMED.items():
            assert TIME_RULES[key] == SUMMAND and name in summands, key
        assert {wall for _kind, wall, _energy, _leg in LOAD_FIGURES} == {"latency_load_s", "latency_programming_s"}

    def test_the_write_laws_wall_states_its_three_callers(self):
        """[m-3] The site's per-inference rule is the table's; its two other callers each read the write law's wall
        and multiply it by the static sum only through the one leg form."""
        import ast
        for key, callers in WALL_CALLERS.items():
            assert TIME_RULES[key] == callers[0][2] == SUMMAND, key
            for module, function, rule in callers:
                tree = ast.parse((ROOT / "deployment_record" / module).read_text())
                body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
                names = {n.id for n in ast.walk(body) if isinstance(n, ast.Name)} | {
                    n.attr for n in ast.walk(body) if isinstance(n, ast.Attribute)}
                assert names & {"price_write_inputs", "price_write_terms", "exposed"}, (module, function)
                if rule != SUMMAND:
                    assert "static_leg" in names and "watts" not in names, (module, function)

    def test_a_planted_time_site_is_red(self):
        planted = ("deployment_record/cost/absolute/planted.py",
                   'def price_planted(ctx, band):\n    ctx.term("latency_planted_s", "s", band, "planted")\n')
        assert [s.key for s in unclassified(_time_sites([planted]), {k: (v, "") for k, v in TIME_RULES.items()})] == [
            "deployment_record/cost/absolute/planted.py:latency_planted_s"]

    def test_a_removed_site_is_red(self):
        sites = [s for s in _time_sites() if s.key != DR + "cost/model.py:total_s"]
        assert stale(sites, {k: (v, "") for k, v in TIME_RULES.items()}) == [DR + "cost/model.py:total_s"]


# --------------------------------------------------------------------------------------------------- T-E1a, runtime
def time_rule_failures(terms):
    """Every priced time of one pricing held to its rule: (name, why) for each that is not."""
    by_name = {t.name: t for t in terms}
    summands = dict(E2E_SUMMANDS)
    load_walls = {wall: energy for _kind, wall, energy, _leg in LOAD_FIGURES}
    failures = []
    for term in terms:
        if term.unit != "s":
            continue
        legs = [t.name for t in terms if f"static over {term.name}" in t.source]
        if term.name == LATENCY_HEADLINE:
            if "energy_static_mj" in by_name and "e2e_latency_s" not in by_name["energy_static_mj"].source:
                failures.append((term.name, "its leg does not name it"))
        elif term.name in summands:
            e2e = by_name.get(LATENCY_HEADLINE)  # a refused e2e carries no summand: only the leg rule holds
            if e2e is not None and e2e.source.count(summands[term.name] + term.band.basis) != 1:
                failures.append((term.name, "not a summand of e2e under its label exactly once"))
            if legs:
                failures.append((term.name, f"a summand with a leg of its own: {legs}"))
        elif term.name in load_walls:
            if legs not in ([], [load_walls[term.name]]) or (by_name.get("energy_static_mj") and load_walls[
                    term.name] in by_name and not legs):
                failures.append((term.name, f"not the wall of exactly one load's leg: {legs}"))
        else:
            failures.append((term.name, "a priced time with no static rule"))
    return failures


def _record_pricings():
    from mimarsinan.deployment_record.quantities.from_record import from_record
    from mimarsinan.deployment_record.quantities.planes.loads import record_load_inputs
    from unit.deployment_record.record_fixtures import make_rewritten_record
    from unit.deployment_record.load_census_fixtures import written_program
    record = make_rewritten_record()
    loads = record_load_inputs(record)
    period = written_program((250, 250), (250,), granularity="cell").period.writes
    writes = WriteInputs.of(replace(period, policy=replace(period.policy, batch_k=4)))
    for name in available_profiles():
        physics = get_platform_physics(name)
        yield name, price_absolute(from_record(record), physics, loads=loads, writes=writes).terms
    # the generic projection with an edge host and the chip's store (the prototype's pricing): every summand priced
    from unit.deployment_record.test_static_every_wall_uenergy import TestEveryPricedTimeIsCoveredOnce as Hosted
    yield "generic with an edge host", tuple(Hosted()._pricing().values())


class TestEveryPricedTimeAtRunTime:
    def test_every_priced_time_of_every_shipped_description_has_its_rule(self):
        seen = set()
        for name, terms in _record_pricings():
            assert not time_rule_failures(terms), (name, time_rule_failures(terms))
            seen.update(t.name for t in terms if t.unit == "s")
        assert {"e2e_latency_s", "latency_write_exposed_s", "latency_load_s", "latency_programming_s", "latency_host_s",
                "latency_host_link_s"} <= seen, seen

    def test_a_planted_time_without_a_rule_is_red(self):
        from mimarsinan.deployment_record.cost.terms import CostTerm
        _name, terms = next(_record_pricings())
        planted = tuple(terms) + (CostTerm("latency_planted_s", "s", 1.0, None, "derived", "planted"),)
        assert ("latency_planted_s", "a priced time with no static rule") in time_rule_failures(planted)

    def test_a_planted_leg_over_a_summand_is_red(self):
        from mimarsinan.deployment_record.cost.terms import CostTerm
        _name, terms = next(_record_pricings())
        assert any(t.name == "latency_write_exposed_s" for t in terms)
        planted = tuple(terms) + (CostTerm("energy_planted_mj", "mJ", 1.0, None, "derived",
                                           "static over latency_write_exposed_s: planted"),)
        assert any(n == "latency_write_exposed_s" and "leg of its own" in why
                   for n, why in time_rule_failures(planted))


# ---------------------------------------------------------------------------------------------------- T-E2a, static
PLANE = "a plane's view in the four states (built by ``planes.states.plane_view``), checked on the views it builds"
FOUR_STATE = "the planes' one four-state constructor"
DERIVED = "a view derived from a plane's view that keeps every state of it (checked on every plane's view)"
BY_HAND = ("a view a caller states by hand (a published case's census, a chip's facts, a step's carry): every key it "
           "does not state is not produced, each declared row over it named by the pricer ('the view states no ...')")
PROBE = "every catalog key at a placeholder: a capability question, never a priced value"
#: Every site that builds a view a pricer reads, by module and function, with its rule.
VIEW_SITES = {
    DR + "quantities/from_candidate.py:from_candidate": PLANE,
    DR + "quantities/from_record.py:from_record": PLANE,
    DR + "quantities/from_level.py:from_level": PLANE,
    DR + "quantities/planes/states.py:plane_view": FOUR_STATE,
    DR + "cost/absolute/host_split.py:share_pricings": DERIVED,
    DR + "cost/absolute/tables/components.py:with_derived": DERIVED,
    DR + "correlation/run.py:correlate": BY_HAND,
    DR + "cost/absolute/step_choice.py:carry_energy": BY_HAND,
    DR + "quantities/from_level.py:lane_area_quantities": BY_HAND,
    DR + "quantities/probe.py:probe_quantities": PROBE,
}


class TestTheViewRatchetOverEveryPlane:
    def test_every_view_site_is_classified(self):
        sites = view_sites(ROOT)
        table = {k: (v, "") for k, v in VIEW_SITES.items()}
        assert not unclassified(sites, table), [(s.key, s.where) for s in unclassified(sites, table)]
        assert not stale(sites, table)

    def test_a_planted_view_site_is_red(self):
        planted = (DR + "quantities/from_planted.py", "def from_planted(values):\n    return Quantities(values)\n")
        table = {k: (v, "") for k, v in VIEW_SITES.items()}
        assert [s.key for s in unclassified(view_sites(ROOT, [planted]), table)] == [
            DR + "quantities/from_planted.py:from_planted"]

    def test_each_plane_builds_its_view_with_the_four_state_constructor(self):
        import importlib
        import inspect

        package = "mimarsinan.deployment_record.quantities."
        for function in ("from_candidate", "from_record", "from_level"):
            module = importlib.import_module(package + function)
            assert "plane_view(" in inspect.getsource(getattr(module, function)), function


# --------------------------------------------------------------------------------------------------- T-E2a, runtime
def _candidate_view():
    import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
    from mimarsinan.deployment_record.introspection.scaling_ladder_lanes import host_census
    from mimarsinan.deployment_record.objectives import CandidateStaticView, declared_core_capacity
    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
    from mimarsinan.deployment_record.quantities.planes.states import declared_zeros
    from unit.deployment_record.introspection.test_scaling_ladder import _facts, _network, _platform
    network = _network(4)
    point = {**_platform(8), "inference_batch_K": 16}
    census = ladder.shape_census(network, point)
    scheduled = ladder.schedule_program(network, census, point, _facts)
    facts, host = scheduled.facts, host_census(network, census)
    context = candidate_context_from_platform(
        point, latency_steps=facts.latency_steps, programming=facts.programming, carry=facts.carry,
        passes=facts.passes, **{k: host.get(k) for k in ("host_macs", "onchip_macs", "host_params",
                                                          "onchip_params", "compute_op_count")})
    view = CandidateStaticView(layout=scheduled.stats, chip_param_capacity=declared_core_capacity(dict(point)),
                               total_params=network.total_params, host_side_segment_count=census.host_segments,
                               physics=None, quantity_context=context, noc_fragments=scheduled.noc).quantities
    return view, declared_zeros(context.output_stage is not None, context.carry_buffer_capacity_bytes is not None,
                                context.value_rule is not None)


def _record_views():
    from mimarsinan.deployment_record.quantities.from_record import from_record
    from mimarsinan.deployment_record.quantities.planes.states import declared_zeros
    from mimarsinan.deployment_record.quantities.value_domain import declared_buffer_bytes, names_value_domain
    from unit.deployment_record.record_fixtures import make_full_record, make_rewritten_record
    for record in (make_full_record(), make_rewritten_record()):
        platform = record.identity.platform
        yield from_record(record), declared_zeros(record.output_stage is not None,
                                                  declared_buffer_bytes(platform) is not None,
                                                  names_value_domain(platform))


def _level_views():
    from mimarsinan.deployment_record.quantities.planes.states import declared_zeros
    from unit.deployment_record.test_level_pricing import _LANE_COUNTERS, _platform, _view
    from unit.mapping.capability.ir_fixtures import lane_graph
    values, spikes = declared_zeros(False, False, True), declared_zeros(False, False, False)  # the platforms' own
    yield _view(lane_graph(), "C0", 4).quantities, values
    yield _view(lane_graph(), "C0", 1).quantities, values
    yield _view(lane_graph(), "C5", 2, platform=_platform(semantics="spiking"), timesteps=4,
                counters=_LANE_COUNTERS).quantities, spikes


def _plane_views():
    yield ("candidate",) + _candidate_view()
    yield from (("record",) + v for v in _record_views())
    yield from (("level",) + v for v in _level_views())


class TestEveryPlanesViews:
    def test_every_absent_factor_of_every_plane_is_in_one_state_with_its_reason(self):
        from unit.deployment_record.test_no_silent_zero_uenergy import _check_view
        planes = {}
        for plane, view, zeros in _plane_views():
            planes[plane] = planes.get(plane, 0) + _check_view(view, zeros)
        assert set(planes) == {"candidate", "record", "level"} and all(planes.values()), planes

    @pytest.mark.parametrize("plane,table", [("from_candidate", "CANDIDATE_UNPRODUCED"),
                                             ("from_level", "LEVEL_UNPRODUCED")])
    def test_a_plane_that_drops_its_table_is_red(self, plane, table, monkeypatch):
        import importlib

        from unit.deployment_record.test_no_silent_zero_uenergy import _check_view
        monkeypatch.setattr(importlib.import_module(f"mimarsinan.deployment_record.quantities.{plane}"), table, {})
        views = {"from_candidate": lambda: [_candidate_view()], "from_level": lambda: list(_level_views())}[plane]()
        with pytest.raises(AssertionError):
            for view, zeros in views:
                _check_view(view, zeros)

    def test_a_zero_beyond_the_platforms_declarations_is_red(self):
        from unit.deployment_record.test_no_silent_zero_uenergy import _check_view
        view, _zeros = _candidate_view()
        with pytest.raises(AssertionError, match="zero by declaration beyond"):
            _check_view(view, {})

    def test_the_derived_views_keep_every_state(self):
        from mimarsinan.deployment_record.cost.absolute.host_split import price_shares
        from mimarsinan.deployment_record.cost.absolute.tables.components import with_derived
        from mimarsinan.deployment_record.platform_physics.conversion import conversion_model_for
        from unit.deployment_record.printed_names import NOT_PRODUCED, unpriced_names
        model = conversion_model_for(get_platform_physics("isaac_like").conversion_model)
        physics = get_platform_physics("generic_estimated_22nm")
        for _plane, view, _zeros in _plane_views():
            derived = with_derived(view, model)
            for state in ("withheld", "unproduced", "known_zero"):
                kept = {k: r for k, r in getattr(view, state).items() if not derived.has(k)}
                assert {k: getattr(derived, state).get(k) for k in kept} == kept, state
            main = {t.name: t for t in price_absolute(view, physics).terms}
            if "energy_dynamic_mj" not in main:  # a spiking level's events are withheld: no energy to compare
                continue
            source = main["energy_dynamic_mj"].source  # the plane's names and its printed zeros, kept by each share
            reasons = [n for n in unpriced_names(source) if NOT_PRODUCED in n] + [
                part for part in source.split("; ") if part.startswith("zero by declaration: ")]
            for variant, bands in price_shares(view, physics)[0].items():
                text = bands["energy_dynamic_mj"].basis
                assert all(n in text for n in reasons if not n.startswith(("e_mac", "e_dma", "e_readout"))), variant

    def test_a_view_stated_by_hand_names_every_declared_row_over_what_it_does_not_state(self):
        from mimarsinan.deployment_record.correlation import available_cases, get_case
        from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
        from unit.deployment_record.test_no_silent_zero_uenergy import TABLES, _every_row_physics, _keys
        from mimarsinan.deployment_record.cost.absolute.context import PricingContext
        checked = 0
        for name in available_cases():
            view = Quantities({k: QuantityValue(v, "measured") for k, v in get_case(name).census.items()})
            ctx = PricingContext.create(view, _every_row_physics())
            for table in TABLES:
                _bands, evidence, unpriced = ctx.component_bands(table, 1.0)
                notes = " | ".join(evidence + unpriced)
                for constant, factors, label in table:
                    absent = [k for f in factors for k in _keys(f) if not ctx.quantities.has(k)]
                    if ctx.priced(constant) and absent:
                        assert f"{constant}({label}" in notes or f"zero by declaration: {constant}" in notes, (
                            name, constant)
                        checked += 1
        assert checked > 0
