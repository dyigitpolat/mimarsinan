"""[E6] W03's residency plans and fit pins, and the pack unit's neuron split and position multiplexing, under W15.

W03 pins every core of a program whose passes that load their own cores fit the declared pool side by side
(``mark_fit_residency``, ``residency_rules.fit_pins``): a wholly resident pass runs on the cores its programming pass
loaded, so H10 counts it on its head's cores, never beside them. A core whose geometry decomposes into no declared
core type (or ambiguously) leaves the fit undecidable on the declared pool: H10 names it (``undeclared_core``) instead
of raising the planner's refusal. A position-multiplexed run (W02) is a head pass and passes resident on its copies
(H10's resident content); neuron-split fragments (WI-5f) on fused cores are placements like any other (tiles, H6)."""

from __future__ import annotations

import copy

import numpy as np

from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENT_CORES

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV


def _failed(report, condition: str, rule: str) -> None:
    result = report.results[condition]
    assert result.status == FAILED, (condition, result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    assert report.theorem1.startswith("failed") and condition in report.theorem1, report.theorem1


def test_a_fit_program_whose_resident_passes_run_on_their_heads_cores_is_established():
    program, ir, decl = UV.fit_chain()
    passes = P.neural(program)
    assert len(passes) >= 2 and all(getattr(s, SCHEDULE_RESIDENT_CORES, None) for s in passes)
    assert any(getattr(s, "schedule_weights_resident", False) for s in passes)
    report = P.run(program, ir, decl)
    assert report.results["H10"].status == ESTABLISHED and report.results["H10"].checked > 0, (
        [v.text() for v in report.results["H10"].violations])
    assert report.established("theorem1"), report.theorem1


def test_a_fit_on_a_pool_too_small_for_its_loading_passes():
    program, ir, _decl = UV.fit_chain()
    report = P.run(program, ir, P.decl("synchronized_lif", core_types=({"max_axons": 8, "max_neurons": 8,
                                                                         "count": 1},)))
    _failed(report, "H10", "pinned_slot_overwritten")


def test_a_pinned_core_no_declared_core_type_holds_is_named():
    """The chain's 8 x 8 cores under G5's 25 x 8 declaration: no declared type decomposes them, so the fit is not
    read on the declared pool; the check names it and finishes (the planner raises ``ResidencyPlanError`` there)."""
    program, ir, _decl = UV.fit_chain()
    report = P.run(program, ir, P.decl("synchronized_lif"))
    _failed(report, "H10", "undeclared_core")


def test_a_resident_pass_pinning_beside_its_head_counts_once():
    """A wholly resident pass's pins are its head's cores: pinning them again in the resident pass adds no tile."""
    program, ir, decl = UV.fit_chain()
    bad = copy.deepcopy(program)
    passes = P.neural(bad)
    for stage in passes[1:]:
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset(range(len(stage.hard_core_mapping.cores))))
    assert P.run(bad, ir, decl).results["H10"].status == ESTABLISHED


def test_a_position_multiplexed_program_is_established_on_its_resident_runs():
    program, ir, decl = UV.multiplexed()
    resident = [s for s in P.neural(program) if getattr(s, "schedule_weights_resident", False)]
    assert resident, "the vehicle multiplexes positions"
    report = P.run(program, ir, decl)
    assert report.results["H10"].status == ESTABLISHED and report.results["H10"].checked > 0
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())


def test_a_multiplexed_pass_holding_other_weights_than_its_runs_copy():
    program, ir, decl = UV.multiplexed()
    bad = copy.deepcopy(program)
    stage = next(s for s in P.neural(bad) if getattr(s, "schedule_weights_resident", False))
    core = stage.hard_core_mapping.cores[0]
    grid = np.array(core.get_core_matrix(), dtype=np.float64)
    grid[0, 0] += 1.0
    core.core_matrix = grid
    _failed(P.run(bad, ir, decl), "H10", "resident_content")


def _split_records(program):
    groups = {}
    for stage in P.neural(program):
        for records in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            for record in records:
                if record.get("split_group_id") is not None:
                    groups.setdefault((id(stage), record["split_group_id"]), []).append(record)
    return [records for records in groups.values() if len(records) >= 2]


def test_neuron_split_fragments_on_fused_cores_are_established():
    program, ir, decl = UV.neuron_split()
    fused = [c for s in P.neural(program) for c in s.hard_core_mapping.cores if c.fused_component_axons]
    assert fused and _split_records(program), "the vehicle splits neurons onto fused cores"
    report = P.run(program, ir, decl)
    assert report.results["H6"].checked > 0 and report.results["tiles"].status == ESTABLISHED
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())


def test_two_fragments_placed_as_each_others_neurons():
    program, ir, decl = UV.neuron_split()
    bad = copy.deepcopy(program)
    first, second = _split_records(bad)[0][:2]
    if first.get("neurons") != second.get("neurons"):
        second, first = sorted(_split_records(bad)[0], key=lambda r: r["neurons"])[:2]
    first["neuron_range_in_original"], second["neuron_range_in_original"] = (
        second["neuron_range_in_original"], first["neuron_range_in_original"])
    report = P.run(bad, ir, decl)
    assert report.results["tiles"].status == FAILED
    assert {"fan_in", "coverage", "placement_extent"} & {v.rule for v in report.results["tiles"].violations}
