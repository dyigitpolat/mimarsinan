"""[W15] Clean programs pass: every built program form of this tree establishes Theorem 1 under its semantics, with
the conditions that bind there checked on real elements (never vacuous where the form holds them), and Proposition 2
reported apart."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.certification.program_check.report import ESTABLISHED, NOT_REQUIRED, NOT_STATIC, VACUOUS

from unit.certification.program_check import programs as P

COUNT_SEMANTICS = ("synchronized_lif", "cascaded_ttfs", "synchronized_ttfs")


def _established(report, *non_vacuous: str) -> None:
    assert report.established("theorem1"), (report.theorem1, {c: [v.text() for v in r.violations]
                                                              for c, r in report.results.items() if r.violations})
    for condition in non_vacuous:
        result = report.results[condition]
        assert result.status == ESTABLISHED and result.checked > 0, (condition, result.status, result.checked)


def test_a_streamed_multi_pass_program_with_pins_and_delays():
    program, ir = P.streamed_multipass()
    report = P.run(program, ir, P.decl("streamed_lif"))
    _established(report, "tiles", "H1-theta", "H1-latency", "R1", "R2", "R3", "H5-collapse", "H8", "H9", "R6")
    assert report.results["H5-skew"].status == ESTABLISHED
    assert report.results["H4"].status == NOT_STATIC
    assert report.proposition2.startswith("conditional") and "H4" in report.proposition2


@pytest.mark.parametrize("semantics", COUNT_SEMANTICS)
def test_the_same_program_under_the_count_and_cascaded_semantics(semantics):
    program, ir = P.streamed_multipass()
    report = P.run(program, ir, P.decl(semantics))
    _established(report, "tiles", "H5-collapse", "H8", "H9")
    timed = semantics == "cascaded_ttfs"
    assert (report.results["R2"].status == ESTABLISHED) == timed
    assert (report.results["R2"].status == NOT_REQUIRED) != timed
    if semantics == "cascaded_ttfs":
        assert report.proposition2.startswith("not claimed")
    else:
        assert report.established("proposition2"), report.proposition2


def test_a_windowed_program_of_level_units():
    program, ir = P.windowed_multipass()
    report = P.run(program, ir, P.decl("windowed_lif"), identity=P.identity_of(ir, retimed=True))
    _established(report, "tiles", "R5", "H0", "H9")


def test_a_coalesced_program_of_fused_cores():
    program, ir = P.coalesced()
    _established(P.run(program, ir, P.decl("streamed_lif")), "tiles", "H6")


def test_an_executed_residency_plan():
    program, ir = P.residency_planned()
    report = P.run(program, ir, P.decl("streamed_lif"))
    _established(report, "tiles", "H10")


def test_a_weight_resident_chain():
    """Bank-backed cores: the HCM uploads each bank block, which the check compares with the grid (bank_content)."""
    program, ir = P.resident_chain()
    report = P.run(program, ir, P.decl("synchronized_lif", core_types=({"max_axons": 8, "max_neurons": 8,
                                                                         "count": 2},)))
    _established(report, "tiles", "H10")
    assert report.counts["bank_cores"] > 0


def test_an_integer_value_program_with_solved_laws():
    program, ir = P.value_program()
    report = P.run(program, ir, P.decl("value_integer", accumulation={"value_census": "float64",
                                                                       "int64_golden": "int64"}))
    _established(report, "tiles", "laws", "H8")
    assert report.established("proposition2"), report.proposition2


def test_a_pruned_value_program_compacts_dead_columns_only():
    program, ir = P.value_program(prune={0: [1, 3]})
    report = P.run(program, ir, P.decl("value_integer", accumulation={"value_census": "float64"}))
    _established(report, "tiles", "laws")
    assert report.results["N-identity"].status in (ESTABLISHED, VACUOUS)


def test_the_identity_program_against_itself():
    _, ir = P.streamed_multipass()
    identity = P.identity_of(ir)
    _established(P.run(identity, ir, P.decl("streamed_lif")), "tiles", "H9")


def test_an_analytical_ttfs_declaration_is_outside_theorem_1_by_name():
    program, ir = P.streamed_multipass()
    report = P.run(program, ir, P.decl("analytical_ttfs"))
    assert report.theorem1.startswith("outside_scope") and "real-valued" in report.theorem1
    assert report.results["tiles"].status == ESTABLISHED


def test_the_report_carries_both_theorems_and_the_walls():
    program, ir = P.streamed_multipass()
    report = P.run(program, ir, P.decl("streamed_lif"), compile_wall_s=1.5)
    records = report.certificate_records()
    assert [r["name"] for r in records] == ["static_program_check_theorem1", "static_program_check_proposition2"]
    assert records[0]["passed"] and not records[1]["passed"]   # Proposition 2 conditional: H4
    assert records[0]["neuron_windows_compared"] == 0 and records[0]["backend"] == "static"
    assert '"compile_wall_s": 1.5' in records[0]["detail"]
    assert report.wall_s > 0 and report.to_dict()["theorem1"].startswith("established on the program")


def test_memos_the_executors_filled_are_fresh():
    """A program whose executors ran (every memo filled) is the same deployment: each memo equals its derivation."""
    import copy

    from unit.mapping.residency.executed import runs as R

    program, ir = P.streamed_multipass()
    ran = copy.deepcopy(program)
    with torch.no_grad():
        R.flow(ran, **R.STREAMED_LIF)(torch.rand(2, P.V.IN))
    assert ran._carried_output_cache is not None and ran._consumer_counts_cache is not None
    _established(P.run(ran, ir, P.decl("streamed_lif")), "tiles", "R3")


def test_a_streamed_program_reads_each_slice_as_the_identity_does():
    program, ir = P.streamed_multipass()
    _established(P.run(program, ir, P.decl("streamed_lif")), "R4", "H5-collapse")


def test_a_deep_copy_of_the_program_is_the_same_deployment():
    """Host modules copied by value are the same host ops (same type, configuration and state)."""
    import copy

    program, ir = P.streamed_multipass()
    _established(P.run(copy.deepcopy(program), ir, P.decl("streamed_lif")), "tiles")
