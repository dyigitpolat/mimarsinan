"""The screen report: exact binomial bounds, teeth by paired control or by mechanism, denominators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimarsinan.certification.randomized.screen import arm_summary, clopper_pearson, one_sided_upper, report

TABLE = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "theorem_box" / "hypothesis_table.json"


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text())


def test_the_zero_event_upper_bound_is_the_closed_form():
    for n in (1, 10, 200, 1000):
        assert one_sided_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-9)
    lo, hi = clopper_pearson(0, 200)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 200), rel=1e-9)


def test_the_two_sided_interval_brackets_the_rate():
    lo, hi = clopper_pearson(30, 100)
    assert 0.21 < lo < 0.3 < hi < 0.40


def _row(arm, outcome, **kw):
    row = {"arm": arm, "outcome": outcome, "detail": "", "wall_s": 0.1, "semantics": "streamed_lif",
           "multipass": False, "carried": False, "coresident": False, "coalesced": False, "split": False}
    row.update(kw)
    return row


def test_an_off_arm_counts_teeth_by_its_paired_control():
    rows = [_row("H5-collapse", "counterexample", control="equal"),
            _row("H5-collapse", "counterexample", control="counterexample"),
            _row("H5-collapse", "equal"), _row("H5-collapse", "refused_unnamed", detail="RuntimeError: x")]
    summary = arm_summary(rows)
    assert summary["admitted"] == 3 and summary["counterexamples"] == 2 and summary["teeth"] == 1
    assert summary["unnamed_refusals"] == {"RuntimeError": 1}


def test_a_base_violation_counts_teeth_by_mechanism(monkeypatch):
    """No arm is a base violation since E1 met R4; an arm declaring a mechanism (R4's arm as it stood before E1)
    still counts its teeth by that mechanism, never by a paired control."""
    from dataclasses import replace

    from mimarsinan.certification.randomized.arms import ARMS

    monkeypatch.setitem(ARMS, "R4/base", replace(ARMS["R4"], id="R4/base", switch=None,
                                                 mechanism="position_dependent_encode"))
    rows = [_row("R4/base", "counterexample", classes=["layer:positions", "position_dependent_encode"]),
            _row("R4/base", "counterexample", classes=["layer:pass_cut", "u1_cut"]), _row("R4/base", "equal")]
    assert arm_summary(rows)["teeth"] == 1


def test_an_arm_that_switches_an_integrated_rule_off_counts_teeth_by_its_paired_control():
    """CL1: R6 and H5-skew switch G5's walks and pass timing off; a mechanism tag alone is no tooth."""
    rows = [_row("R6", "counterexample", classes=["layer:walk", "latency_order_walk"], control="equal"),
            _row("R6", "counterexample", classes=["layer:walk", "latency_order_walk"]), _row("R6", "equal")]
    assert arm_summary(rows)["teeth"] == 1


def test_the_report_pools_the_shipped_on_arms_and_states_the_denominators(table):
    rows = [_row("on", "equal"), _row("on_scheduled", "counterexample", classes=["u1_cut"]),
            _row("on_values_classes", "counterexample", classes=["latency_order_walk"]),
            _row("H8", "counterexample", control="equal")]
    doc = report(rows, table=table)
    assert doc["hypothesis_on"]["admitted"] == 2 and doc["hypothesis_on"]["counterexamples"] == 1
    assert doc["hypothesis_on_harness_setting"]["admitted"] == 1
    assert doc["hypothesis_on_all_strata"]["counterexamples"] == 2
    denominators = doc["denominators"]
    assert "H7" in denominators["not_exercisable_at_code_level"]
    assert "H8" in denominators["with_teeth"]
    assert set(denominators["exercisable_at_code_level"]) == {"H0", "H1-theta", "H5-collapse", "H5-skew", "H8",
                                                               "R6", "R4"}


def test_the_long_window_stratum_is_reported_apart_from_the_declared_line(table):
    """Targeted pass: the stratum beyond the declared window range (T in 9-64) has its own summary and network
    relation; the hypothesis-on line, its pools and its coverage stay the declared range's."""
    rows = [_row("on", "equal", semantics="synchronized_lif", network_claim="claimed", network_windows=4,
                 network_mismatches=0),
            _row("on_long_window", "counterexample", semantics="synchronized_lif",
                 classes=["layer:counts", "count_rebuild"], network_claim="claimed", network_windows=6,
                 network_mismatches=2)]
    doc = report(rows, table=table)
    assert (doc["hypothesis_on"]["admitted"], doc["hypothesis_on"]["counterexamples"]) == (1, 0)
    assert doc["hypothesis_on_all_strata"]["admitted"] == 1 and doc["coverage"]["admitted"] == 1
    beyond = doc["hypothesis_on_beyond_declared_window"]
    assert (beyond["admitted"], beyond["counterexamples"], beyond["classes"]["count_rebuild"]) == (1, 1, 1)
    assert "9-64" in doc["hypothesis_on_beyond_declared_window_scope"]
    assert doc["network_relation"]["per_semantics"]["synchronized_lif"]["differ"] == 0
    assert doc["network_relation_beyond_declared_window"]["per_semantics"]["synchronized_lif"]["differ"] == 1


def test_the_denominators_list_every_hypothesis_of_the_box(table):
    """Review round 2: the denominator is the fixture's, every row with its code arm, its model arm or where
    it is pending (CL1), so no condition of the box drops out of the count."""
    rows = [_row("R4", "counterexample", classes=["layer:positions", "position_dependent_encode"], control="equal"),
            _row("H1-theta", "counterexample", control="equal")]
    den = report(rows, table=table)["denominators"]
    ids = [h["id"] for h in table["hypotheses"]]
    assert den["hypotheses"] == ids and list(den["per_hypothesis"]) == ids
    assert set(den["switched_off_with_paired_controls"]) == {"H1-theta", "H5-collapse", "H8", "H5-skew", "R6",
                                                             "H0", "R4"}
    assert set(den["violated_by_the_base"]) == set()  # E1 met R4, the last base violation
    assert "H1-latency" in den["model_level_only"]  # asserted at run time on the integrated tree (CL1)
    for hid, entry in den["per_hypothesis"].items():
        assert entry["code_arm"] or entry["model_arm"] or entry["pending"], hid
    assert den["per_hypothesis"]["R5"]["pending"] == "CL1 (G5)"
    assert {"R1", "R2", "R3"} <= set(den["model_level_only"]) and "R5" in den["not_exercised"]
    assert {"R4", "H1-theta"} <= set(den["with_teeth"])
    assert den["restated"].startswith("7 switched off with paired controls + 0 violated by the base, of 20")


def test_the_report_states_the_value_runs_against_the_int64_golden(table):
    """CL1: every admitted law-carrying run (identity program and deployment) against G2's int64 golden at atol 0,
    the on strata apart from every arm's rows; a refused instance compares nothing."""
    rows = [_row("on", "equal", semantics="value", golden_windows=40, golden_mismatches=0),
            _row("on_scheduled", "counterexample", semantics="value", golden_windows=20, golden_mismatches=3),
            _row("on", "reference_error", semantics="value", golden_windows=0),
            _row("R6", "equal", semantics="value", golden_windows=10, golden_mismatches=0)]
    doc = report(rows, table=table)
    on = doc["golden_relation"]
    assert (on["instances"], on["windows"], on["differing_windows"], on["differ"]) == (2, 60, 3, 1)
    assert doc["golden_relation_all_arms"]["instances"] == 3


def test_a_table_arm_the_registry_lacks_is_refused_by_name(table):
    from mimarsinan.certification.randomized.screen import ScreenTableError

    broken = json.loads(json.dumps(table))
    broken["hypotheses"][0]["arms"]["code"] = "no_such_arm"
    with pytest.raises(ScreenTableError):
        report([], table=broken)


def test_the_network_relation_is_reported_per_semantics_with_its_claim():
    rows = [_row("on", "equal", semantics="synchronized_lif", network_claim="claimed", network_windows=10,
                 network_mismatches=2),
            _row("on", "refused_unnamed", semantics="synchronized_lif", network_claim="claimed", network_windows=6,
                 network_mismatches=0),
            _row("on", "equal", semantics="streamed_lif", network_claim="conditional", network_windows=8,
                 network_mismatches=1)]
    relation = report(rows, table=json.loads(TABLE.read_text()))["network_relation"]
    sync = relation["per_semantics"]["synchronized_lif"]
    assert (sync["instances"], sync["differ"], sync["windows"], sync["differing_windows"]) == (2, 1, 16, 2)
    assert relation["claimed_and_differing"] == ["synchronized_lif"]


def test_the_coverage_reports_weights_input_dependence_and_the_fixed_axes():
    from mimarsinan.certification.randomized.screen import coverage

    rows = [_row("on", "equal", semantics="value", weight_ranges=[8, 0], input_dependent=0.5),
            _row("on", "equal", semantics="value", weight_ranges=[7, 8], input_dependent=0.0)]
    cov = coverage(rows)
    value = cov["weights_and_input_dependence"]["value"]
    assert value["layers"]["1"] == {"layers": 2, "all_zero": 1, "max_abs": {0: 1, 8: 1}}
    assert value["input_independent_instances"] == 1 and value["input_dependent_median"] == 0.25
    assert set(cov["fixed_function_bearing_axes"]) == {"comparator", "reset", "membrane", "encoder", "granularity",
                                                       "value_law"}


def test_the_coverage_reports_every_declared_variable():
    from mimarsinan.certification.randomized.screen import coverage

    decl = {"cores": [{"max_axons": 20, "max_neurons": 5, "count": 3}], "allow_coalescing": True,
            "allow_neuron_splitting": False, "allow_scheduling": True, "max_schedule_passes": 4,
            "core_value_granularity": {}, "residency_plan": {"step": "floor"}}
    rows = [_row("on", "equal", declaration=decl, drawn=["cores.count", "residency_plan.step"], motifs=["chain"],
                 timesteps=4)]
    cov = coverage(rows)
    assert cov["drawn_paths"]["residency_plan.step"] == 1 and "present in 1 " in cov["drawn_by_fields"]["residency_plan"]
    # residency_plan is drawn, not deferred; U-F4's pipelined form and [U-L2] its banked reduction cores are deferred
    # beside position_multiplexing, [thru:k] and the layer copies' two keys
    assert set(cov["deferred_to_cl1"]) == {"position_multiplexing", "pipelined_multiplexing", "reduction_core_banks",
                                           "layer_replication", "wavefront_group_cores"}
    assert cov["declared"]["residency_plan"] == {'{"step": "floor"}': 1}
    declared = cov["declared"]
    assert declared["max_axons"] == {"17-32": 1} and declared["count"] == {"1-8": 1}
    assert declared["flags_true"] == {"allow_coalescing": 1, "allow_neuron_splitting": 0, "allow_scheduling": 1}
    assert declared["core_value_granularity"] == {"{}": 1}
