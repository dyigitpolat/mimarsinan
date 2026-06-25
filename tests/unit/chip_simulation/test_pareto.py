"""E5 — the Pareto DECISION layer over (cost, accuracy) per (cell, schedule).

Locks the mechanism that turns the campaign's per-schedule accuracy rows + a
DEFENSIBLE cost PROXY (with an uncertainty band) into the program's
"automatic genericity" evidence:

* :func:`pareto_front` — the non-dominated (cost↓ × accuracy↑) subset of a hand-set
  of points, mirroring ``cost_extraction._dominates`` (no-worse on all, strictly
  better on one);
* :func:`schedule_cost_band` — the documented latency proxy
  (sync ``= S × latency_groups``; cascaded ``= S + latency_groups``) carried as a
  (lo, mid, hi) band, LABELED a model-estimate, never measured per-sample energy;
* :func:`cascaded_vs_synchronized` — the per-dataset deep_cnn verdict: which schedule
  is on the front, the measured accuracy gap (the ~6pp d10/S4 MNIST gap MUST be
  detected), the cost-band assumption, and the E5b RETIRE-or-REGIME recommendation,
  CONDITIONAL on the cost-proxy band;
* :func:`propose_recipe` — E5c selector: accuracy-priority budget → synchronized;
  hard-latency budget → cascaded IF on the front, else synchronized.

The real-ledger assertions read ``runs/campaign/ledger.jsonl`` when present and are
skipped (not failed) when the runtime artifact is absent from the worktree.
"""

import json
import os

import pytest

from mimarsinan.chip_simulation.cost_extraction import (
    CostRecord,
    save_cost_record,
)
from mimarsinan.chip_simulation.pareto import (
    CostProxyBand,
    ScheduleVerdict,
    cascaded_vs_synchronized,
    load_deep_cnn_rows,
    load_measured_cost,
    pareto_front,
    propose_recipe,
    schedule_cost_band,
)


# Real ledger lives in the MAIN repo runtime tree, not the worktree git state.
_LEDGER_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs", "campaign", "ledger.jsonl"),
    "/home/yigit/repos/research_stuff/mimarsinan/runs/campaign/ledger.jsonl",
]


def _real_ledger_path():
    for cand in _LEDGER_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    return None


# --------------------------------------------------------------------------- #
# pareto_front — pure dominance math on a hand-set of points.
# --------------------------------------------------------------------------- #

def test_pareto_front_picks_non_dominated_subset():
    points = [
        {"label": "A", "cost": 1.0, "accuracy": 0.90},  # cheap, low acc  -> on front
        {"label": "B", "cost": 2.0, "accuracy": 0.99},  # dear, high acc  -> on front
        {"label": "C", "cost": 2.0, "accuracy": 0.95},  # B dominates C   -> off
        {"label": "D", "cost": 3.0, "accuracy": 0.90},  # A dominates D   -> off
    ]
    front = {p["label"] for p in pareto_front(points)}
    assert front == {"A", "B"}


def test_pareto_front_single_point_is_on_front():
    assert pareto_front([{"label": "solo", "cost": 5.0, "accuracy": 0.5}])[0]["label"] == "solo"


def test_pareto_front_empty_is_empty():
    assert pareto_front([]) == []


def test_pareto_front_equal_points_both_kept():
    # Two identical (cost, accuracy) points dominate nobody (no strict improvement).
    pts = [{"label": "X", "cost": 1.0, "accuracy": 0.9}, {"label": "Y", "cost": 1.0, "accuracy": 0.9}]
    front = {p["label"] for p in pareto_front(pts)}
    assert front == {"X", "Y"}


def test_pareto_front_strictly_cheaper_same_accuracy_dominates():
    pts = [
        {"label": "cheap", "cost": 1.0, "accuracy": 0.9},
        {"label": "dear", "cost": 2.0, "accuracy": 0.9},
    ]
    front = {p["label"] for p in pareto_front(pts)}
    assert front == {"cheap"}


# --------------------------------------------------------------------------- #
# schedule_cost_band — the documented latency proxy + band, LABELED a model.
# --------------------------------------------------------------------------- #

def test_synchronized_latency_proxy_matches_documented_model():
    # Synchronized: sim_time = S x latency_groups (groups run sequentially).
    band = schedule_cost_band("synchronized", s_global=4, depth=10)
    assert band.nominal_latency_steps == 4 * 10
    assert band.is_model_estimate is True
    assert "not measured per-sample energy" in band.disclaimer.lower()


def test_cascaded_latency_proxy_is_pipelined_lower():
    # Cascaded pipelined: sim_time = S + latency_groups -> strictly lower than sync.
    casc = schedule_cost_band("cascaded", s_global=4, depth=10)
    sync = schedule_cost_band("synchronized", s_global=4, depth=10)
    assert casc.nominal_latency_steps == 4 + 10
    assert casc.nominal_latency_steps < sync.nominal_latency_steps


def test_cost_band_is_ordered_lo_mid_hi():
    band = schedule_cost_band("synchronized", s_global=4, depth=8, cores=480)
    assert band.lo_latency_steps <= band.nominal_latency_steps <= band.hi_latency_steps
    # Energy proxy (cores x active steps) carries the same ordering when cores known.
    assert band.lo_energy_proxy <= band.nominal_energy_proxy <= band.hi_energy_proxy


def test_energy_axis_uninstrumented_without_cores():
    band = schedule_cost_band("synchronized", s_global=4, depth=8, cores=None)
    assert band.energy_uninstrumented is True
    assert band.nominal_energy_proxy is None


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError):
        schedule_cost_band("teleport", s_global=4, depth=8)


# --------------------------------------------------------------------------- #
# cascaded_vs_synchronized — verdict on a synthetic + the real ledger.
# --------------------------------------------------------------------------- #

def _synthetic_rows():
    """A small deep_cnn-shaped set: MNIST has the ~6pp gap, a no-gap dataset too."""
    return [
        {
            "model": "deep_cnn", "dataset": "mnist", "depth": 10, "S": 4, "cores_count": 480,
            "cascaded_deployed_mean": 0.9297, "synchronized_deployed_mean": 0.9903,
            "ann_test_acc_mean": 0.9907, "cascaded_to_sync_gap_pp": -6.06,
        },
        {
            "model": "deep_cnn", "dataset": "tied_ds", "depth": 4, "S": 4, "cores_count": 480,
            "cascaded_deployed_mean": 0.9898, "synchronized_deployed_mean": 0.9898,
            "ann_test_acc_mean": 0.992, "cascaded_to_sync_gap_pp": 0.0,
        },
    ]


def test_cascaded_vs_synchronized_detects_6pp_gap_synthetic():
    verdict = cascaded_vs_synchronized(_synthetic_rows())
    mnist = verdict.per_dataset["mnist"]
    assert isinstance(mnist, ScheduleVerdict)
    # ~6pp accuracy gap (synchronized better) detected to within rounding.
    assert mnist.accuracy_gap_pp == pytest.approx(6.06, abs=0.5)
    assert mnist.front_schedule == "synchronized"
    # Synchronized dominates ONLY if it is also not-worse on cost; it is NOT (higher
    # latency), so the verdict must be a REGIME, not an unconditional RETIRE.
    assert mnist.recommendation in {"REGIME_DEPENDENT", "RETIRE_CASCADED"}
    assert mnist.conditional_on_cost_band is True


def test_cascaded_vs_synchronized_regime_when_cascaded_cheaper():
    # MNIST: sync wins accuracy but cascaded has strictly lower latency -> both on the
    # (cost, accuracy) front -> REGIME (cascaded wins the hard-latency budget).
    verdict = cascaded_vs_synchronized(_synthetic_rows())
    mnist = verdict.per_dataset["mnist"]
    front_labels = {p["label"] for p in mnist.front_points}
    assert "synchronized" in front_labels
    assert "cascaded" in front_labels  # cascaded survives on the latency axis
    assert mnist.recommendation == "REGIME_DEPENDENT"


def test_cascaded_vs_synchronized_tied_dataset_no_gap():
    verdict = cascaded_vs_synchronized(_synthetic_rows())
    tied = verdict.per_dataset["tied_ds"]
    assert abs(tied.accuracy_gap_pp) < 0.5


def test_verdict_carries_cost_band_assumption_text():
    verdict = cascaded_vs_synchronized(_synthetic_rows())
    mnist = verdict.per_dataset["mnist"]
    assert isinstance(mnist.cost_band, CostProxyBand)
    assert "model-estimate" in mnist.cost_band_assumption.lower()


def test_cell_selection_prefers_cores_instrumented_then_latest():
    # Two MNIST cells: a deeper UN-instrumented one and a shallower cores-instrumented
    # one. The verdict must pick the INSTRUMENTED cell (real energy axis) deterministically.
    rows = [
        {
            "model": "deep_cnn", "dataset": "mnist", "depth": 12, "S": 4, "cores_count": None,
            "cascaded_deployed_mean": 0.878, "synchronized_deployed_mean": 0.9923,
            "ann_test_acc_mean": 0.9921, "ts": 1.0,
        },
        {
            "model": "deep_cnn", "dataset": "mnist", "depth": 10, "S": 4, "cores_count": 480,
            "cascaded_deployed_mean": 0.9297, "synchronized_deployed_mean": 0.9903,
            "ann_test_acc_mean": 0.9907, "ts": 2.0,
        },
    ]
    cell = cascaded_vs_synchronized(rows).per_dataset["mnist"]
    assert cell.cores == 480
    assert cell.depth == 10
    assert cell.cost_band.energy_uninstrumented is False  # energy axis is real here


def test_cell_selection_breaks_ties_by_latest_ts():
    rows = [
        {
            "model": "deep_cnn", "dataset": "mnist", "depth": 10, "S": 4, "cores_count": 480,
            "cascaded_deployed_mean": 0.80, "synchronized_deployed_mean": 0.99,
            "ann_test_acc_mean": 0.99, "ts": 1.0,
        },
        {
            "model": "deep_cnn", "dataset": "mnist", "depth": 10, "S": 4, "cores_count": 480,
            "cascaded_deployed_mean": 0.9297, "synchronized_deployed_mean": 0.9903,
            "ann_test_acc_mean": 0.9907, "ts": 9.0,  # the reconciled, latest measurement
        },
    ]
    cell = cascaded_vs_synchronized(rows).per_dataset["mnist"]
    assert cell.cascaded_accuracy == pytest.approx(0.9297)


@pytest.mark.skipif(_real_ledger_path() is None, reason="campaign ledger not present in worktree runtime tree")
def test_real_ledger_d10_mnist_6pp_gap_detected():
    rows = load_deep_cnn_rows(_real_ledger_path())
    assert rows, "no deep_cnn rows mined from the real ledger"
    verdict = cascaded_vs_synchronized(rows)
    assert "mnist" in verdict.per_dataset
    mnist = verdict.per_dataset["mnist"]
    # The headline measured cell (d10/S4, cores=480): sync ~0.9903 vs cascaded ~0.9297,
    # the ~6pp deployment gap the unit is required to detect.
    assert mnist.cores == 480
    assert mnist.cost_band.energy_uninstrumented is False
    assert mnist.front_schedule == "synchronized"
    assert mnist.accuracy_gap_pp == pytest.approx(6.06, abs=1.0)
    # Synchronized wins accuracy but cascaded keeps the latency axis -> REGIME, not retire.
    assert mnist.recommendation == "REGIME_DEPENDENT"
    assert mnist.conditional_on_cost_band is True


@pytest.mark.skipif(_real_ledger_path() is None, reason="campaign ledger not present in worktree runtime tree")
def test_real_ledger_harder_datasets_show_larger_gap():
    rows = load_deep_cnn_rows(_real_ledger_path())
    verdict = cascaded_vs_synchronized(rows)
    # fmnist is the hardest small dataset -> the largest cascaded deficit.
    if "fmnist" in verdict.per_dataset and "mnist" in verdict.per_dataset:
        assert verdict.per_dataset["fmnist"].accuracy_gap_pp >= verdict.per_dataset["mnist"].accuracy_gap_pp


# --------------------------------------------------------------------------- #
# propose_recipe — E5c budget-driven selector off the front.
# --------------------------------------------------------------------------- #

def test_propose_recipe_accuracy_priority_picks_synchronized():
    rec = propose_recipe("accuracy", rows=_synthetic_rows(), dataset="mnist")
    assert rec.schedule == "synchronized"
    assert rec.firing == "ttfs"
    assert rec.s_global == 4
    assert rec.placement == "on_chip_majority"


def test_propose_recipe_hard_latency_picks_cascaded_when_on_front():
    rec = propose_recipe("latency", rows=_synthetic_rows(), dataset="mnist")
    # Cascaded is on the (latency, accuracy) front for MNIST -> hard-latency picks it.
    assert rec.schedule == "cascaded"


def test_propose_recipe_falls_back_to_synchronized_when_cascaded_off_front():
    # A degenerate row where cascaded is BOTH lower accuracy AND not lower latency
    # (depth makes sync latency the same as cascaded only when depth==1 -> tie).
    rows = [
        {
            "model": "deep_cnn", "dataset": "edge", "depth": 1, "S": 4, "cores_count": 480,
            "cascaded_deployed_mean": 0.80, "synchronized_deployed_mean": 0.99,
            "ann_test_acc_mean": 0.99, "cascaded_to_sync_gap_pp": -19.0,
        },
    ]
    rec = propose_recipe("latency", rows=rows, dataset="edge")
    assert rec.schedule == "synchronized"


def test_propose_recipe_unknown_budget_rejected():
    with pytest.raises(ValueError):
        propose_recipe("free-energy", rows=_synthetic_rows(), dataset="mnist")


# --------------------------------------------------------------------------- #
# load_measured_cost — E5 PREFERS a measured cost_record over the proxy.
# --------------------------------------------------------------------------- #

def _write_cost_record(run_dir, *, schedule, latency_steps, mj_per_sample, acc):
    sync = schedule if schedule in ("cascaded", "synchronized") else None
    record = CostRecord(
        cell_key=f"ttfs/{sync}@sanafe" if sync else "ttfs@sanafe",
        mode=f"ttfs/{sync}" if sync else "ttfs",
        backend="sanafe",
        acc_deploy=float(acc),
        mj_per_sample=float(mj_per_sample),
        spikes=123,
        latency_steps=int(latency_steps),
        cores=480,
        s_global=4,
        depth=10,
    )
    save_cost_record(record, run_dir)


def test_load_measured_cost_resolves_run_dir_record(tmp_path):
    run_dir = tmp_path / "dcnn_d10_synchronized_s0"
    _write_cost_record(
        str(run_dir), schedule="synchronized",
        latency_steps=40, mj_per_sample=12.5, acc=0.9903,
    )
    row = {"synchronized_run_ids": ["dcnn_d10_synchronized_s0"]}

    def _resolver(run_id):
        return str(tmp_path / run_id)

    measured = load_measured_cost(
        row, schedule="synchronized", run_dir_resolver=_resolver,
    )
    assert measured is not None
    assert measured.latency_steps == 40
    assert measured.mj_per_sample == pytest.approx(12.5)


def test_load_measured_cost_returns_none_when_unresolvable(tmp_path):
    row = {"synchronized_run_ids": ["does_not_exist"]}

    def _resolver(run_id):
        return str(tmp_path / run_id)  # directory has no cost_record.json

    assert load_measured_cost(
        row, schedule="synchronized", run_dir_resolver=_resolver,
    ) is None


def test_load_measured_cost_no_resolver_is_none():
    # No resolver -> no measured cost (the proxy path stays in force).
    row = {"synchronized_run_ids": ["dcnn_d10_synchronized_s0"]}
    assert load_measured_cost(row, schedule="synchronized") is None


def test_verdict_prefers_measured_latency_when_record_present(tmp_path):
    rows = _synthetic_rows()
    mnist = next(r for r in rows if r["dataset"] == "mnist")
    mnist["cascaded_run_ids"] = ["dcnn_d10_cascaded_s0"]
    mnist["synchronized_run_ids"] = ["dcnn_d10_synchronized_s0"]

    # Measured latency that DIFFERS from the proxy (proxy: sync=40, cascaded=14).
    _write_cost_record(
        str(tmp_path / "dcnn_d10_synchronized_s0"), schedule="synchronized",
        latency_steps=37, mj_per_sample=9.9, acc=0.9903,
    )
    _write_cost_record(
        str(tmp_path / "dcnn_d10_cascaded_s0"), schedule="cascaded",
        latency_steps=11, mj_per_sample=2.2, acc=0.9297,
    )

    def _resolver(run_id):
        return str(tmp_path / run_id)

    verdict = cascaded_vs_synchronized(rows, run_dir_resolver=_resolver)
    cell = verdict.per_dataset["mnist"]
    # The decision points now use the MEASURED latency, not the proxy
    # (proxy would be sync=40 / cascaded=14; measured is 37 / 11).
    costs = {p["label"]: p["cost"] for p in cell.decision_points}
    assert costs["synchronized"] == 37
    assert costs["cascaded"] == 11
    # The proxy cost band is still carried as the documented fallback record.
    assert cell.cost_band.nominal_latency_steps == 40
    # The verdict still flags conditionality on the cost band.
    assert cell.conditional_on_cost_band is True


def test_verdict_falls_back_to_proxy_byte_identical_without_resolver():
    """Without a resolver (no measured records), the verdict is byte-identical
    to the established proxy path."""
    rows = _synthetic_rows()
    baseline = cascaded_vs_synchronized(rows).per_dataset["mnist"]
    with_default = cascaded_vs_synchronized(rows, run_dir_resolver=None).per_dataset["mnist"]
    assert baseline.front_points == with_default.front_points
    assert baseline.recommendation == with_default.recommendation
    assert baseline.accuracy_gap_pp == with_default.accuracy_gap_pp
    # The proxy latency (sync = S*depth = 40) is in force.
    sync_cost = next(
        p["cost"] for p in with_default.front_points if p["label"] == "synchronized"
    )
    assert sync_cost == 40
