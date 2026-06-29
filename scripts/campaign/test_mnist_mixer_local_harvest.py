"""Tests for the LOCAL MNIST mixer ledger harvester (run_info + manifest, no GPU)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mnist_mixer_local_harvest as h  # noqa: E402


def _manifest_job(run_id: str, *, cell_id: str, firing: str, sync: str, recipe_id: str,
                  recipe_family: str, diagnostic_role: str = "diagnostic") -> dict:
    return {
        "id": run_id,
        "tags": {
            "study": h.STUDY,
            "cluster": h.CLUSTER,
            "cell_id": cell_id,
            "diagnostic_role": diagnostic_role,
            "vehicle": "mlp_mixer_core",
            "dataset": "MNIST_DataProvider",
            "firing": firing,
            "sync": sync,
            "backend": "sanafe",
            "acceptance_min_deployed_acc": 0.97,
            "acceptance_max_relative_time": 1.0,
            "recipe_id": recipe_id,
            "recipe_family": recipe_family,
            "budget_schedule": {"ramp_steps": 1},
            "required_probes": [],
            "batch_id": f"{h.RUN_PREFIX}{cell_id}_{recipe_id}",
        },
    }


def _manifest(tmp_path: Path) -> h.ManifestIndex:
    jobs = [
        _manifest_job(
            f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_analytical_control_mixer_ttfs_analytical_control_s{s}",
            cell_id="mnist_mmixcore_ttfs_analytical_control",
            firing="ttfs", sync="analytical",
            recipe_id="mixer_ttfs_analytical_control", recipe_family="ttfs_analytical",
            diagnostic_role="analytical_control",
        )
        for s in (0, 1)
    ]
    jobs.append(
        _manifest_job(
            f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_cascaded_genuine_blend_fast_s0",
            cell_id="mnist_mmixcore_ttfs_cycle_cascaded",
            firing="ttfs_cycle_based", sync="cascaded",
            recipe_id="mixer_cascaded_genuine_blend_fast", recipe_family="ttfs_cycle_cascaded",
        )
    )
    return h.build_manifest_index(jobs)


def _write_run(generated: Path, run_id: str, *, status: str, started: float | None,
               finished: float | None, acc: float | None, error=None) -> None:
    run_dir = generated / f"{run_id}{h.RUN_SUFFIX}"
    (run_dir / "_GUI_STATE").mkdir(parents=True)
    (run_dir / "_GUI_STATE" / "run_info.json").write_text(
        json.dumps({"status": status, "started_at": started, "finished_at": finished, "error": error})
    )
    if acc is not None:
        (run_dir / "__target_metric.json").write_text(str(acc))


# --------------------------------------------------------------------------- axis recovery


def test_recover_axes_exact_match(tmp_path):
    index = _manifest(tmp_path)
    run_id = f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_cascaded_genuine_blend_fast_s0"
    axes, source = h.recover_axes(run_id, index)
    assert source == "manifest_exact"
    assert axes["recipe_family"] == "ttfs_cycle_cascaded"
    assert axes["firing"] == "ttfs_cycle_based"
    assert axes["seed"] == 0
    assert axes["budget_schedule"] == {"ramp_steps": 1}


def test_recover_axes_cell_derived_controller(tmp_path):
    """A controller_baseline recipe absent from the manifest still recovers cell axes."""
    index = _manifest(tmp_path)
    run_id = f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_controller_baseline_s2"
    axes, source = h.recover_axes(run_id, index)
    assert source == "manifest_cell_derived"
    assert axes["recipe_id"] == "mixer_controller_baseline"
    assert axes["recipe_family"] == "controller"
    assert axes["firing"] == "ttfs_cycle_based"  # inherited from the cell
    assert axes["sync"] == "cascaded"
    assert axes["seed"] == 2
    assert axes["budget_schedule"] is None


def test_recover_axes_cell_derived_diagnostic(tmp_path):
    index = _manifest(tmp_path)
    run_id = f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_some_new_recipe_s1"
    axes, source = h.recover_axes(run_id, index)
    assert source == "manifest_cell_derived"
    assert axes["recipe_family"] == "ttfs_cycle_cascaded"  # cell diagnostic family


# --------------------------------------------------------------------------- per-run gates


def test_completed_pass_below_baseline(tmp_path):
    run = h.LocalRun("r", tmp_path, "completed", 0, 0.98, 100.0, None,
                     {"firing": "ttfs_quantized", "sync": "analytical", "backend": "sanafe"}, "x")
    rec = h.classify_local_run(run, baseline_wall_s=200.0, min_acc=0.97)
    assert rec.status == h.PASS
    assert rec.relative_time == 0.5
    assert rec.failure_reason is None


def test_completed_flag_accuracy(tmp_path):
    run = h.LocalRun("r", tmp_path, "completed", 0, 0.94, 100.0, None, {"firing": "ttfs_cycle_based"}, "x")
    rec = h.classify_local_run(run, baseline_wall_s=200.0, min_acc=0.97)
    assert rec.status == h.FLAG
    assert rec.failure_reason == "accuracy_below_97"
    assert rec.owner == "conversion"


def test_completed_flag_slower_than_baseline(tmp_path):
    run = h.LocalRun("r", tmp_path, "completed", 0, 0.99, 300.0, None, {"firing": "ttfs"}, "x")
    rec = h.classify_local_run(run, baseline_wall_s=200.0, min_acc=0.97)
    assert rec.status == h.FLAG
    assert rec.failure_reason == "slower_than_baseline"
    assert rec.owner == "tuning"


def test_failed_collapse_is_measured_dead(tmp_path):
    run = h.LocalRun("r", tmp_path, "failed", 1, 0.1135, 50.0,
                     "[TTFS Cycle Fine-Tuning] step failed to retain performance", {}, "x")
    rec = h.classify_local_run(run, baseline_wall_s=200.0, min_acc=0.97)
    assert rec.status == h.DEAD
    assert rec.failure_reason == "measured_dead"
    assert rec.owner == "research"


def test_failed_parity_is_flagged_faithfulness(tmp_path):
    run = h.LocalRun("r", tmp_path, "failed", 1, 0.958, 50.0,
                     "NF<->SCM per-neuron parity failed", {}, "x")
    rec = h.classify_local_run(run, baseline_wall_s=200.0, min_acc=0.97)
    assert rec.status == h.FLAG
    assert rec.failure_reason == "parity_failure"
    assert rec.owner == "faithfulness"


def test_incomplete_running_has_no_wall_or_rc(tmp_path):
    run = h.LocalRun("r", tmp_path, "running", None, 0.94, None, None, {}, "x")
    rec = h.classify_local_run(run, baseline_wall_s=200.0, min_acc=0.97)
    assert rec.status == h.INCOMPLETE
    assert rec.returncode is None
    assert rec.relative_time is None


# --------------------------------------------------------------------------- deploy cost


def _write_cost(generated: Path, run_id: str, **fields) -> None:
    run_dir = generated / f"{run_id}{h.RUN_SUFFIX}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cost_record.json").write_text(json.dumps(fields))


def test_read_cost_record_parses_on_chip_axes(tmp_path):
    _write_cost(tmp_path, "r", latency_steps=13, mj_per_sample=0.0359, spikes=322933,
                energy_proxy_neuron_steps=399360, cores=115, s_global=4, acc_deploy=0.972)
    cost = h.read_cost_record(tmp_path / f"r{h.RUN_SUFFIX}")
    assert cost is not None
    assert cost.latency_steps == 13
    assert cost.mj_per_sample == 0.0359
    assert cost.spikes == 322933
    assert cost.cores == 115
    assert cost.acc_deploy == 0.972


def test_read_cost_record_absent_returns_none(tmp_path):
    (tmp_path / f"r{h.RUN_SUFFIX}").mkdir(parents=True)
    assert h.read_cost_record(tmp_path / f"r{h.RUN_SUFFIX}") is None


def _cost(latency, energy):
    return h.DeployCost(latency_steps=latency, mj_per_sample=energy, spikes=None,
                        energy_proxy_neuron_steps=None, cores=None, s_global=None, acc_deploy=None)


def test_cost_baseline_returns_fastest_analytical_winner_cost(tmp_path):
    runs = [
        h.LocalRun("ctrl_slow", tmp_path, "completed", 0, 0.98, 500.0, None,
                   {"diagnostic_role": "analytical_control"}, "x", _cost(20, 0.05)),
        h.LocalRun("ctrl_fast", tmp_path, "completed", 0, 0.98, 400.0, None,
                   {"diagnostic_role": "analytical_control"}, "x", _cost(11, 0.02)),
    ]
    cost = h.cost_baseline(runs, min_acc=0.97)
    assert cost is not None
    assert cost.latency_steps == 11  # the fastest-WALL control is the canonical baseline
    assert cost.mj_per_sample == 0.02


def test_build_row_emits_relative_deploy_cost(tmp_path):
    run = h.LocalRun("r", tmp_path, "completed", 0, 0.972, 660.0, None,
                     {"firing": "ttfs", "vehicle": "mlp_mixer_core"}, "x", _cost(13, 0.0359))
    rec = h.classify_local_run(run, baseline_wall_s=400.0, min_acc=0.97)
    row = h.build_row(run, rec, baseline_wall_s=400.0, baseline_run_id="ctrl",
                      baseline_cost=_cost(11, 0.02))
    assert row["deploy_cost"]["latency_steps"] == 13
    assert row["deploy_cost"]["mj_per_sample"] == 0.0359
    assert row["relative_deploy_latency"] == 13 / 11
    assert row["relative_deploy_energy"] == 0.0359 / 0.02


def test_build_row_no_cost_leaves_deploy_cost_absent(tmp_path):
    run = h.LocalRun("r", tmp_path, "failed", 1, 0.95, 50.0, "parity failed",
                     {"firing": "ttfs", "vehicle": "mlp_mixer_core"}, "x", None)
    rec = h.classify_local_run(run, baseline_wall_s=400.0, min_acc=0.97)
    row = h.build_row(run, rec, baseline_wall_s=400.0, baseline_run_id="ctrl", baseline_cost=None)
    assert row["deploy_cost"] is None
    assert row["relative_deploy_latency"] is None
    assert row["relative_deploy_energy"] is None


def _deploy_axes(cell_id, recipe_id):
    return {"firing": "ttfs", "vehicle": "mlp_mixer_core", "sync": "cycle",
            "cell_id": cell_id, "recipe_id": recipe_id, "recipe_family": "genuine_blend"}


def test_build_rollup_aggregates_deploy_cost(tmp_path):
    base = _cost(10, 0.02)
    runs = [
        h.LocalRun("a", tmp_path, "completed", 0, 0.972, 600.0, None,
                   _deploy_axes("cell_x", "recipe_x"), "x", _cost(12, 0.030)),
        h.LocalRun("b", tmp_path, "completed", 0, 0.974, 620.0, None,
                   _deploy_axes("cell_x", "recipe_x"), "x", _cost(16, 0.050)),
    ]
    rows = [h.build_row(r, h.classify_local_run(r, baseline_wall_s=400.0, min_acc=0.97),
                        baseline_wall_s=400.0, baseline_run_id="ctrl", baseline_cost=base)
            for r in runs]
    families = h.build_rollup(rows)
    fam = next(f for f in families if f["cell_id"] == "cell_x")
    assert fam["latency_steps"]["min"] == 12
    assert fam["latency_steps"]["max"] == 16
    assert fam["latency_steps"]["mean"] == 14
    assert fam["mj_per_sample"]["min"] == 0.030
    assert fam["relative_deploy_latency"]["min"] == 12 / 10
    assert fam["relative_deploy_latency"]["max"] == 16 / 10
    assert fam["relative_deploy_energy"]["mean"] == ((0.030 / 0.02) + (0.050 / 0.02)) / 2


def test_build_rollup_deploy_cost_absent_yields_none_stats(tmp_path):
    run = h.LocalRun("a", tmp_path, "failed", 1, 0.95, 50.0, "parity failed",
                     _deploy_axes("cell_y", "recipe_y"), "x", None)
    rows = [h.build_row(run, h.classify_local_run(run, baseline_wall_s=400.0, min_acc=0.97),
                        baseline_wall_s=400.0, baseline_run_id="ctrl", baseline_cost=None)]
    fam = next(f for f in h.build_rollup(rows) if f["cell_id"] == "cell_y")
    assert fam["latency_steps"] == {"min": None, "max": None, "mean": None}
    assert fam["relative_deploy_energy"] == {"min": None, "max": None, "mean": None}


# --------------------------------------------------------------------------- family classify


def _family_row(*, returncode, validity, acc=0.96, reason=None):
    return {"returncode": returncode, "deployment_validity": validity,
            "deployed_acc": acc, "failure_reason": reason, "seed": 0}


def test_classify_family_gate_rejected_not_dead():
    """rc=1 parity rejections with HEALTHY accuracy are gate-rejections, never DEAD."""
    rows = [
        _family_row(returncode=1, validity=h.FLAG, acc=0.965, reason="parity_failure"),
        _family_row(returncode=1, validity=h.FLAG, acc=0.959, reason="parity_failure"),
        _family_row(returncode=0, validity=h.FLAG, acc=0.965, reason="accuracy_below_97"),
    ]
    assert h.classify_family(rows) == h.GATE_REJECTED


def test_classify_family_collapse_is_dead_via_per_row_validity():
    """A genuinely collapsed seed (row DEAD) keeps the family MEASURED_DEAD."""
    rows = [
        _family_row(returncode=1, validity=h.DEAD, acc=0.11, reason="measured_dead"),
        _family_row(returncode=1, validity=h.FLAG, acc=0.96, reason="parity_failure"),
    ]
    assert h.classify_family(rows) == h.DEAD


def test_classify_family_all_pass_is_valid_97_fast():
    rows = [_family_row(returncode=0, validity=h.PASS), _family_row(returncode=0, validity=h.PASS)]
    assert h.classify_family(rows) == h.PASS


def test_classify_family_completed_under_target_is_valid_flagged():
    rows = [_family_row(returncode=0, validity=h.PASS),
            _family_row(returncode=0, validity=h.FLAG, reason="accuracy_below_97")]
    assert h.classify_family(rows) == "VALID_FLAGGED"


def test_classify_family_all_running_is_incomplete():
    rows = [_family_row(returncode=None, validity=h.INCOMPLETE, acc=None)]
    assert h.classify_family(rows) == h.INCOMPLETE


# --------------------------------------------------------------------------- end-to-end


def test_harvest_end_to_end(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    index = _manifest(tmp_path)
    base = 1_000_000.0
    # analytical control successes (the local baseline reference)
    _write_run(generated, f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_analytical_control_mixer_ttfs_analytical_control_s0",
               status="completed", started=base, finished=base + 400.0, acc=0.98)
    _write_run(generated, f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_analytical_control_mixer_ttfs_analytical_control_s1",
               status="completed", started=base, finished=base + 500.0, acc=0.985)
    # cascaded genuine -> completed but acc<0.97 (flag)
    _write_run(generated, f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_cascaded_genuine_blend_fast_s0",
               status="completed", started=base, finished=base + 300.0, acc=0.94)
    # cascaded controller -> collapse (dead, derived axes)
    _write_run(generated, f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_controller_baseline_s1",
               status="failed", started=base, finished=base + 200.0, acc=0.1135,
               error="[TTFS Cycle Fine-Tuning] step failed to retain performance")
    # orphaned running -> incomplete (derived axes)
    _write_run(generated, f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_controller_baseline_s2",
               status="running", started=base, finished=None, acc=0.31)

    rows, meta = h.harvest_rows(generated, index, min_acc=0.97)
    by_id = {r["run_id"]: r for r in rows}

    assert meta["n_rows"] == 5
    assert meta["n_incomplete"] == 1
    assert meta["baselines"]["local_fastest_analytical_control_success_wall_s"] == 400.0
    assert meta["baselines"]["canonical_slurmech_analytical_baseline_s"] == 428.0

    cascaded = by_id[f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_cascaded_genuine_blend_fast_s0"]
    assert cascaded["deployment_validity"] == h.FLAG
    assert cascaded["failure_reason"] == "accuracy_below_97"
    assert cascaded["local_relative_time"] == 300.0 / 400.0
    assert cascaded["faster_than_baseline"] is True

    collapse = by_id[f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_controller_baseline_s1"]
    assert collapse["deployment_validity"] == h.DEAD
    assert collapse["axes_source"] == "manifest_cell_derived"

    orphan = by_id[f"{h.RUN_PREFIX}mnist_mmixcore_ttfs_cycle_cascaded_mixer_controller_baseline_s2"]
    assert orphan["deployment_validity"] == h.INCOMPLETE
    assert orphan["returncode"] is None
    assert orphan["run_wall_s"] is None
    assert orphan["relative_time"] is None
    assert orphan["incomplete"] is True
    assert orphan["observed_partial_metric"] == 0.31

    families = h.build_rollup(rows)
    counts = h.rollup_counts(families)
    by_family = {(f["cell_id"], f["recipe_id"]): f for f in families}

    ana = by_family[("mnist_mmixcore_ttfs_analytical_control", "mixer_ttfs_analytical_control")]
    assert ana["classification"] == "VALID_FLAGGED"  # baseline itself sits at rel>=1.0
    assert ana["n_seeds"] == 2

    controller = by_family[("mnist_mmixcore_ttfs_cycle_cascaded", "mixer_controller_baseline")]
    assert controller["classification"] == h.DEAD  # has an rc=1 seed
    assert controller["n_incomplete"] == 1
    assert "research" in controller["owners"]

    assert counts.get(h.DEAD) == 1
