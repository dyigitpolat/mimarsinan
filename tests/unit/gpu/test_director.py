"""Locks the research DIRECTOR: staged-plan rollout + harvest-todo gap detection.

The director keeps the campaign advancing unattended — it enables the next disabled
"plan" batch when enabled work runs low (so GPUs never starve), and surfaces
finalized-but-unanalyzed cells into harvest_todo (so consolidation knows what's
waiting). It never writes verdicts. No real GPU; queue states injected on disk.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "campaign"))

import gpu_queue as gq  # noqa: E402
import director as dr  # noqa: E402


def _write_template(tmp_path):
    t = tmp_path / "tpl.json"
    t.write_text(json.dumps({
        "experiment_name": "x", "seed": 0, "pipeline_mode": "phased",
        "deployment_parameters": {"model_config": {"depth": 8}},
    }))
    return str(t)


def _batch(template, bid, *, enabled, plan_stage=None, depths=(4, 8), seeds=(0, 1)):
    b = {
        "id": bid, "template": template,
        "grid": {"deployment_parameters.model_config.depth": list(depths), "seed": list(seeds)},
        "id_template": bid + "_d{depth}_s{seed}", "priority": 20, "enabled": enabled,
    }
    if plan_stage is not None:
        b["plan_stage"] = plan_stage
    return b


def _finalize(q, jid, tags, deployed, state="done"):
    """Drop a finalized job record straight into done/ or failed/ with an artifact."""
    rec = {"id": jid, "tags": tags, "result": {"returncode": 0 if state == "done" else 1},
           "expect_artifact": None}
    with open(os.path.join(q._dir(state), jid + ".json"), "w") as fh:
        json.dump(rec, fh)


# --------------------------------------------------------------------------- #
# enabled_remaining: un-enqueued enabled work (the fuel gauge)
# --------------------------------------------------------------------------- #

def test_enabled_remaining_counts_only_un_enqueued_enabled_points(tmp_path, monkeypatch):
    monkeypatch.setattr(dr.sch, "REPO", str(tmp_path))
    tpl = _write_template(tmp_path)
    rel = os.path.relpath(tpl, str(tmp_path))
    q = gq.GpuQueue(str(tmp_path / "q"))
    backlog = [_batch(rel, "b_on", enabled=True),          # 2x2 = 4 points
               _batch(rel, "b_off", enabled=False, plan_stage=0)]  # disabled -> not counted
    # one of b_on's points already enqueued -> excluded
    q.enqueue({"id": "b_on_d4_s0", "cmd": ["true"]})
    rem = dr.enabled_remaining(backlog, dr.sch.existing_ids(q))
    assert rem == 3  # 4 enabled points - 1 already enqueued; disabled batch ignored


# --------------------------------------------------------------------------- #
# plan rollout: enable the next disabled plan batch when work runs low
# --------------------------------------------------------------------------- #

def test_tick_enables_next_plan_stage_when_work_runs_low(tmp_path, monkeypatch):
    monkeypatch.setattr(dr.sch, "REPO", str(tmp_path))
    tpl = _write_template(tmp_path)
    rel = os.path.relpath(tpl, str(tmp_path))
    blp = tmp_path / "backlog.json"
    blp.write_text(json.dumps([
        _batch(rel, "b_done", enabled=True, depths=(4,), seeds=(0,)),  # 1 enabled point
        _batch(rel, "stage1", enabled=False, plan_stage=1),
        _batch(rel, "stage2", enabled=False, plan_stage=2),
    ]))
    q = gq.GpuQueue(str(tmp_path / "q"))
    # the single enabled point is already done -> enabled_remaining = 0 (< lo) -> roll out
    _finalize(q, "b_done_d4_s0", {}, 0.9)
    d = dr.Director(q, backlog_path=str(blp), lo=2, poll=0,
                    todo_path=str(tmp_path / "todo.json"), status_path=str(tmp_path / "st.json"))
    res = d.tick()
    assert res["enabled_plan_batch"] == "stage1"  # lowest disabled plan_stage first
    bl = json.load(open(blp))
    assert {b["id"]: b["enabled"] for b in bl}["stage1"] is True
    assert {b["id"]: b["enabled"] for b in bl}["stage2"] is False  # only one per tick


def test_tick_does_not_roll_out_when_work_remains(tmp_path, monkeypatch):
    monkeypatch.setattr(dr.sch, "REPO", str(tmp_path))
    tpl = _write_template(tmp_path)
    rel = os.path.relpath(tpl, str(tmp_path))
    blp = tmp_path / "backlog.json"
    blp.write_text(json.dumps([
        _batch(rel, "b_on", enabled=True, depths=(4, 8), seeds=(0, 1)),  # 4 enabled points, none done
        _batch(rel, "stage1", enabled=False, plan_stage=1),
    ]))
    q = gq.GpuQueue(str(tmp_path / "q"))
    d = dr.Director(q, backlog_path=str(blp), lo=2, poll=0,
                    todo_path=str(tmp_path / "todo.json"), status_path=str(tmp_path / "st.json"))
    res = d.tick()
    assert res["enabled_plan_batch"] is None  # 4 >= lo=2 -> no rollout
    bl = json.load(open(blp))
    assert {b["id"]: b["enabled"] for b in bl}["stage1"] is False


# --------------------------------------------------------------------------- #
# harvest_todo: finalized cells whose runs are NOT all cited in the ledger
# --------------------------------------------------------------------------- #

def test_harvest_todo_flags_uncovered_finalized_cells(tmp_path):
    q = gq.GpuQueue(str(tmp_path / "q"))
    # cell A: two seeds finalized, NOT in ledger -> flagged
    _finalize(q, "a_s0", {"model": "deep_cnn", "dataset": "mnist", "sched": "cascaded"}, 0.9)
    _finalize(q, "a_s1", {"model": "deep_cnn", "dataset": "mnist", "sched": "cascaded"}, 0.91)
    # cell B: finalized AND already cited in ledger -> not flagged
    _finalize(q, "b_s0", {"model": "lenet5", "dataset": "mnist", "sched": "cascaded"}, 0.98)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"cluster": "WS3", "cascaded_run_ids": ["b_s0"]}) + "\n")
    todo = dr.harvest_todo(q, str(ledger))
    keys = {t["cell"] for t in todo}
    assert "deep_cnn/mnist/cascaded" in keys
    assert "lenet5/mnist/cascaded" not in keys  # already consolidated
    a = [t for t in todo if t["cell"] == "deep_cnn/mnist/cascaded"][0]
    assert sorted(a["run_ids"]) == ["a_s0", "a_s1"] and a["n_finalized"] == 2


def test_harvest_todo_ignores_cells_with_partial_ledger_coverage(tmp_path):
    """A cell counts as covered if ANY of its runs is already cited (the harvest
    row references the cell); avoids re-flagging a consolidated cell forever."""
    q = gq.GpuQueue(str(tmp_path / "q"))
    _finalize(q, "c_s0", {"model": "deep_mlp", "dataset": "mnist", "sched": "synchronized", "depth": 8}, 0.96)
    _finalize(q, "c_s1", {"model": "deep_mlp", "dataset": "mnist", "sched": "synchronized", "depth": 8}, 0.96)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"synchronized_run_ids": ["c_s0"]}) + "\n")
    todo = dr.harvest_todo(q, str(ledger))
    assert todo == []  # c_s0 cited -> the deep_mlp/mnist/synchronized cell is covered
