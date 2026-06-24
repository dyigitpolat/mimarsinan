"""Locks the research scheduler: grid instantiation, dedupe, watermark refill."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "campaign"))

import gpu_queue as gq  # noqa: E402
import scheduler as sch  # noqa: E402


def _batch(tmp_template):
    return {
        "id": "b1", "template": tmp_template,
        "base": {"deployment_parameters.enable_sanafe_simulation": False},
        "grid": {"deployment_parameters.model_config.depth": [4, 8],
                 "seed": [0, 1]},
        "id_template": "j_d{depth}_s{seed}", "priority": 20, "tags": {"ws": "T"},
        "enabled": True,
    }


def _write_template(tmp_path):
    t = tmp_path / "tpl.json"
    t.write_text(json.dumps({
        "experiment_name": "x", "seed": 0, "pipeline_mode": "phased",
        "deployment_parameters": {"model_config": {"depth": 8}, "enable_sanafe_simulation": True},
    }))
    return str(t)


def test_set_path_nested():
    d = {}
    sch.set_path(d, "a.b.c", 5)
    assert d == {"a": {"b": {"c": 5}}}


def test_instantiate_expands_grid_and_sets_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "REPO", str(tmp_path))
    tpl = _write_template(tmp_path)
    rel = os.path.relpath(tpl, str(tmp_path))
    jobs = list(sch.instantiate({**_batch(rel)}))
    assert {j for j, _ in jobs} == {"j_d4_s0", "j_d4_s1", "j_d8_s0", "j_d8_s1"}
    cfg = dict(jobs)["j_d4_s1"]
    assert cfg["deployment_parameters"]["model_config"]["depth"] == 4
    assert cfg["seed"] == 1
    assert cfg["deployment_parameters"]["enable_sanafe_simulation"] is False  # base override
    assert cfg["experiment_name"] == "j_d4_s1"


def test_refill_respects_watermark_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "REPO", str(tmp_path))
    monkeypatch.setattr(sch, "CFG_DIR", str(tmp_path / "cfg"))
    os.makedirs(str(tmp_path / "cfg"), exist_ok=True)
    tpl = _write_template(tmp_path)
    rel = os.path.relpath(tpl, str(tmp_path))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([_batch(rel)]))
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=3, poll=0, backlog_path=str(backlog))
    # first refill: 4 grid jobs but hi=3 -> only 3 enqueued
    assert s.refill() == 3 and q.counts()["pending"] == 3
    # drain one (claim+finish) -> room for 1 more, and it must be the un-enqueued 4th (dedupe)
    _, p = q.claim_next(); q.finish(p, {"returncode": 0}, success=True)
    added = s.refill()
    assert added == 1
    allids = {j["id"] for j in q.list_state("pending")} | {j["id"] for j in q.list_state("done")}
    assert allids == {"j_d4_s0", "j_d4_s1", "j_d8_s0", "j_d8_s1"}  # all 4, none duplicated


def test_refill_skips_a_malformed_batch_without_crashing(tmp_path, monkeypatch):
    """One bad batch (e.g. a synthesize-agent id_template with a dotted-path
    placeholder) must not kill the daemon — it is logged and skipped, good
    batches still enqueue."""
    monkeypatch.setattr(sch, "REPO", str(tmp_path))
    monkeypatch.setattr(sch, "CFG_DIR", str(tmp_path / "cfg"))
    tpl = _write_template(tmp_path)
    rel = os.path.relpath(tpl, str(tmp_path))
    good = _batch(rel)
    bad = {**_batch(rel), "id": "b_bad",
           "id_template": "x_{deployment_parameters.model_config.depth}_s{seed}"}
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([bad, good]))
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()  # must NOT raise on the malformed batch
    assert added == 4  # only the good batch's 2x2 grid enqueued
    assert {j["id"] for j in q.list_state("pending")} == {"j_d4_s0", "j_d4_s1", "j_d8_s0", "j_d8_s1"}


def test_disabled_batch_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "REPO", str(tmp_path))
    monkeypatch.setattr(sch, "CFG_DIR", str(tmp_path / "cfg"))
    tpl = _write_template(tmp_path)
    rel = os.path.relpath(tpl, str(tmp_path))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([{**_batch(rel), "enabled": False}]))
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    assert s.refill() == 0 and q.counts()["pending"] == 0
