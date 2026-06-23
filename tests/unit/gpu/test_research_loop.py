"""Locks the research-loop primitives (enqueue / wait / results / ledger)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "campaign"))

import gpu_queue as gq  # noqa: E402
import research_loop as rl  # noqa: E402


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MIM_CAMPAIGN_DIR", str(tmp_path))
    # rl computes LEDGER + queue root at import; repoint both to tmp
    rl.LEDGER = str(tmp_path / "ledger.jsonl")
    return gq.GpuQueue(str(tmp_path / "q"))


def test_enqueue_then_results_reports_finalized(tmp_path, monkeypatch):
    q = _setup(tmp_path, monkeypatch)
    # one done (with a deployed-acc artifact), one failed
    art = tmp_path / "m.json"; art.write_text("0.93")
    q.enqueue({"id": "a", "cmd": ["true"], "expect_artifact": str(art), "tags": {"depth": 8}})
    q.enqueue({"id": "b", "cmd": ["true"]})
    _, pa = q.claim_next(); _, pb = q.claim_next()
    # finalize a=done b=failed (order-independent: find by id)
    for p in (pa, pb):
        jid = json.load(open(p))["id"]
        q.finish(p, {"returncode": 0 if jid == "a" else 1}, success=(jid == "a"))
    idx = rl._index(q)
    assert idx["a"]["state"] == "done" and idx["a"]["deployed_acc"] == 0.93
    assert idx["b"]["state"] == "failed" and idx["b"]["returncode"] == 1
    assert idx["a"]["tags"]["depth"] == 8


def test_wait_returns_when_all_finalized(tmp_path, monkeypatch, capsys):
    q = _setup(tmp_path, monkeypatch)
    q.enqueue({"id": "x", "cmd": ["true"]})
    _, p = q.claim_next(); q.finish(p, {"returncode": 0}, success=True)
    monkeypatch.setattr(gq, "GpuQueue", lambda *a, **k: q)
    rc = rl.cmd_wait(type("A", (), {"ids": ["x"], "timeout": 5, "poll": 0.01})())
    assert rc == 0 and json.loads(capsys.readouterr().out)["all_done"] is True


def test_wait_times_out_on_missing(tmp_path, monkeypatch, capsys):
    q = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(gq, "GpuQueue", lambda *a, **k: q)
    rc = rl.cmd_wait(type("A", (), {"ids": ["never"], "timeout": 0.05, "poll": 0.01})())
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["timed_out"] is True and out["missing"] == ["never"]


def test_ledger_append_and_read_filters_by_cluster(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    rl.cmd_ledger_append(type("A", (), {"record": json.dumps(
        [{"cluster": "WS3", "exp": "d8", "deployed_acc": 0.89},
         {"cluster": "WS6", "exp": "ci", "deployed_acc": 0.97}])})())
    capsys.readouterr()
    rl.cmd_ledger_read(type("A", (), {"cluster": "WS3"})())
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1 and rows[0]["exp"] == "d8"
