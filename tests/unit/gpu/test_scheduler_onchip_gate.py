"""Scheduler enqueue pre-check: a host-majority job must NOT claim a GPU.

Before enqueuing each instantiated job, the scheduler statically estimates the
model's on-chip parameter fraction (no GPU, no run) and SKIPS host-majority jobs
so they never reach the queue. A model-build failure is non-fatal: the job is
enqueued anyway so a builder edge case never silently drops valid work.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "campaign"))

import gpu_queue as gq  # noqa: E402
import scheduler as sch  # noqa: E402


class _FakeEstimate:
    def __init__(self, fraction):
        self.fraction = fraction
        self.onchip = int(fraction * 1000)
        self.host = 1000 - self.onchip
        self.total = 1000
        self.metric = "params"
        self.placement = "subsume"


def _template(tmp_path, model_type):
    t = tmp_path / f"tpl_{model_type}.json"
    t.write_text(json.dumps({
        "experiment_name": "x", "seed": 0, "pipeline_mode": "phased",
        "data_provider_name": "MNIST_DataProvider",
        "deployment_parameters": {
            "model_type": model_type,
            "model_config": {"depth": 8, "width": 64},
        },
    }))
    return str(t)


def _batch(rel, bid):
    return {
        "id": bid, "template": rel, "base": {}, "grid": {"seed": [0]},
        "id_template": f"{bid}_s{{seed}}", "priority": 20, "enabled": True,
    }


def _common_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "REPO", str(tmp_path))
    monkeypatch.setattr(sch, "CFG_DIR", str(tmp_path / "cfg"))
    os.makedirs(str(tmp_path / "cfg"), exist_ok=True)


def test_host_majority_job_is_not_enqueued_but_valid_is(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    bad_tpl = os.path.relpath(_template(tmp_path, "host_heavy"), str(tmp_path))
    good_tpl = os.path.relpath(_template(tmp_path, "chip_heavy"), str(tmp_path))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([_batch(bad_tpl, "bad"), _batch(good_tpl, "good")]))

    # The model-build is patched: host_heavy is host-majority (0.20), chip_heavy
    # is on-chip-majority (0.80). No real datasets / converters are touched.
    def _fake_estimate(cfg):
        if cfg["deployment_parameters"]["model_type"] == "host_heavy":
            return _FakeEstimate(0.20)
        return _FakeEstimate(0.80)

    monkeypatch.setattr(sch, "_estimate_cfg_onchip_fraction", _fake_estimate)

    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()

    pending_ids = {j["id"] for j in q.list_state("pending")}
    assert "good_s0" in pending_ids
    assert "bad_s0" not in pending_ids
    assert added == 1


def test_gate_disabled_enqueues_host_majority(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    tpl = os.path.relpath(_template(tmp_path, "host_heavy"), str(tmp_path))
    backlog = tmp_path / "backlog.json"
    b = _batch(tpl, "bad")
    # turn the gate off in the config
    bad_template = json.loads((tmp_path / "tpl_host_heavy.json").read_text())
    bad_template["deployment_parameters"]["onchip_majority_gate"] = False
    (tmp_path / "tpl_host_heavy.json").write_text(json.dumps(bad_template))
    backlog.write_text(json.dumps([b]))

    monkeypatch.setattr(
        sch, "_estimate_cfg_onchip_fraction", lambda cfg: _FakeEstimate(0.05)
    )
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()
    assert added == 1
    assert {j["id"] for j in q.list_state("pending")} == {"bad_s0"}


def test_build_failure_is_non_fatal_and_enqueues(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    tpl = os.path.relpath(_template(tmp_path, "broken"), str(tmp_path))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([_batch(tpl, "broken")]))

    def _boom(cfg):
        raise RuntimeError("builder blew up")

    monkeypatch.setattr(sch, "_estimate_cfg_onchip_fraction", _boom)
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()
    assert added == 1
    assert {j["id"] for j in q.list_state("pending")} == {"broken_s0"}


def test_custom_floor_respected(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    tpl = os.path.relpath(_template(tmp_path, "mid"), str(tmp_path))
    # raise the floor to 0.9 so a 0.80 model is rejected
    template = json.loads((tmp_path / "tpl_mid.json").read_text())
    template["deployment_parameters"]["onchip_majority_min_fraction"] = 0.9
    (tmp_path / "tpl_mid.json").write_text(json.dumps(template))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([_batch(tpl, "mid")]))

    monkeypatch.setattr(
        sch, "_estimate_cfg_onchip_fraction", lambda cfg: _FakeEstimate(0.80)
    )
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()
    assert added == 0
    assert q.counts()["pending"] == 0
