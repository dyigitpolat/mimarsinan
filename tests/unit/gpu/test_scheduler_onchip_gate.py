"""Scheduler enqueue pre-check: the TIERED validity gate (v2).

Before enqueuing each instantiated job, the scheduler statically classifies the
model's tiered validity (no GPU, no run) on BOTH params and MACs. It REJECTS only
INVALID jobs (``min(param,mac)`` on-chip below the 20% floor — host does
~everything) so they never claim a GPU; it ADMITS VALID and VALID_FLAGGED, logging
the research gap for flagged jobs. A model-build failure is non-fatal: the job is
enqueued anyway so a builder edge case never silently drops valid work.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "campaign"))

import gpu_queue as gq  # noqa: E402
import scheduler as sch  # noqa: E402


class _FakeVerdict:
    """Stand-in for ``ValidityVerdict`` so the gate test never builds a real model."""

    def __init__(self, tier, param_frac, mac_frac, research_gap_ops=None,
                 placement_fixable_ops=None):
        self.tier = tier
        self.param_frac = param_frac
        self.mac_frac = mac_frac
        self.research_gap_ops = list(research_gap_ops or [])
        self.placement_fixable_ops = list(placement_fixable_ops or [])


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


def test_invalid_rejected_but_valid_and_flagged_admitted(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    inv_tpl = os.path.relpath(_template(tmp_path, "host_heavy"), str(tmp_path))
    flag_tpl = os.path.relpath(_template(tmp_path, "flagged"), str(tmp_path))
    good_tpl = os.path.relpath(_template(tmp_path, "chip_heavy"), str(tmp_path))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([
        _batch(inv_tpl, "inv"), _batch(flag_tpl, "flag"), _batch(good_tpl, "good"),
    ]))

    # Classification is patched: host_heavy is INVALID (0.10 < floor), flagged is
    # VALID_FLAGGED (0.33 in [floor, majority)), chip_heavy is VALID (0.80). No real
    # datasets / converters / models are touched.
    def _fake_classify(cfg, *, floor, majority):
        mt = cfg["deployment_parameters"]["model_type"]
        if mt == "host_heavy":
            return _FakeVerdict("INVALID", 0.10, 0.10)
        if mt == "flagged":
            return _FakeVerdict(
                "VALID_FLAGGED", 0.33, 0.33,
                research_gap_ops=["MultiheadAttention", "LayerNorm"],
            )
        return _FakeVerdict("VALID", 0.80, 0.80)

    monkeypatch.setattr(sch, "_classify_cfg_validity", _fake_classify)

    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()

    pending_ids = {j["id"] for j in q.list_state("pending")}
    assert "good_s0" in pending_ids       # VALID admitted
    assert "flag_s0" in pending_ids       # VALID_FLAGGED admitted (deploys + flagged)
    assert "inv_s0" not in pending_ids    # INVALID rejected, no GPU claimed
    assert added == 2


def test_precheck_flagged_carries_research_gap_and_admits():
    cfg = {
        "deployment_parameters": {
            "model_type": "torch_vit",
            "onchip_majority_gate": True,
        }
    }

    def _fake_classify(c, *, floor, majority):
        assert floor == 0.20 and majority == 0.50
        return _FakeVerdict(
            "VALID_FLAGGED", 0.33, 0.33,
            research_gap_ops=["MultiheadAttention", "LayerNorm"],
            placement_fixable_ops=[],
        )

    import unittest.mock as mock
    with mock.patch.object(sch, "_classify_cfg_validity", _fake_classify):
        ok, info = sch.onchip_precheck(cfg)
    assert ok is True
    assert info["reason"] == "valid_flagged"
    assert info["tier"] == "VALID_FLAGGED"
    assert "MultiheadAttention" in info["research_gap_ops"]


def test_precheck_invalid_below_floor_rejected():
    cfg = {"deployment_parameters": {"model_type": "host_heavy"}}

    import unittest.mock as mock
    with mock.patch.object(
        sch, "_classify_cfg_validity",
        lambda c, *, floor, majority: _FakeVerdict("INVALID", 0.05, 0.05),
    ):
        ok, info = sch.onchip_precheck(cfg)
    assert ok is False
    assert info["reason"] == "invalid_host_majority"


def test_gate_disabled_enqueues_invalid(tmp_path, monkeypatch):
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
        sch, "_classify_cfg_validity",
        lambda cfg, *, floor, majority: _FakeVerdict("INVALID", 0.05, 0.05),
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

    def _boom(cfg, *, floor, majority):
        raise RuntimeError("builder blew up")

    monkeypatch.setattr(sch, "_classify_cfg_validity", _boom)
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()
    assert added == 1
    assert {j["id"] for j in q.list_state("pending")} == {"broken_s0"}


def test_custom_floor_rejects_flagged_when_raised(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    tpl = os.path.relpath(_template(tmp_path, "mid"), str(tmp_path))
    # raise the floor to 0.4 so a 0.33/0.33 model now falls BELOW floor -> INVALID
    template = json.loads((tmp_path / "tpl_mid.json").read_text())
    template["deployment_parameters"]["onchip_min_fraction"] = 0.4
    (tmp_path / "tpl_mid.json").write_text(json.dumps(template))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([_batch(tpl, "mid")]))

    def _fake_classify(cfg, *, floor, majority):
        # honour the configured floor: 0.33 < 0.4 -> INVALID
        tier = "INVALID" if min(0.33, 0.33) < floor else "VALID_FLAGGED"
        return _FakeVerdict(tier, 0.33, 0.33)

    monkeypatch.setattr(sch, "_classify_cfg_validity", _fake_classify)
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()
    assert added == 0
    assert q.counts()["pending"] == 0
