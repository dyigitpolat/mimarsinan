"""Scheduler enqueue capacity pre-check (E4): reject provably-infeasible configs.

Before enqueuing a job, the scheduler statically estimates the hard cores its IR
needs vs. its declared core budget (no GPU, no placement, no run). It REJECTS
(``False`` — no GPU claimed) a config whose SOUND lower bound exceeds the budget
(VGG16@224 on a 1000-core budget) and ADMITS a feasible one. Model-build / IR /
estimate failures are NON-FATAL — the job is enqueued anyway so a builder edge
case never silently drops valid work. Opt-out via
``deployment_parameters.capacity_gate=False``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "campaign"))

import gpu_queue as gq  # noqa: E402
import scheduler as sch  # noqa: E402

from mimarsinan.mapping.verification.capacity import (  # noqa: E402
    CapacityEstimate,
    PackFeasibility,
)


def _feasible_estimate():
    return CapacityEstimate(
        cores_needed=830, cores_available=2048, feasible=True,
        overflowing_segment=None, per_segment={"seg": 830},
    )


def _infeasible_estimate():
    return CapacityEstimate(
        cores_needed=315816, cores_available=1000, feasible=False,
        overflowing_segment="neural_segment_until:features_6",
        per_segment={"neural_segment_until:features_6": 112896},
    )


def _scheduled_feasible_estimate():
    """VGG16@224-shaped verdict: SUM huge but feasible-via-scheduling (PEAK fits)."""
    return CapacityEstimate(
        cores_needed=315816, cores_available=2048, feasible=True,
        overflowing_segment=None,
        per_segment={"neural_segment_until:features_6": 112896},
        scheduled=True, peak_phase_cores=2048, phase_count=155,
    )


def test_capacity_precheck_admits_scheduled_feasible():
    cfg = {"deployment_parameters": {"model_type": "torch_vgg16"}}
    import unittest.mock as mock
    with mock.patch.object(
        sch, "_estimate_cfg_capacity", lambda c: _scheduled_feasible_estimate()
    ), mock.patch.object(sch, "_dryrun_cfg_packing", lambda c: _pack_feasible()):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "ok"
    assert info["feasible"] is True
    assert info["scheduled"] is True
    assert info["peak_phase_cores"] == 2048
    assert info["phase_count"] == 155


def test_capacity_precheck_admits_feasible():
    cfg = {"deployment_parameters": {"model_type": "lenet5"}}
    import unittest.mock as mock
    with mock.patch.object(
        sch, "_estimate_cfg_capacity", lambda c: _feasible_estimate()
    ), mock.patch.object(sch, "_dryrun_cfg_packing", lambda c: _pack_feasible()):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "ok"
    assert info["feasible"] is True
    assert info["cores_needed"] == 830
    assert info["cores_available"] == 2048


def test_capacity_precheck_rejects_infeasible_naming_segment():
    cfg = {"deployment_parameters": {"model_type": "torch_vgg16"}}
    import unittest.mock as mock
    with mock.patch.object(
        sch, "_estimate_cfg_capacity", lambda c: _infeasible_estimate()
    ):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is False
    assert info["reason"] == "capacity_exceeded"
    assert info["cores_needed"] == 315816
    assert info["cores_available"] == 1000
    assert info["overflowing_segment"] == "neural_segment_until:features_6"


def test_capacity_precheck_gate_disabled_admits_infeasible():
    cfg = {"deployment_parameters": {"model_type": "torch_vgg16", "capacity_gate": False}}
    import unittest.mock as mock
    with mock.patch.object(
        sch, "_estimate_cfg_capacity", lambda c: _infeasible_estimate()
    ):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "gate_disabled"


def _pack_feasible():
    return PackFeasibility(
        feasible=True, hard_cores=312, overflowing_segment=None, error=None,
    )


def _pack_infeasible():
    return PackFeasibility(
        feasible=False, hard_cores=None,
        overflowing_segment="neural_segment_until:features_13",
        error="Hard-core packing failed in segment "
        "'neural_segment_until:features_13': No more hard cores available.",
    )


def test_capacity_precheck_rejects_pack_infeasible_after_lower_bound_admits():
    """The SOUND lower bound admits (cores fit ideally) but the real packer cannot
    place it (threshold-group fragmentation overflows) → reject, no GPU claimed."""
    cfg = {"deployment_parameters": {"model_type": "deep_cnn"}}
    import unittest.mock as mock
    with mock.patch.object(sch, "_estimate_cfg_capacity", lambda c: _feasible_estimate()), \
         mock.patch.object(sch, "_dryrun_cfg_packing", lambda c: _pack_infeasible()):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is False
    assert info["reason"] == "pack_infeasible"
    assert info["overflowing_segment"] == "neural_segment_until:features_13"
    assert "No more hard cores available" in info["pack_error"]


def test_capacity_precheck_admits_pack_feasible():
    cfg = {"deployment_parameters": {"model_type": "deep_cnn"}}
    import unittest.mock as mock
    with mock.patch.object(sch, "_estimate_cfg_capacity", lambda c: _feasible_estimate()), \
         mock.patch.object(sch, "_dryrun_cfg_packing", lambda c: _pack_feasible()):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "ok"
    assert info["packed_hard_cores"] == 312


def test_capacity_precheck_lower_bound_rejection_skips_dryrun():
    """A config the lower bound already rejects must not pay for a real-packer dry-run."""
    cfg = {"deployment_parameters": {"model_type": "torch_vgg16"}}

    def _must_not_run(c):
        raise AssertionError("dry-run must not run when the lower bound rejects")

    import unittest.mock as mock
    with mock.patch.object(sch, "_estimate_cfg_capacity", lambda c: _infeasible_estimate()), \
         mock.patch.object(sch, "_dryrun_cfg_packing", _must_not_run):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is False
    assert info["reason"] == "capacity_exceeded"


def test_capacity_precheck_dryrun_gate_disabled_skips_dryrun():
    """capacity_dryrun_gate=False keeps only the cheap lower bound; the dry-run is skipped."""
    cfg = {"deployment_parameters": {"model_type": "deep_cnn", "capacity_dryrun_gate": False}}

    def _must_not_run(c):
        raise AssertionError("dry-run must not run when capacity_dryrun_gate=False")

    import unittest.mock as mock
    with mock.patch.object(sch, "_estimate_cfg_capacity", lambda c: _feasible_estimate()), \
         mock.patch.object(sch, "_dryrun_cfg_packing", _must_not_run):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "ok"


def test_capacity_precheck_dryrun_failure_is_non_fatal():
    """An unexpected dry-run error never drops valid work — the job is admitted."""
    cfg = {"deployment_parameters": {"model_type": "deep_cnn"}}

    def _boom(c):
        raise RuntimeError("packer import blew up")

    import unittest.mock as mock
    with mock.patch.object(sch, "_estimate_cfg_capacity", lambda c: _feasible_estimate()), \
         mock.patch.object(sch, "_dryrun_cfg_packing", _boom):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "dryrun_failed"
    assert "packer import blew up" in info["dryrun_error"]


def test_capacity_precheck_gate_disabled_skips_both_checks():
    """capacity_gate=False short-circuits before the estimate AND the dry-run."""
    cfg = {"deployment_parameters": {"model_type": "deep_cnn", "capacity_gate": False}}

    def _must_not_run(c):
        raise AssertionError("no capacity check may run when capacity_gate=False")

    import unittest.mock as mock
    with mock.patch.object(sch, "_estimate_cfg_capacity", _must_not_run), \
         mock.patch.object(sch, "_dryrun_cfg_packing", _must_not_run):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "gate_disabled"


def test_capacity_precheck_estimate_failure_is_non_fatal():
    cfg = {"deployment_parameters": {"model_type": "broken"}}

    def _boom(c):
        raise RuntimeError("IR build blew up")

    import unittest.mock as mock
    with mock.patch.object(sch, "_estimate_cfg_capacity", _boom):
        ok, info = sch.capacity_precheck(cfg)
    assert ok is True
    assert info["reason"] == "estimate_failed"
    assert "IR build blew up" in info["error"]


def _template(tmp_path, model_type):
    t = tmp_path / f"tpl_{model_type}.json"
    t.write_text(json.dumps({
        "experiment_name": "x", "seed": 0, "pipeline_mode": "phased",
        "data_provider_name": "MNIST_DataProvider",
        "deployment_parameters": {"model_type": model_type, "model_config": {}},
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
    # neutralize the on-chip-fraction gate so only the capacity gate decides
    monkeypatch.setattr(sch, "onchip_precheck", lambda cfg: (True, {"reason": "ok"}))


def test_refill_rejects_infeasible_admits_feasible(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    inf_tpl = os.path.relpath(_template(tmp_path, "torch_vgg16"), str(tmp_path))
    fea_tpl = os.path.relpath(_template(tmp_path, "lenet5"), str(tmp_path))
    backlog = tmp_path / "backlog.json"
    backlog.write_text(json.dumps([_batch(inf_tpl, "vgg"), _batch(fea_tpl, "lenet")]))

    def _fake_cap(cfg):
        mt = cfg["deployment_parameters"]["model_type"]
        return _infeasible_estimate() if mt == "torch_vgg16" else _feasible_estimate()

    monkeypatch.setattr(sch, "_estimate_cfg_capacity", _fake_cap)
    q = gq.GpuQueue(str(tmp_path / "q"))
    s = sch.Scheduler(q, hi=10, poll=0, backlog_path=str(backlog))
    added = s.refill()

    pending_ids = {j["id"] for j in q.list_state("pending")}
    assert "lenet_s0" in pending_ids       # feasible admitted
    assert "vgg_s0" not in pending_ids     # infeasible rejected, no GPU claimed
    assert added == 1
