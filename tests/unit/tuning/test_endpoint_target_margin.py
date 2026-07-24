"""[§17.9 Arm C] endpoint target margin: bank measured downstream conversion
debt at an endpoint instead of anchoring at the incoming envelope.

Anchoring the target at the D-hat high-water (the origin) GUARANTEES ending
delta below origin, where delta is the measured cost of everything downstream
(t2_04: AQ install -0.96pp, WQ -0.28pp, LIF +0.20pp => 1.04pp net). The margin
lifts the endpoint target deliberately; the rollback guard and keep-best
selection are untouched, so an unreachable margin costs budget, never accuracy.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "tests/unit/tuning")

from mimarsinan.tuning.orchestration import dhat_highwater, retention_envelope
from mimarsinan.tuning.orchestration.frontier import endpoint_recovery
from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import (
    run_endpoint_recovery,
)
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from test_endpoint_recovery import _lif_tuner, _prepare_endpoint_scaffold


def _run(tmp_path, monkeypatch, *, margin=None, floor=None, envelope=None,
         highwater=0.86, entry=0.86):
    tuner = _lif_tuner(tmp_path)
    seen = {}
    try:
        _prepare_endpoint_scaffold(tuner)
        if margin is not None:
            tuner.pipeline.config["endpoint_target_margin"] = margin
        if floor is not None:
            tuner.pipeline.config["endpoint_target_floor"] = floor
        if envelope is not None:
            retention_envelope.seed(tuner.pipeline, envelope)
        dhat_highwater.observe(tuner.pipeline, highwater)
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: entry,
        )

        def fake_train(trainer, lr, target, *, max_steps, **kwargs):
            seen.update(target=target, max_steps=max_steps)
            return entry, 1

        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(fake_train),
        )
        report = run_endpoint_recovery(tuner, base_steps=100)
    finally:
        tuner.close()
    return report, seen


def test_no_margin_is_byte_identical(tmp_path, monkeypatch):
    report, _seen = _run(tmp_path, monkeypatch, highwater=0.86, entry=0.80)
    assert report.target == pytest.approx(0.86)
    assert report.target_margin == pytest.approx(0.0)


def test_margin_lifts_the_target_above_the_highwater(tmp_path, monkeypatch):
    report, seen = _run(
        tmp_path, monkeypatch, margin=0.0104, highwater=0.8687, entry=0.8687,
    )
    assert report.target == pytest.approx(0.8791)
    assert report.target_margin == pytest.approx(0.0104)
    # Entry sits AT the old target, so only the margin can engage the stage.
    assert report.engaged is True
    assert seen["target"] == pytest.approx(0.8791)


def test_margin_is_not_capped_by_the_retention_envelope(tmp_path, monkeypatch):
    """Absolute FLOORS may never exceed the incoming clean envelope; a margin
    is an explicit request to train past it (that is the entire point)."""
    report, _seen = _run(
        tmp_path, monkeypatch, margin=0.01, floor=0.99, envelope=0.87,
        highwater=0.87, entry=0.86,
    )
    assert report.target == pytest.approx(0.88)
    assert report.target_floor == pytest.approx(0.99)


def test_unreachable_margin_never_costs_accuracy(tmp_path, monkeypatch):
    """The exit read is the entry when training cannot improve: the endpoint
    stays non-destructive regardless of how ambitious the margin is."""
    report, _seen = _run(
        tmp_path, monkeypatch, margin=0.05, highwater=0.86, entry=0.86,
    )
    assert report.reached is False
    assert report.exit == pytest.approx(0.86)
    assert report.rolled_back is False


def test_knob_registered_default_zero():
    from mimarsinan.config_schema.registry import effective_value

    assert effective_value({}, "endpoint_target_margin") == pytest.approx(0.0)
    assert effective_value(
        {"endpoint_target_margin": 0.0104}, "endpoint_target_margin",
    ) == pytest.approx(0.0104)
