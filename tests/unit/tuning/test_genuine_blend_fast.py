"""EXPERIMENTAL fast fixed-increment path for the genuine teacher->cascade blend ramp.

With ``ttfs_genuine_blend_ramp=True`` AND ``ttfs_genuine_blend_fast=True`` (cascaded
only) ``TTFSCycleAdaptationTuner.run()`` SKIPS the heavy SmoothAdaptation controller
entirely. Instead it walks a FIXED rate schedule
(``ttfs_blend_fast_rates``, default ``[0.5, 0.75, 0.9, 0.97, 1.0]``), training a FIXED
number of steps per rate (``ttfs_blend_fast_steps_per_rate``, default 120) with ONE
Adam optimizer + warmup/cosine LR schedule, loss
``CE((1-R)*teacher + R*genuine) + 0.3*CE(genuine)`` — mirroring the validated
prototype (``generated/_genuine_ab/full_ramp.py``). It then deploys the PURE genuine
cascade (the existing finalize), commits rate 1.0, and reports a final metric.

These are MECHANISM tests: with the flag ON the fast path runs (NO scheduler / NO
``_adaptation`` cycles — ``_cycle_log`` stays empty, a fast marker is set, EXACTLY
``len(rates) * steps_per_rate`` optimizer steps ran), the committed rate ends at 1.0,
the deployed forward is the pure genuine single-spike cascade, and ``validate()``
returns a metric. With the flag OFF every other test in the suite (golden traces, the
existing blend-ramp + ttfs-cycle-step tests) must stay byte-identical: the fast flag
is opt-in and default-off.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
    _SegmentSpikeForward,
)


def _make_pipeline(
    tmp_path,
    *,
    schedule="cascaded",
    blend=True,
    fast=True,
    steps_per_rate=4,
    rates=None,
):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    cfg["ttfs_genuine_blend_ramp"] = blend
    cfg["ttfs_genuine_blend_fast"] = fast
    cfg["ttfs_blend_fast_steps_per_rate"] = steps_per_rate
    if rates is not None:
        cfg["ttfs_blend_fast_rates"] = rates
    # Tiny calibration loops keep the unit test fast and deterministic.
    cfg["ttfs_distmatch_bias_iters"] = 3
    cfg["ttfs_distmatch_bias_eta"] = 0.7
    cfg["ttfs_distmatch_quantile"] = 0.99
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return pipeline


def _make_tuner(tmp_path, **kw):
    pipeline = _make_pipeline(tmp_path, **kw)
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model, am


# ── Flag ON: the fast fixed-increment path runs ──────────────────────────────


class TestFastPathFlag:
    def test_fast_flag_recognized(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True)
        assert tuner._genuine_blend_fast is True

    def test_fast_flag_requires_blend_ramp(self, tmp_path):
        # fast path is inert without the genuine blend ramp.
        tuner, _, _ = _make_tuner(tmp_path, blend=False, fast=True)
        assert tuner._genuine_blend_ramp is False
        assert tuner._genuine_blend_fast is False

    def test_fast_rates_default_schedule(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True, rates=None)
        assert tuner._blend_fast_rates == [0.5, 0.75, 0.9, 0.97, 1.0]

    def test_fast_rates_accepts_json_list(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True, rates=[0.6, 1.0])
        assert tuner._blend_fast_rates == [0.6, 1.0]


class TestFastRunTakesFastPath:
    def test_run_sets_fast_marker_and_skips_scheduler(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=3)
        tuner.run()
        assert getattr(tuner, "_fast_blend_path", False) is True, (
            "the fast path must mark itself so the scheduler/_adaptation flow is "
            "provably skipped"
        )
        # The SmoothAdaptation scheduler never ran: no adaptation cycles logged.
        assert len(tuner._cycle_log) == 0

    def test_run_runs_exactly_rates_times_steps_optimizer_steps(self, tmp_path):
        torch.manual_seed(0)
        rates = [0.5, 1.0]
        steps = 5
        tuner, _, _ = _make_tuner(
            tmp_path, fast=True, steps_per_rate=steps, rates=rates,
        )
        tuner.run()
        assert tuner._fast_optimizer_steps == len(rates) * steps

    def test_run_commits_rate_one(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        tuner.run()
        assert tuner._committed_rate == pytest.approx(1.0)

    def test_run_deploys_pure_genuine_cascade(self, tmp_path):
        torch.manual_seed(0)
        tuner, model, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        tuner.run()
        installed = model.__dict__.get("forward")
        assert isinstance(installed, _SegmentSpikeForward), (
            "the fast path must run the existing finalize so the deployed forward "
            "is the pure genuine single-spike cascade"
        )
        # Bit-exact to a freshly built cascade on the same weights.
        T = int(tuner.pipeline.config["simulation_steps"])
        x = torch.randn(3, *tuner.pipeline.config["input_shape"])
        fresh = _SegmentSpikeForward(model, T)
        with torch.no_grad():
            torch.testing.assert_close(model(x), fresh(x), rtol=0, atol=0)

    def test_run_returns_validate_metric(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        result = tuner.run()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # validate() returns the cached final metric.
        validated = tuner.validate()
        assert isinstance(validated, float)
        assert 0.0 <= validated <= 1.0

    def test_run_records_phase_seconds(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        tuner.run()
        assert "fast_blend" in tuner._phase_seconds


# ── Flag OFF (default): the SmoothAdaptation path is unchanged ────────────────


class TestFastFlagOffUnchanged:
    def test_fast_disabled_by_default(self, tmp_path):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 16
        cfg["ttfs_genuine_blend_ramp"] = True
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=AdaptationManager(),
        )
        assert tuner._genuine_blend_fast is False

    def test_blend_ramp_without_fast_uses_scheduler(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=False)
        assert tuner._genuine_blend_fast is False
        tuner.run()
        # The SmoothAdaptation scheduler ran: cycles were logged and no fast marker.
        assert getattr(tuner, "_fast_blend_path", False) is False
        assert len(tuner._cycle_log) > 0

    def test_fast_flag_inert_without_blend_ramp(self, tmp_path):
        # blend OFF + fast ON: the value-domain proxy ramp (scheduler) still runs.
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, blend=False, fast=True)
        tuner.run()
        assert getattr(tuner, "_fast_blend_path", False) is False
        assert len(tuner._cycle_log) > 0


# ── Synchronized: the genuine blend (and so the fast path) is ignored ─────────


class TestSynchronizedIgnoresFast:
    def test_synchronized_ignores_fast_flag(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, schedule="synchronized", fast=True)
        assert tuner._genuine_blend_ramp is False
        assert tuner._genuine_blend_fast is False
