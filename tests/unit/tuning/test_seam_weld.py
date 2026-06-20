"""EF2 — the run loop drives uniformly through the E1 seam verbs.

EF1 made every rate-tuner family READ the ``optimization_driver`` axis. EF2 is the
deferred E1↔E2 weld: the ``OptimizationDriver`` now drives the tuner LITERALLY
through the three seam verbs (``ramp`` / ``recover_to`` / ``probe``), so "the
driver drives an ``AdaptationAxis``-shaped tuner" is REAL — the controller cycle
and the fast-ladder rung both route their predictor / corrector / read through the
seam, not through ad-hoc inlined private calls.

BYTE-IDENTICAL: each verb DELEGATES to the existing primitive, so routing through
it changes no number. This file locks:

1. WELD — the controller cycle phases call ``self.ramp`` (predictor), the
   ``recover_to`` SSOT (corrector), and ``self.probe`` (reads), and the fast-ladder
   rung reads its post-accuracy through ``self.probe``.
2. BYTE-IDENTITY — the welded run reproduces the legacy trace exactly (the golden
   traces + the driver-equivalence trajectories cover this; here we assert the
   per-phase delegation is observed, so a future regression that bypasses a verb
   trips a unit test, not only a golden diff).
"""

from __future__ import annotations

import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_scripted_run_tuner,
    make_tiny_supermodel,
)
from mimarsinan.tuning.orchestration.adaptation_driver import CycleContext


# ── construction helpers (mirror the seam / fast-ladder tests) ────────────────

def _clamp_tuner(tmp_path, **cfg_over):
    from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
        create_adaptation_manager_for_model,
    )
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    cfg = default_config()
    cfg.update(cfg_over)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    stats = make_activation_scale_stats(model, scales)
    return ClampTuner(pipeline, model, 0.5, 0.001, manager, scales, stats)


def _scripted(tmp_path, instant_fn, post_fn, ladder=False):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    return make_scripted_run_tuner(
        pipeline, make_tiny_supermodel(),
        instant_fn=instant_fn, post_fn=post_fn, ladder=ladder,
    )


# ── 1. the controller cycle phases route through the seam verbs ───────────────

class TestControllerCyclePhasesRouteThroughSeam:
    """Each per-cycle phase invokes the named seam verb, not an inlined private."""

    def test_probe_instant_calls_ramp(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            seen = {}

            def _ramp(rate):
                seen["rate"] = rate
                return 0.812

            t.ramp = _ramp
            t._get_target = lambda: 0.9
            ctx = CycleContext(rate=0.375, t_cycle_start=0.0, pre_state=None,
                               pre_cycle_acc=0.0)
            t._probe_instant(ctx)
            assert seen["rate"] == pytest.approx(0.375)
            assert ctx.instant_acc == pytest.approx(0.812)
        finally:
            t.close()

    def test_recover_calls_recover_to_seam(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            seen = {}
            t._get_target = lambda: 0.9

            def _seam(target, rate=None):
                seen["target"] = target
                seen["rate"] = rate
                t._last_recover_lr = 0.0123
                return None

            t.recover_to = _seam
            ctx = CycleContext(rate=0.5, t_cycle_start=0.0, pre_state=None,
                               pre_cycle_acc=0.0)
            t._recover(ctx)
            assert seen["target"] == pytest.approx(0.9)
            assert seen["rate"] == pytest.approx(0.5)
            # the lr the seam used is threaded back onto the cycle context
            assert ctx.lr == pytest.approx(0.0123)
        finally:
            t.close()

    def test_begin_cycle_pre_acc_reads_through_probe(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            t._validation_baseline = 0.9
            t._last_post_acc = None  # force a fresh pre-cycle read
            calls = []
            t.probe = lambda: (calls.append(1), 0.811)[1]
            ctx = t._begin_cycle(0.5)
            assert calls == [1]
            assert ctx.pre_cycle_acc == pytest.approx(0.811)
        finally:
            t.close()

    def test_measure_post_reads_through_probe(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            t._validation_baseline = 0.9
            t._rollback_tolerance = 0.05
            calls = []
            t.probe = lambda: (calls.append(1), 0.88)[1]
            ctx = CycleContext(rate=0.5, t_cycle_start=0.0, pre_state=None,
                               pre_cycle_acc=0.9)
            t._measure_post(ctx)
            assert calls == [1]
            assert ctx.post_acc == pytest.approx(0.88)
        finally:
            t.close()


# ── 2. the recover_to seam exposes the lr it used (the cycle reads it) ─────────

class TestRecoverToExposesLr:
    def test_recover_to_stashes_last_recover_lr(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            t._recover_to_target = lambda target, rate: (0.07, "result")
            t._committed_rate = 0.5
            out = t.recover_to(0.9)
            assert out == "result"  # the public verb still returns the RESULT
            # ... and stashes the lr so the cycle path can record it
            assert t._last_recover_lr == pytest.approx(0.07)
        finally:
            t.close()


# ── 3. the fast-ladder rung routes its read through the probe verb ────────────

class TestFastLadderRoutesThroughSeam:
    def test_fast_rate_attempt_reads_post_acc_through_probe(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            t._setup_fast_ladder(enabled=True, rates=[1.0], steps_per_rate=1)
            t._cycle_log = type(t._cycle_log).new() if hasattr(t, "_cycle_log") \
                else t._cycle_log
            calls = []
            real_probe = t.probe
            t.probe = lambda: (calls.append(1), real_probe())[1]
            t.run()
            # the rung's post-accuracy read went through the seam probe verb
            assert calls, "fast rung did not route its read through probe()"
        finally:
            t.close()

    def test_fast_ramp_seam_applies_rate_and_trains(self, tmp_path):
        """The fast-ladder predictor verb (``_fast_ramp``) sets the rate via the
        uniform setter and trains the rung; the read is the universal probe."""
        torch.manual_seed(0)
        t = _clamp_tuner(tmp_path)
        try:
            t._setup_fast_ladder(enabled=True, rates=[0.5, 1.0], steps_per_rate=2)
            seen = []
            orig = t._fast_set_rate
            t._fast_set_rate = lambda r: (seen.append(float(r)), orig(r))[1]
            t.run()
            assert t._committed_rate == pytest.approx(1.0)
            # every scheduled rung's rate flowed through the uniform fast setter
            assert 1.0 in seen
        finally:
            t.close()


# ── 4. byte-identity: the welded run reproduces the legacy trajectory ─────────

class TestWeldIsByteIdentical:
    def _committed_trace(self, tuner):
        return [
            (r.outcome, round(float(r.rate), 6), round(float(r.committed), 6))
            for r in tuner._cycle_log.records
        ]

    def test_one_shot_run_trajectory_unchanged(self, tmp_path, deterministic_rng):
        t = _scripted(tmp_path, lambda r: 0.87, lambda r: 0.9)
        t.run()
        assert t._committed_rate == pytest.approx(1.0)
        assert [o for o, _, _ in self._committed_trace(t)] == ["commit"]

    def test_ramp_after_failed_one_shot_trajectory_monotone(
        self, tmp_path, deterministic_rng,
    ):
        instant = lambda r: 0.1 if r >= 0.99 else 0.85
        t = _scripted(tmp_path, instant, lambda r: 0.9)
        t.run()
        assert t._committed_rate == pytest.approx(1.0)
        committed = [c for _, _, c in self._committed_trace(t)]
        assert committed == sorted(committed)
        outcomes = [o for o, _, _ in self._committed_trace(t)]
        assert outcomes[0] == "catastrophic"
        assert "commit" in outcomes

    def test_kdblend_ladder_trajectory_unchanged(self, tmp_path, deterministic_rng):
        t = _scripted(tmp_path, lambda r: 0.87, lambda r: 0.9, ladder=True)
        t.run()
        assert t._committed_rate == pytest.approx(1.0)
        commits = [r for o, r, _ in self._committed_trace(t) if o == "commit"]
        assert commits == sorted(commits)
        assert commits[-1] == pytest.approx(1.0)

