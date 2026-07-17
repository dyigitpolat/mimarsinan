"""E2 — the fast fixed-ladder driver lifted to the shared ``SmoothAdaptationTuner``.

Fix A unbinds the optimization driver from a KD-blend island into a pipeline-wide
``controller | fast`` axis consumed by EVERY rate tuner. The schedule-not-search
machinery now lives in ``FastLadderMixin`` (mixed into ``SmoothAdaptationTuner``),
so the analytical clamp/shift/activation-quant chain and the manager-rate family —
which had NO fast path before — inherit it. This file locks:

1. STRUCTURE — the fast machinery is inherited by the smooth base, KD-blend no
   longer redefines it, and the mixin precedes the run mixin in the MRO (so its
   ``super().run()`` / ``super()._driver_attempt`` reach the controller path).
2. DEFAULT-OFF ⇒ BYTE-IDENTICAL — a tuner that never calls ``_setup_fast_ladder``
   has ``_fixed_ladder_policy`` False and runs the unchanged controller loop.
3. THE LIFT WORKS FOR THE ANALYTICAL CHAIN — opting an analytical/manager tuner
   into the fast ladder drives its rate via the uniform setter (``_set_rate`` /
   ``_apply_rate``), commits each rung, records the trace, and reaches rate 1.0.
"""

from __future__ import annotations

import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.config_schema.defaults import DEFAULT_TUNING_RECIPE
from mimarsinan.tuning.orchestration.fast_ladder import FastLadderMixin
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
    SmoothAdaptationRunMixin,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
    SmoothAdaptationTuner,
)


# ── per-family construction (the analytical / manager families) ───────────────

def _clamp_tuner(tmp_path, **cfg_over):
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


def _activation_quantization_tuner(tmp_path, **cfg_over):
    from mimarsinan.tuning.tuners.activation_quantization_tuner import (
        ActivationQuantizationTuner,
    )

    cfg = default_config()
    cfg.update(cfg_over)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return ActivationQuantizationTuner(pipeline, model, 4, 0.5, 0.001, manager)


# ── 1. structure ──────────────────────────────────────────────────────────────

class TestStructure:
    def test_smooth_base_inherits_the_fast_mixin(self):
        assert issubclass(SmoothAdaptationTuner, FastLadderMixin)

    def test_fast_mixin_precedes_run_mixin_in_mro(self):
        mro = SmoothAdaptationTuner.__mro__
        assert mro.index(FastLadderMixin) < mro.index(SmoothAdaptationRunMixin)

    def test_fast_mixin_typing_host_base_is_runtime_inert(self):
        # The TYPE_CHECKING-only host contract must never become a runtime base:
        # the mixin stays object-based so the composed tuner's MRO is unchanged.
        assert FastLadderMixin.__bases__ == (object,)

    def test_smooth_tuner_mro_is_the_locked_composition_order(self):
        from mimarsinan.tuning.orchestration.rate_tuner_seam import RateTunerSeamMixin
        from mimarsinan.tuning.orchestration.smooth_adaptation_cycle import (
            SmoothAdaptationCycleMixin,
        )
        from mimarsinan.tuning.orchestration.tuner_base import TunerBase

        assert SmoothAdaptationTuner.__mro__ == (
            SmoothAdaptationTuner,
            RateTunerSeamMixin,
            FastLadderMixin,
            SmoothAdaptationCycleMixin,
            SmoothAdaptationRunMixin,
            TunerBase,
            object,
        )

    def test_kd_blend_does_not_redefine_the_lifted_methods(self):
        # The fast machinery moved up; KD-blend must inherit it, not shadow it.
        for name in (
            "_setup_fast_ladder",
            "_fast_rate_attempt",
            "_ensure_fast_optimizer",
            "_record_fast_cycle",
            "_build_fast_lr_schedule",
        ):
            assert name not in KDBlendAdaptationTuner.__dict__, name
            assert name in FastLadderMixin.__dict__, name

    def test_analytical_tuner_exposes_the_fast_api(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            for name in ("_setup_fast_ladder", "_fast_rate_attempt",
                         "_fast_set_rate"):
                assert callable(getattr(t, name)), name
        finally:
            t.close()


# ── 2. default-off ⇒ byte-identical (the controller path is unchanged) ─────────

class TestDefaultOffIsController:
    def test_no_setup_means_no_fixed_ladder_policy(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            assert getattr(t, "_fixed_ladder_policy", False) is False
        finally:
            t.close()

    def test_stabilization_budget_is_the_controller_value_when_off(self, tmp_path):
        t = _clamp_tuner(tmp_path)
        try:
            # ClampTuner inherits the base 2*max_training_steps; the fast override
            # must NOT short-circuit it to 0 unless the policy is enabled.
            assert t._stabilization_budget() == 2 * int(t._budget.max_training_steps)
        finally:
            t.close()

    def test_controller_run_unchanged(self, tmp_path):
        torch.manual_seed(0)
        t = _clamp_tuner(tmp_path)
        try:
            t.run()
            assert getattr(t, "_fast_blend_path", False) is False
            assert t._committed_rate == pytest.approx(1.0)
            assert len(t._cycle_log) > 0
        finally:
            t.close()


# ── 3. the lift works: an analytical tuner can be driven by the fast ladder ────

class TestAnalyticalChainGainsAFastPath:
    def _fast(self, tuner, *, rates, steps_per_rate):
        tuner._setup_fast_ladder(
            enabled=True, rates=rates, steps_per_rate=steps_per_rate,
        )
        return tuner

    def test_clamp_fast_ladder_commits_each_rung(self, tmp_path):
        torch.manual_seed(0)
        t = self._fast(_clamp_tuner(tmp_path), rates=[0.5, 1.0], steps_per_rate=2)
        try:
            t.run()
            assert t._fast_blend_path is True
            assert t._committed_rate == pytest.approx(1.0)
            assert len(t._cycle_log) == len(t._fixed_ladder_rates)
            assert [e["outcome"] for e in t._cycle_log] == \
                ["commit"] * len(t._fixed_ladder_rates)
            assert t._fast_optimizer_steps == len(t._fixed_ladder_rates) * 2
        finally:
            t.close()

    def test_fast_ladder_disables_stabilization(self, tmp_path):
        t = self._fast(_clamp_tuner(tmp_path), rates=[0.5, 1.0], steps_per_rate=2)
        try:
            assert t._stabilization_budget() == 0
        finally:
            t.close()

    def test_manager_rate_family_drives_via_apply_rate(self, tmp_path):
        # The manager-rate family defines _apply_rate (not _set_rate); the uniform
        # _fast_set_rate must resolve it so the lifted ladder drives the rate.
        torch.manual_seed(0)
        t = self._fast(
            _activation_quantization_tuner(tmp_path), rates=[0.5, 1.0],
            steps_per_rate=2,
        )
        try:
            seen = []
            orig = t._apply_rate
            t._apply_rate = lambda r: (seen.append(float(r)), orig(r))[1]
            t.run()
            assert t._committed_rate == pytest.approx(1.0)
            # the fast attempt set each scheduled rate through _apply_rate
            assert 1.0 in seen
        finally:
            t.close()

    def test_rerun_resets_fast_scratch(self, tmp_path):
        torch.manual_seed(0)
        t = self._fast(_clamp_tuner(tmp_path), rates=[0.5, 1.0], steps_per_rate=2)
        try:
            t.run()
            first = t._fast_optimizer
            t.run()
            assert t._fast_optimizer is not first
            assert t._fast_optimizer_steps == len(t._fixed_ladder_rates) * 2
        finally:
            t.close()


# ── 4. WS-A: LR backoff scaling + recipe grad-clip threading ─────────────────

class TestScaleFastLr:
    """[WS-A A1] the retention gate's LR backoff: scale every optimizer group
    AND the spanning schedule's children together so the halving survives
    schedule.step() (naive param_group mutation is overwritten each step)."""

    def _fast_tuner(self, tmp_path):
        tuner = _clamp_tuner(
            tmp_path, optimization_driver="fast",
            clamp_fast_rates=[0.5, 1.0], clamp_fast_steps_per_rate=2,
        )
        tuner._ensure_fast_optimizer()
        return tuner

    def test_scales_groups_children_and_eta_min_together(self, tmp_path):
        tuner = self._fast_tuner(tmp_path)
        try:
            optimizer = tuner._fast_optimizer
            schedule = tuner._fast_lr_schedule
            group_before = [g["lr"] for g in optimizer.param_groups]
            init_before = [g["initial_lr"] for g in optimizer.param_groups]
            base_before = [list(s.base_lrs) for s in schedule._schedulers]
            eta_before = getattr(schedule._schedulers[-1], "eta_min", 0.0)
            tuner._scale_fast_lr(0.5)
            assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(
                [0.5 * v for v in group_before])
            assert [g["initial_lr"] for g in optimizer.param_groups] == (
                pytest.approx([0.5 * v for v in init_before]))
            for scheduler, before in zip(schedule._schedulers, base_before):
                assert list(scheduler.base_lrs) == pytest.approx(
                    [0.5 * v for v in before])
            assert getattr(schedule._schedulers[-1], "eta_min", 0.0) == (
                pytest.approx(0.5 * eta_before))
        finally:
            tuner.close()

    def test_scaled_schedule_still_steps(self, tmp_path):
        tuner = self._fast_tuner(tmp_path)
        try:
            tuner._scale_fast_lr(0.5)
            for _ in range(3):
                tuner._fast_optimizer.step()
                tuner._fast_lr_schedule.step()
            lr = tuner._fast_optimizer.param_groups[0]["lr"]
            assert 0.0 <= lr <= 0.001
        finally:
            tuner.close()


class TestRungGradClip:
    """[WS-A A2] the rung loop must honor the tuning recipe's declared
    grad_clip_norm (it silently ignored it: recovery paths clip, fast rungs
    did not)."""

    def _spy_clip(self, monkeypatch):
        calls = []
        original = torch.nn.utils.clip_grad_norm_

        def spy(params, max_norm, **kwargs):
            calls.append(float(max_norm))
            return original(params, max_norm, **kwargs)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
        return calls

    def test_recipe_clip_reaches_the_rung_loop(self, tmp_path, monkeypatch):
        calls = self._spy_clip(monkeypatch)
        tuner = _clamp_tuner(
            tmp_path, optimization_driver="fast",
            spiking_mode="ttfs_quantized", activation_quantization=True,
            tuning_recipe=dict(DEFAULT_TUNING_RECIPE),
            clamp_fast_rates=[0.5], clamp_fast_steps_per_rate=2,
        )
        try:
            tuner._ensure_fast_optimizer()
            tuner._fast_train_rung(0.5)
            assert calls == [1.0, 1.0]  # DEFAULT_TUNING_RECIPE.grad_clip_norm
        finally:
            tuner.close()

    def test_null_recipe_clip_stays_unclipped(self, tmp_path, monkeypatch):
        calls = self._spy_clip(monkeypatch)
        cfg_recipe = dict(DEFAULT_TUNING_RECIPE)
        cfg_recipe["grad_clip_norm"] = None
        tuner = _clamp_tuner(
            tmp_path, optimization_driver="fast", tuning_recipe=cfg_recipe,
            spiking_mode="ttfs_quantized", activation_quantization=True,
            clamp_fast_rates=[0.5], clamp_fast_steps_per_rate=2,
        )
        try:
            tuner._ensure_fast_optimizer()
            tuner._fast_train_rung(0.5)
            assert calls == []
        finally:
            tuner.close()

    def test_explicit_caller_clip_wins(self, tmp_path, monkeypatch):
        calls = self._spy_clip(monkeypatch)
        tuner = _clamp_tuner(
            tmp_path, optimization_driver="fast",
            spiking_mode="ttfs_quantized", activation_quantization=True,
            clamp_fast_rates=[0.5], clamp_fast_steps_per_rate=2,
        )
        try:
            tuner._ensure_fast_optimizer()
            tuner._fast_train_rung(0.5, grad_clip_norm=0.25)
            assert calls == [0.25, 0.25]
        finally:
            tuner.close()


class TestRetryStepScale:
    """[WS-A A1 completion] a retention retry halves the LR AND doubles the
    step budget (LR x steps preserved: finer steps, same path length —
    measured: fixed-step halvings converged 0.09->0.39->0.67->0.77 and
    exhausted short of the 0.856 retention bound). Scaled retries hold the
    spanning schedule so the cosine never overruns its T_max."""

    def _tuner(self, tmp_path):
        return _clamp_tuner(
            tmp_path, optimization_driver="fast",
            spiking_mode="ttfs_quantized", activation_quantization=True,
            clamp_fast_rates=[0.5], clamp_fast_steps_per_rate=2,
        )

    def test_scaled_rung_runs_scaled_steps_and_holds_the_schedule(self, tmp_path):
        tuner = self._tuner(tmp_path)
        try:
            tuner._ensure_fast_optimizer()
            tuner._fast_retry_step_scale = 2
            lr_before = tuner._fast_optimizer.param_groups[0]["lr"]
            steps_before = tuner._fast_optimizer_steps
            tuner._fast_train_rung(0.5)
            assert tuner._fast_optimizer_steps == steps_before + 4  # 2 steps x2
            assert tuner._fast_optimizer.param_groups[0]["lr"] == (
                pytest.approx(lr_before))  # schedule held: LR constant
        finally:
            tuner.close()

    def test_unit_scale_keeps_the_spanning_schedule(self, tmp_path):
        tuner = self._tuner(tmp_path)
        try:
            tuner._ensure_fast_optimizer()
            lr_before = tuner._fast_optimizer.param_groups[0]["lr"]
            tuner._fast_train_rung(0.5)
            assert tuner._fast_optimizer_steps == 2
            assert tuner._fast_optimizer.param_groups[0]["lr"] != lr_before
        finally:
            tuner.close()


# ── fast_lr_scale: the retry-economics lever reaches the ONE build site ───────

class TestFastLrScale:
    def test_default_is_byte_identical(self, tmp_path):
        tuner = _clamp_tuner(
            tmp_path, optimization_driver="fast",
            clamp_fast_rates=[0.5, 1.0], clamp_fast_steps_per_rate=2,
        )
        try:
            tuner._ensure_fast_optimizer()
            # initial_lr is the peak; the warmup schedule has already
            # rescaled the live lr at build.
            lrs = sorted({g["initial_lr"] for g in tuner._fast_optimizer.param_groups})
            assert lrs == pytest.approx([float(tuner.pipeline_lr)])
        finally:
            tuner.close()

    def test_scale_multiplies_the_fast_start_lr_only(self, tmp_path):
        tuner = _clamp_tuner(
            tmp_path, optimization_driver="fast", fast_lr_scale=0.25,
            clamp_fast_rates=[0.5, 1.0], clamp_fast_steps_per_rate=2,
        )
        try:
            tuner._ensure_fast_optimizer()
            lrs = sorted({g["initial_lr"] for g in tuner._fast_optimizer.param_groups})
            assert lrs == pytest.approx([0.25 * float(tuner.pipeline_lr)])
            # The pipeline lr itself (anchors, recovery, preload) is untouched.
            assert float(tuner.pipeline_lr) == pytest.approx(0.001)
        finally:
            tuner.close()
