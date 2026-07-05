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
