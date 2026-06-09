"""The unified finalize contract.

- Every rate tuner commits rate=1.0 and then enforces the pipeline floor via
  ``_ensure_pipeline_threshold`` exactly once — the floor check lives in the
  ``AdaptationRateTuner`` base, not re-implemented per subclass.
- ``KDBlendAdaptationTuner._finalize`` is the single finalize template; LIF
  customizes it through ordered hooks (set ``lif_active`` before the activation
  rebuild, apply cycle-accurate trains after) rather than copying the body.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from conftest import make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager


class TestAdaptationRateTunerEnforcesFloor:
    def test_after_run_commits_then_checks_floor_and_sets_metric(self):
        from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner

        t = AdaptationRateTuner.__new__(AdaptationRateTuner)
        t._committed_rate = 0.5
        calls = []
        t._continue_to_full_rate = lambda: calls.append("continue")
        t._apply_rate = lambda r: calls.append(("apply", r))
        t._ensure_pipeline_threshold = lambda: calls.append("floor") or 0.93

        result = t._after_run()

        assert calls == ["continue", ("apply", 1.0), "floor"]
        assert t._committed_rate == 1.0
        assert t._final_metric == 0.93 and result == 0.93

    def test_subclasses_inherit_the_floor_check(self):
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )
        from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner

        assert "_after_run" not in ActivationQuantizationTuner.__dict__
        assert "_after_run" not in NoiseTuner.__dict__


class TestLIFFinalizeViaHooks:
    def test_lif_does_not_copy_the_finalize_body(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

        assert "_finalize" not in LIFAdaptationTuner.__dict__, (
            "LIF must customize finalize through hooks, not copy the base body"
        )

    def test_lif_active_set_before_rebuild(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

        t = LIFAdaptationTuner.__new__(LIFAdaptationTuner)
        t.adaptation_manager = AdaptationManager()
        t.model = make_tiny_supermodel()
        t._cycle_accurate = False
        seen = {}

        def _rebuild():
            seen["lif_active_at_rebuild"] = t.adaptation_manager.lif_active

        t._update_target_activations = _rebuild
        t._finalize_forward = lambda: None

        t._finalize()

        assert seen["lif_active_at_rebuild"] is True, (
            "lif_active must be set BEFORE the activation rebuild so the rebuilt "
            "activations subsume the clamp/quant/shift decorators"
        )
        assert t.adaptation_manager.lif_active is True
        assert t._stabilization_refinds_lr is False  # _finalize_forward returned None

    def test_cycle_accurate_trains_applied_after_rebuild(self, monkeypatch):
        from mimarsinan.tuning.tuners import lif_adaptation_tuner as lif_mod
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner
        import mimarsinan.spiking.lif_utils as lif_utils

        applied = []
        monkeypatch.setattr(
            lif_utils, "apply_cycle_accurate_trains_to_model",
            lambda model, flag: applied.append(flag),
        )

        t = LIFAdaptationTuner.__new__(LIFAdaptationTuner)
        t.adaptation_manager = AdaptationManager()
        t.model = make_tiny_supermodel()
        t._cycle_accurate = True
        t._update_target_activations = lambda: None
        t._finalize_forward = lambda: None

        t._finalize()

        assert applied == [True], "cycle-accurate trains must be applied at finalize"
