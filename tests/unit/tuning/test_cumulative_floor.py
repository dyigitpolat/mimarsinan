"""Tests for the absolute baseline-anchored floor inside ``_adaptation``.

Without this floor, every cycle only has to stay within the noise margin of
its own ``pre_cycle_acc``. Across N cycles, the accepted cumulative drop can
compound up to ``N * rollback_tolerance`` even though no individual cycle
ever looks like a regression. This test suite pins down the invariant that
the tuner must never commit a ``post_acc`` below
``absolute_floor = max(pipeline_hard_floor, validation_baseline - tolerance)``
once the baseline has been captured.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class _ScriptedTuner(SmoothAdaptationTuner):
    """Concrete tuner whose ``_update_and_evaluate`` and trainer.validate
    return scripted sequences, so that the acceptance gates can be tested
    in isolation from actual training."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.update_calls = []
        self._instant_acc = 0.85
        self._validate_queue = []
        self._validate_idx = 0

    def _update_and_evaluate(self, rate):
        self.update_calls.append(rate)
        return self._instant_acc

    def _find_lr(self):
        return 0.001

    def set_validate_sequence(self, seq):
        self._validate_queue = list(seq)
        self._validate_idx = 0

    def _install(self):
        tuner = self

        def _mock_validate_n_batches(n):
            idx = tuner._validate_idx
            tuner._validate_idx += 1
            if idx < len(tuner._validate_queue):
                return tuner._validate_queue[idx]
            return tuner._validate_queue[-1] if tuner._validate_queue else 0.5

        self.trainer.validate_n_batches = _mock_validate_n_batches
        self.trainer.train_steps_until_target = lambda *a, **kw: None


@pytest.fixture
def scripted(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    tuner = _ScriptedTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
    tuner._rollback_tolerance = 0.02  # realistic noise margin
    tuner._validation_baseline = 0.90  # baseline captured before any cycle
    tuner._pipeline_hard_floor = None
    return tuner


class TestAbsoluteFloorInAdaptation:
    """The adaptation cycle must reject ``post_acc`` that falls below an
    absolute floor anchored to ``_validation_baseline``, independent of
    ``pre_cycle_acc``. This is the primary safeguard against cumulative
    drift across many cycles."""

    def test_post_acc_below_absolute_floor_rolls_back(self, scripted):
        """Even if ``post_acc`` is within the noise margin of a *degraded*
        ``pre_cycle_acc``, dropping below ``baseline - tolerance`` must
        trigger rollback."""
        tuner = scripted
        # baseline is 0.90; degradation_tolerance 0.05 -> absolute floor 0.855.
        # pre_cycle_acc of 0.86 is (barely) above the floor; post_acc of 0.84
        # is within rollback_tolerance=0.02 of pre (0.86-0.84=0.02), so the
        # pre-relative gate would admit it. The absolute floor must reject it.
        tuner.set_validate_sequence([0.86, 0.84])
        tuner._install()
        tuner._committed_rate = 0.0

        result = tuner._adaptation(0.5)

        assert result == 0.0, (
            "Cycle must roll back when post_acc falls below the "
            f"baseline-anchored absolute floor (post={0.84}, "
            f"baseline={tuner._validation_baseline}, tol=0.05)."
        )

    def test_cumulative_drift_is_bounded(self, scripted):
        """Across many cycles, every commit must keep post_acc >= absolute
        floor. The pre-relative gate alone permits arbitrary drift; the
        absolute floor must cap it."""
        tuner = scripted
        tuner._validation_baseline = 0.90
        # absolute floor = 0.90 - 0.05 = 0.85.
        tuner._install()
        tuner._committed_rate = 0.0

        commit_count = 0
        # pre=0.88, post=0.86 -> pre-relative gate admits (diff=0.02=tol);
        # absolute gate admits (post 0.86 > 0.85). Commit.
        tuner.set_validate_sequence([0.88, 0.86])
        r = tuner._adaptation(0.2)
        if r == 0.2:
            commit_count += 1

        # pre=0.86, post=0.84 -> pre-relative gate admits (diff=0.02);
        # absolute gate REJECTS (post 0.84 < floor 0.85). Must roll back.
        tuner._validate_idx = 0
        tuner.set_validate_sequence([0.86, 0.84])
        r = tuner._adaptation(0.4)
        assert r < 0.4, (
            "Second cycle must roll back because absolute floor was breached"
        )

    def test_unseeded_baseline_falls_back_to_relative_gate(self, scripted):
        """When ``_validation_baseline`` has not been set (first cycles before
        baseline capture), the rollback must fall back to the pre-relative
        gate to preserve existing tuner behaviour for unit-test fixtures
        that construct the tuner without calling ``run()``."""
        tuner = scripted
        tuner._validation_baseline = None
        tuner._pipeline_hard_floor = None

        # pre=0.88, post=0.86, rollback_tolerance=0.02 -> diff=0.02 == tol -> admit.
        tuner.set_validate_sequence([0.88, 0.86])
        tuner._install()
        tuner._committed_rate = 0.0

        result = tuner._adaptation(0.5)
        assert result == 0.5, (
            "Without a baseline seed, the cycle must behave exactly as "
            "before this fix (pre-relative gate only)."
        )

    def test_pipeline_hard_floor_participates_in_absolute_floor(self, scripted):
        """When a pipeline_hard_floor is set higher than baseline-tolerance,
        the absolute floor uses the stricter of the two."""
        tuner = scripted
        tuner._validation_baseline = 0.90
        tuner._pipeline_hard_floor = 0.88  # stricter than baseline-0.05=0.85

        # pre=0.89, post=0.87. diff=0.02=tol -> pre-relative admits.
        # baseline-anchored floor = 0.85; 0.87>0.85 -> would admit.
        # pipeline_hard_floor = 0.88; 0.87<0.88 -> must REJECT.
        tuner.set_validate_sequence([0.89, 0.87])
        tuner._install()
        tuner._committed_rate = 0.0

        result = tuner._adaptation(0.5)
        assert result == 0.0, (
            "Pipeline_hard_floor must participate in the absolute floor "
            "and reject post_acc below it."
        )


class TestTargetDecayBoundedByBaselineFloor:
    """Target relaxation via ``_missed_target_streak`` must never take the
    target below the baseline-anchored absolute floor. Currently the floor
    is derived from ``original_metric * (1 - degradation_tolerance)``; the
    fix must also clamp to ``_validation_baseline * (1 - tolerance)``."""

    def test_target_never_decays_below_baseline_floor(self, tmp_path):
        """After many missed streaks, target stays >= absolute floor."""
        cfg = default_config()
        cfg["degradation_tolerance"] = 0.05
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _ScriptedTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        tuner._rollback_tolerance = 0.02
        tuner._validation_baseline = 0.90

        absolute_floor = 0.90 * (1.0 - 0.05)  # 0.855

        # Drive many missed-target cycles.
        tuner._instant_acc = 0.86
        tuner.set_validate_sequence([0.86, 0.86])

        tuner._install()

        for _ in range(30):
            # Cycle the scripted sequence so each _adaptation sees fresh data.
            tuner._validate_idx = 0
            tuner._adaptation(0.5)
            tuner._committed_rate = 0.0  # keep the loop alive for rollbacks

        assert tuner.target_adjuster.target_metric >= absolute_floor - 1e-6, (
            "target_metric decayed below baseline-anchored absolute floor: "
            f"target={tuner.target_adjuster.target_metric}, "
            f"absolute_floor={absolute_floor}"
        )
