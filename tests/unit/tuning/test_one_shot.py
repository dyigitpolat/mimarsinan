"""Tests for the one-shot-first optimization in SmoothAdaptationTuner.run().

The one-shot tries _adaptation(1.0) before starting SmartSmoothAdaptation.
For light transformations, this completes in a single cycle. For heavy
transformations, the fast-fail mechanism restores state and falls back
to gradual adaptation.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner, CATASTROPHIC_DROP_FACTOR


class _OneShotTrackingTuner(SmoothAdaptationTuner):
    """Tuner that tracks calls and simulates light or heavy transformations."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_calls = []
        self.before_cycle_calls = 0
        self._instant_acc_fn = lambda rate: 0.85
        self._post_acc_fn = lambda rate: 0.88
        self._after_run_called = False

    def _update_and_evaluate(self, rate):
        self.adaptation_calls.append(("update", rate))
        return self._instant_acc_fn(rate)

    def _find_lr(self):
        return 0.001

    def _before_cycle(self):
        self.before_cycle_calls += 1

    def _after_run(self):
        self._after_run_called = True
        self._continue_to_full_rate()
        self._committed_rate = 1.0
        return self._post_acc_fn(1.0)

    def _patch_trainer(self):
        tuner = self

        def _mock_validate_n_batches(n):
            return tuner._post_acc_fn(tuner._committed_rate)

        self.trainer.validate_n_batches = _mock_validate_n_batches
        self.trainer.validate = lambda: tuner._post_acc_fn(tuner._committed_rate)
        self.trainer.train_steps_until_target = lambda *a, **kw: None
        self.trainer.test = lambda: tuner._post_acc_fn(1.0)


class TestOneShotSuccess:
    """When transformation is light, one-shot succeeds and skips gradual loop."""

    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        cfg["degradation_tolerance"] = 0.05
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _OneShotTrackingTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        return tuner

    def test_one_shot_commits_directly(self, setup):
        """If rate=1.0 succeeds, committed_rate reaches 1.0 immediately."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.87  # > 0.9*0.8 = 0.72 (catastrophic)
        tuner._post_acc_fn = lambda r: 0.87     # > 0.9*0.95 = 0.855 (rollback)
        tuner._patch_trainer()

        tuner.run()

        assert tuner._committed_rate >= 1.0 - 1e-6
        assert tuner._after_run_called

    def test_one_shot_calls_before_cycle(self, setup):
        """_before_cycle() is called before the one-shot attempt."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.87
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        tuner.run()

        assert tuner.before_cycle_calls >= 1

    def test_one_shot_skips_smooth_adaptation(self, setup):
        """When one-shot succeeds, SmartSmoothAdaptation is never invoked.
        Verify by checking that only rate=1.0 was attempted."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.87
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        tuner.run()

        update_rates = [r for tag, r in tuner.adaptation_calls if tag == "update"]
        assert update_rates == [1.0], (
            f"Expected only rate=1.0 attempt, got {update_rates}"
        )

    def test_one_shot_returns_after_run_result(self, setup):
        """run() returns the result of _after_run()."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.87
        tuner._post_acc_fn = lambda r: 0.92
        tuner._patch_trainer()

        result = tuner.run()

        assert result == 0.92


class TestNaturalRateTracking:
    """Verify that _natural_rate records the highest rate reached before _after_run."""

    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        cfg["degradation_tolerance"] = 0.05
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _OneShotTrackingTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        return tuner

    def test_natural_rate_equals_committed_on_one_shot_success(self, setup):
        """When one-shot succeeds, _natural_rate == 1.0."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.87
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        tuner.run()

        assert tuner._natural_rate >= 1.0 - 1e-6

    def test_natural_rate_below_one_triggers_warning(self, setup):
        """When natural adaptation doesn't reach 1.0, a warning is emitted."""
        tuner = setup

        call_count = [0]
        def instant_acc(rate):
            call_count[0] += 1
            if rate >= 0.99:
                return 0.1
            return 0.85

        tuner._instant_acc_fn = instant_acc
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tuner.run()

            natural_warnings = [
                x for x in w
                if "natural adaptation" in str(x.message).lower()
            ]
            if tuner._natural_rate < 1.0 - 1e-6:
                assert len(natural_warnings) >= 1, (
                    f"Expected a warning about natural rate, got none. "
                    f"natural_rate={tuner._natural_rate:.4f}"
                )


class TestOneShotFailure:
    """When transformation is heavy, one-shot fails and falls back to gradual."""

    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        cfg["degradation_tolerance"] = 0.05
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _OneShotTrackingTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        return tuner

    def test_catastrophic_fail_falls_back(self, setup):
        """If rate=1.0 causes catastrophic drop, model state is restored and
        gradual adaptation proceeds."""
        tuner = setup

        def instant_acc(rate):
            if rate >= 0.99:
                return 0.1  # Catastrophic: < 0.9*0.8 = 0.72
            return 0.85

        tuner._instant_acc_fn = instant_acc
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        tuner.run()

        # Should have attempted rate=1.0 first (one-shot), then gradual rates
        update_rates = [r for tag, r in tuner.adaptation_calls if tag == "update"]
        assert update_rates[0] == 1.0, "First attempt must be one-shot at rate=1.0"
        assert len(update_rates) > 1, "Must fall back to gradual adaptation"

    def test_rollback_fail_falls_back(self, setup):
        """If rate=1.0 passes catastrophic check but fails the rate=1.0
        validation gate, model state is restored and gradual loop runs.

        The rate=1.0 strict gate is now validation-based (Phase A1); the
        former test()-gate leaked test labels into the rollback decision.
        """
        tuner = setup

        tuner._instant_acc_fn = lambda r: 0.85  # passes catastrophic
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        # validate_n_batches is called many times in _adaptation:
        # - Once pre-cycle (baseline),
        # - Once post-cycle (post_acc),
        # - Once for the strict rate=1.0 gate.
        # We sabotage the strict gate (3rd call during the one-shot) to
        # be well below ``validation_baseline - rollback_tolerance``.
        val_call_count = [0]
        def mock_validate(_n):
            val_call_count[0] += 1
            # The run() noise calibration uses the first two calls, then
            # _adaptation uses the pre/post/strict sequence.
            # Make only the strict probes after baseline fail hard.
            if val_call_count[0] <= 2:
                return 0.87  # noise calibration (run())
            if val_call_count[0] == 5:
                return 0.40  # strict rate=1.0 gate -> reject
            return 0.87
        tuner.trainer.validate_n_batches = mock_validate
        # Ensure the tuner never hits trainer.test() -- A1 invariant.
        tuner.trainer.test = lambda: (_ for _ in ()).throw(
            AssertionError("trainer.test() must not be called from tuner code")
        )

        tuner.run()

        update_rates = [r for tag, r in tuner.adaptation_calls if tag == "update"]
        assert update_rates[0] == 1.0, "First attempt must be one-shot"
        assert len(update_rates) > 1, (
            "Must fall back after one-shot validation gate rejection"
        )

    def test_state_restored_after_failed_one_shot(self, setup):
        """Model parameters are identical before and after a failed one-shot."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.1  # catastrophic
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        pre_state = {k: v.clone() for k, v in tuner.model.state_dict().items()}

        # Run just the one-shot portion
        tuner._before_cycle()
        tuner._adaptation(1.0)

        assert tuner._committed_rate == 0.0
        for k, v in tuner.model.state_dict().items():
            assert torch.allclose(v, pre_state[k], atol=1e-6), (
                f"Parameter {k} should be restored after failed one-shot"
            )

    def test_committed_rate_zero_after_failed_one_shot(self, setup):
        """_committed_rate stays at 0.0 after failed one-shot, so gradual
        loop starts from the beginning."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.1
        tuner._patch_trainer()

        tuner._before_cycle()
        tuner._adaptation(1.0)

        assert tuner._committed_rate == 0.0

    def test_before_cycle_called_for_one_shot_and_gradual(self, setup):
        """_before_cycle() is called at least once for the one-shot, and
        again during gradual adaptation."""
        tuner = setup
        tuner._instant_acc_fn = lambda r: 0.1 if r >= 0.99 else 0.85
        tuner._post_acc_fn = lambda r: 0.87
        tuner._patch_trainer()

        tuner.run()

        # Once for one-shot + at least once for gradual
        assert tuner.before_cycle_calls >= 2
