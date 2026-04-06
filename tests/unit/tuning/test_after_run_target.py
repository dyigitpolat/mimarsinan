"""Tests verifying that _after_run() recovery training uses the original
target accuracy, not a decayed one.

With target decay removed, _get_target() always returns the original
target_accuracy. This ensures recovery training in _after_run() aims for
the correct accuracy level.
"""

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner
from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner
from mimarsinan.tuning.adaptation_manager import AdaptationManager


class TestAfterRunTarget:
    """_after_run() calls train_steps_until_target(..., self._get_target(), ...).
    Verify that _get_target() returns the original target after adaptation."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        cfg["degradation_tolerance"] = 0.05
        return MockPipeline(config=cfg, working_directory=str(tmp_path))

    def test_clamp_tuner_target_is_original(self, pipeline):
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            p.set_activation_scale(1.0)
            am.update_activation(pipeline.config, p)
        from conftest import make_activation_scale_stats
        tuner = ClampTuner(
            pipeline, model, target_accuracy=0.85, lr=0.001,
            adaptation_manager=am,
            activation_scales=[1.0] * len(list(model.get_perceptrons())),
            activation_scale_stats=make_activation_scale_stats(
                model, [1.0] * len(list(model.get_perceptrons()))
            ),
        )
        assert tuner._get_target() == 0.85

    def test_activation_quantization_tuner_target_is_original(self, pipeline):
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = ActivationQuantizationTuner(
            pipeline, model, target_tq=4, target_accuracy=0.90, lr=0.001,
            adaptation_manager=am,
        )
        assert tuner._get_target() == 0.90

    def test_noise_tuner_target_is_original(self, pipeline):
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = NoiseTuner(pipeline, model, target_accuracy=0.88, lr=0.001,
                          adaptation_manager=am)
        assert tuner._get_target() == 0.88

    def test_target_calibrated_from_baseline_after_run(self, pipeline):
        """After run(), _get_target() returns the baseline validation accuracy
        (calibrated from validate_n_batches at rate 0.0), not the originally
        passed target_accuracy."""
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = ActivationQuantizationTuner(
            pipeline, model, target_tq=4, target_accuracy=0.90, lr=0.001,
            adaptation_manager=am,
        )
        tuner.run()
        # After baseline calibration, the target must reflect actual model
        # performance (a small random model on tiny data), NOT the originally
        # passed 0.90.
        assert tuner._get_target() != 0.90
        assert 0.0 <= tuner._get_target() <= 1.0
