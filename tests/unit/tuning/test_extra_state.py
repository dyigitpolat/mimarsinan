"""Tests for the extra-state protocol in clone/restore.

Verifies that _get_extra_state / _set_extra_state correctly round-trip
tuner-specific decoration state (AdaptationManager rates) alongside
model parameters, preventing the stale-activation bug.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
    ActivationAdaptationTuner,
)
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner
from mimarsinan.tuning.tuners.activation_quantization_tuner import (
    ActivationQuantizationTuner,
)
from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner


def _make_pipeline(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


class TestActivationAdaptationExtraState:
    @pytest.fixture
    def tuner(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            am.update_activation(pipeline.config, p)
        return ActivationAdaptationTuner(pipeline, model, 0.9, 0.001, am)

    def test_clone_captures_activation_rate(self, tuner):
        tuner.adaptation_manager.activation_adaptation_rate = 0.6
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        state = tuner._clone_state()
        _, extra = state
        rate, base_acts = extra
        assert rate == pytest.approx(0.6)

    def test_restore_resets_activation_rate(self, tuner):
        state_at_zero = tuner._clone_state()

        tuner.adaptation_manager.activation_adaptation_rate = 0.8
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)
        assert tuner.adaptation_manager.activation_adaptation_rate == 0.8

        tuner._restore_state(state_at_zero)
        assert tuner.adaptation_manager.activation_adaptation_rate == pytest.approx(0.0)

    def test_clone_captures_base_activation(self, tuner):
        """base_activation must be part of the extra state snapshot."""
        from mimarsinan.models.perceptron_mixer.perceptron import make_activation

        perceptrons = list(tuner.model.get_perceptrons())
        original_names = [p.base_activation_name for p in perceptrons]

        state = tuner._clone_state()

        for p in perceptrons:
            p.base_activation = make_activation("ReLU")
            p.base_activation_name = "ReLU"

        tuner._restore_state(state)

        for p, orig_name in zip(tuner.model.get_perceptrons(), original_names):
            assert p.base_activation_name == orig_name

    def test_forward_pass_consistent_after_restore(self, tuner):
        """After restore, the model's forward pass should produce the same
        output as before the rate was changed."""
        x = torch.randn(2, 1, 8, 8)

        tuner.model.eval()
        with torch.no_grad():
            out_before = tuner.model(x).clone()

        state = tuner._clone_state()

        tuner.adaptation_manager.activation_adaptation_rate = 1.0
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        tuner._restore_state(state)

        tuner.model.eval()
        with torch.no_grad():
            out_after = tuner.model(x)

        assert torch.allclose(out_before, out_after, atol=1e-5), (
            "Forward pass must match after clone/restore round-trip"
        )


class TestClampTunerExtraState:
    @pytest.fixture
    def tuner(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            am.update_activation(pipeline.config, p)
        n_perceptrons = len(list(model.get_perceptrons()))
        scales = [1.0] * n_perceptrons
        from conftest import make_activation_scale_stats
        stats = make_activation_scale_stats(model, scales)
        return ClampTuner(pipeline, model, 0.9, 0.001, am, scales, stats)

    def test_clone_captures_clamp_rate(self, tuner):
        tuner.adaptation_manager.clamp_rate = 0.5
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        state = tuner._clone_state()
        _, extra = state
        assert extra == pytest.approx(0.5)

    def test_restore_resets_clamp_rate(self, tuner):
        state_at_zero = tuner._clone_state()

        tuner.adaptation_manager.clamp_rate = 0.7
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        tuner._restore_state(state_at_zero)
        assert tuner.adaptation_manager.clamp_rate == pytest.approx(0.0)


class TestActivationQuantizationExtraState:
    @pytest.fixture
    def tuner(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            am.update_activation(pipeline.config, p)
        return ActivationQuantizationTuner(pipeline, model, 4, 0.9, 0.001, am)

    def test_clone_captures_quantization_rate(self, tuner):
        tuner.adaptation_manager.quantization_rate = 0.3
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        state = tuner._clone_state()
        _, extra = state
        assert extra == pytest.approx(0.3)

    def test_restore_resets_quantization_rate(self, tuner):
        state_at_zero = tuner._clone_state()

        tuner.adaptation_manager.quantization_rate = 0.9
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        tuner._restore_state(state_at_zero)
        assert tuner.adaptation_manager.quantization_rate == pytest.approx(0.0)


class TestNoiseTunerExtraState:
    @pytest.fixture
    def tuner(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            am.update_activation(pipeline.config, p)
        return NoiseTuner(pipeline, model, 0.9, 0.001, am)

    def test_clone_captures_noise_rate(self, tuner):
        tuner.adaptation_manager.noise_rate = 0.4
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        state = tuner._clone_state()
        _, extra = state
        assert extra == pytest.approx(0.4)

    def test_restore_resets_noise_rate(self, tuner):
        state_at_zero = tuner._clone_state()

        tuner.adaptation_manager.noise_rate = 0.6
        for p in tuner.model.get_perceptrons():
            tuner.adaptation_manager.update_activation(tuner.pipeline.config, p)

        tuner._restore_state(state_at_zero)
        assert tuner.adaptation_manager.noise_rate == pytest.approx(0.0)
