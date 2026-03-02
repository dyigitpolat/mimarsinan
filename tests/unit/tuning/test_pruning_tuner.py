"""Tests for PruningTuner and AdaptationManager pruning_rate integration."""

import pytest
import torch
import torch.nn as nn
import copy

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.layers import LeakyGradReLU
from conftest import default_config, MockPipeline, make_tiny_supermodel


class TestAdaptationManagerPruningRate:
    def test_initial_pruning_rate_is_zero(self):
        am = AdaptationManager()
        assert am.pruning_rate == 0.0

    def test_pruning_rate_can_be_set(self):
        am = AdaptationManager()
        am.pruning_rate = 0.5
        assert am.pruning_rate == 0.5

    def test_update_activation_still_works_with_pruning_rate(self):
        """Setting pruning_rate should not break update_activation."""
        am = AdaptationManager()
        am.pruning_rate = 0.7
        p = Perceptron(8, 16)
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        p.set_activation_scale(2.0)
        cfg = default_config()
        am.update_activation(cfg, p)
        x = torch.randn(2, 16)
        out = p(x)
        assert not torch.isnan(out).any()


class TestPruningTuner:
    def test_constructs(self):
        """PruningTuner should be constructable with mock pipeline."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.25,
        )
        assert tuner is not None
        assert tuner.pruning_fraction == 0.25

    def test_update_evaluate_at_rate_one(self):
        """At rate=1.0, pruned weights should be zeroed."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.5,
        )
        original_weights = [p.layer.weight.data.clone() for p in model.get_perceptrons()]
        tuner._update_and_evaluate(1.0)

        # At least some weights should differ from original (should be zeroed)
        any_changed = False
        for orig, p in zip(original_weights, model.get_perceptrons()):
            if not torch.allclose(orig, p.layer.weight.data):
                any_changed = True
                # Some weights should now be zero
                assert (p.layer.weight.data == 0.0).any(), \
                    "Some weights should be zeroed at rate=1.0"
        assert any_changed, "At least one perceptron should have changed weights"

    def test_update_evaluate_at_rate_zero_preserves_weights(self):
        """At rate=0.0, weights should be unchanged."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.5,
        )
        original_weights = [p.layer.weight.data.clone() for p in model.get_perceptrons()]
        tuner._update_and_evaluate(0.0)

        for orig, p in zip(original_weights, model.get_perceptrons()):
            assert torch.allclose(orig, p.layer.weight.data), \
                "Weights should be unchanged at rate=0.0"
