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

    def test_apply_pruning_at_rate_one_zeros_weights(self):
        """At rate=1.0, pruned weights should be zeroed."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.transformations.pruning import apply_pruning_masks

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

        perceptrons = model.get_perceptrons()
        # Manually set up importance (normally done in run())
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))

        original_weights = [p.layer.weight.data.clone() for p in perceptrons]
        row_masks, col_masks = tuner._get_masks(1.0)

        for i, p in enumerate(perceptrons):
            apply_pruning_masks(p, row_masks[i], col_masks[i], 1.0,
                                original_weights[i], None)

        # At rate=1.0, pruned weights should be exactly zero
        any_zeroed = False
        for p in perceptrons:
            if (p.layer.weight.data == 0.0).any():
                any_zeroed = True
        assert any_zeroed, "At least some weights should be zeroed at rate=1.0"

    def test_apply_pruning_at_rate_zero_preserves_weights(self):
        """At rate=0.0, weights should be unchanged."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.transformations.pruning import apply_pruning_masks

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

        perceptrons = model.get_perceptrons()
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))

        original_weights = [p.layer.weight.data.clone() for p in perceptrons]
        row_masks, col_masks = tuner._get_masks(1.0)

        for i, p in enumerate(perceptrons):
            apply_pruning_masks(p, row_masks[i], col_masks[i], 0.0,
                                original_weights[i], None)

        for orig, p in zip(original_weights, perceptrons):
            assert torch.allclose(orig, p.layer.weight.data), \
                "Weights should be unchanged at rate=0.0"
