"""Tests for PruningTuner and AdaptationManager pruning_rate integration."""

import pytest
import torch
import torch.nn as nn
import copy
from unittest.mock import patch

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

    def test_first_layer_columns_and_last_layer_rows_exempt_at_rate_one(self):
        """At rate=1.0, first layer column mask and last layer row mask must be all True
        (input-buffer and output-buffer dimensions are never pruned)."""
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
            pruning_fraction=1.0,
        )

        perceptrons = model.get_perceptrons()
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))

        row_masks, col_masks = tuner._get_masks(1.0)

        assert col_masks[0].all(), "First layer column mask must be all True (input-buffer exempt)"
        assert row_masks[-1].all(), "Last layer row mask must be all True (output-buffer exempt)"

    def test_refresh_pruning_importance_called_per_cycle(self):
        """When run(max_cycles=N) is used, activation stats (importance) should be collected at least N times."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.transformations.pruning import _collect_activation_stats

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.99,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.25,
        )
        # Low validation (0.5) so adapter takes small steps and runs 3 cycles before t>=1
        tuner.trainer = type(
            "T",
            (),
            {
                "validate": lambda self: 0.5,
                "validate_n_batches": lambda self, n: 0.5,
                "train_one_step": lambda self, lr: None,
                "train_until_target_accuracy": lambda self, *a: None,
                "train_steps_until_target": lambda self, *a, **k: None,
            },
        )()
        tuner.trainer.validation_loader = [(torch.randn(2, 1, 8, 8), torch.zeros(2, dtype=torch.long))]
        collect_calls = []

        def counting_collect(*args, **kwargs):
            collect_calls.append(1)
            return _collect_activation_stats(*args, **kwargs)

        with patch("mimarsinan.tuning.tuners.pruning_tuner._collect_activation_stats", side_effect=counting_collect):
            tuner.run(max_cycles=3)
        assert len(collect_calls) >= 3, "Activation stats should be collected at least once per cycle (3 cycles)"
