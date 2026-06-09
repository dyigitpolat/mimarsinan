"""The perceptron-rate application SSOT must match the inlined tuner loops."""

from __future__ import annotations

import copy

import torch.nn as nn

from conftest import default_config, make_tiny_supermodel

from mimarsinan.models.nn.layers import NoisyDropout
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.perceptron_rate import (
    apply_manager_rate,
    rebuild_activations,
    set_blend_rate,
)


def _model_and_manager():
    model = make_tiny_supermodel()
    am = AdaptationManager()
    cfg = default_config()
    cfg["target_tq"] = 4
    return model, am, cfg


class TestApplyManagerRate:
    def test_sets_field_and_rebuilds(self):
        model, am, cfg = _model_and_manager()
        apply_manager_rate(model, am, cfg, "noise_rate", 0.5)
        assert am.noise_rate == 0.5
        # update_activation installs NoisyDropout regularization when noise_rate>0
        for p in model.get_perceptrons():
            assert isinstance(p.regularization, NoisyDropout)

    def test_matches_inlined_loop_bit_for_bit(self):
        """apply_manager_rate == the legacy `setattr; for p: update_activation`."""
        model_a, am_a, cfg = _model_and_manager()
        model_b = copy.deepcopy(model_a)
        am_b = copy.deepcopy(am_a)

        # legacy inline form
        setattr(am_a, "quantization_rate", 0.3)
        for p in model_a.get_perceptrons():
            am_a.update_activation(cfg, p)
        # SSOT form
        apply_manager_rate(model_b, am_b, cfg, "quantization_rate", 0.3)

        assert am_a.quantization_rate == am_b.quantization_rate == 0.3
        for pa, pb in zip(model_a.get_perceptrons(), model_b.get_perceptrons()):
            assert type(pa.activation) is type(pb.activation)

    def test_zero_rate_clears_regularization(self):
        model, am, cfg = _model_and_manager()
        apply_manager_rate(model, am, cfg, "noise_rate", 0.5)
        apply_manager_rate(model, am, cfg, "noise_rate", 0.0)
        assert am.noise_rate == 0.0
        for p in model.get_perceptrons():
            assert isinstance(p.regularization, nn.Identity)


class TestRebuildActivations:
    def test_rebuilds_without_changing_rates(self):
        model, am, cfg = _model_and_manager()
        am.quantization_rate = 0.7
        rebuild_activations(model, am, cfg)
        assert am.quantization_rate == 0.7  # rebuild does not touch the field


class TestSetBlendRate:
    def test_sets_every_blend_rate(self):
        from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import BlendActivation

        model, am, cfg = _model_and_manager()
        for p in model.get_perceptrons():
            p.base_activation = BlendActivation(
                nn.ReLU(), nn.ReLU(), 0.0, target_type="T", old_type="ReLU",
            )
        set_blend_rate(model, 0.42)
        for p in model.get_perceptrons():
            assert p.base_activation.rate == 0.42
