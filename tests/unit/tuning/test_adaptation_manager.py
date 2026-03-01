"""Tests for AdaptationManager: activation decoration and rate management."""

import pytest
import torch
import torch.nn as nn

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.layers import (
    TransformedActivation, LeakyGradReLU, RateAdjustedDecorator,
)
from conftest import default_config


class TestAdaptationManager:
    def _make_perceptron(self):
        p = Perceptron(8, 16)
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        p.set_activation_scale(2.0)
        return p

    def test_initial_rates_are_zero(self):
        am = AdaptationManager()
        assert am.clamp_rate == 0.0
        assert am.shift_rate == 0.0
        assert am.quantization_rate == 0.0
        assert am.noise_rate == 0.0

    def test_update_activation_sets_transformed(self):
        am = AdaptationManager()
        p = self._make_perceptron()
        cfg = default_config()
        am.update_activation(cfg, p)
        assert isinstance(p.activation, TransformedActivation)

    def test_update_activation_forward_works(self):
        am = AdaptationManager()
        p = self._make_perceptron()
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(4, 16)
        out = p(x)
        assert out.shape == (4, 8)
        assert not torch.isnan(out).any()

    def test_clamp_rate_affects_activation(self):
        am = AdaptationManager()
        am.clamp_rate = 1.0
        p = self._make_perceptron()
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(4, 16)
        out = p(x)
        assert out.max() <= p.activation_scale.item() + 1e-5

    def test_rates_can_be_set(self):
        am = AdaptationManager()
        am.clamp_rate = 0.5
        am.quantization_rate = 0.3
        am.shift_rate = 0.1
        assert am.clamp_rate == 0.5
        assert am.quantization_rate == 0.3
        assert am.shift_rate == 0.1

    def test_multiple_updates_dont_break(self):
        """Calling update_activation multiple times should be stable."""
        am = AdaptationManager()
        p = self._make_perceptron()
        cfg = default_config()
        for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
            am.clamp_rate = rate
            am.update_activation(cfg, p)
            x = torch.randn(2, 16)
            out = p(x)
            assert not torch.isnan(out).any()
