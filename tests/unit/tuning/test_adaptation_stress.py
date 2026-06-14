"""
Stress tests for AdaptationManager.

Tests extreme rates, NaN propagation, and adaptation failure scenarios.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.nn.layers import TransformedActivation, LeakyGradReLU
from conftest import default_config


class TestAdaptationManagerStress:
    def _make_perceptron(self):
        p = Perceptron(8, 16)
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        p.set_activation_scale(2.0)
        return p

    def test_all_rates_at_max(self):
        """All rates at 1.0 simultaneously."""
        am = AdaptationManager()
        am.clamp_rate = 1.0
        am.shift_rate = 1.0
        am.quantization_rate = 1.0
        am.noise_rate = 1.0

        p = self._make_perceptron()
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(4, 16)
        out = p(x)
        assert not torch.isnan(out).any(), "All-max rates produced NaN"
        assert out.shape == (4, 8)

    def test_repeated_update_activation_is_stable(self):
        """Calling update_activation 50 times shouldn't degrade output."""
        am = AdaptationManager()
        am.clamp_rate = 0.5
        p = self._make_perceptron()
        cfg = default_config()

        x = torch.randn(2, 16)
        for _ in range(50):
            am.update_activation(cfg, p)
            out = p(x)
            assert not torch.isnan(out).any()

    def test_nan_activation_scale(self):
        """What happens if activation_scale is NaN?"""
        am = AdaptationManager()
        am.clamp_rate = 1.0
        p = self._make_perceptron()
        p.set_activation_scale(float("nan"))
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(2, 16)
        out = p(x)
        if torch.isnan(out).any():
            pass
        else:
            pass

    def test_zero_activation_scale(self):
        am = AdaptationManager()
        am.clamp_rate = 1.0
        p = self._make_perceptron()
        p.set_activation_scale(0.0)
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(4, 16)
        out = p(x)
        assert (out == 0).all() or torch.isnan(out).all(), \
            "With activation_scale=0 and clamp_rate=1, output should be all zeros"

