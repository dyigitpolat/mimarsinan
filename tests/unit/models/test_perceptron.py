"""Tests for Perceptron: forward pass, scale setters, activation setup."""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.layers import LeakyGradReLU, TransformedActivation


class TestPerceptronForward:
    def test_output_shape(self):
        p = Perceptron(8, 16)
        x = torch.randn(4, 16)
        out = p(x)
        assert out.shape == (4, 8)

    def test_with_batchnorm(self):
        p = Perceptron(8, 16, normalization=nn.BatchNorm1d(8))
        p.train()
        x = torch.randn(4, 16)
        out = p(x)
        assert out.shape == (4, 8)

    def test_identity_normalization(self):
        p = Perceptron(4, 8, normalization=nn.Identity())
        x = torch.randn(2, 8)
        out = p(x)
        assert out.shape == (2, 4)

    def test_no_bias(self):
        p = Perceptron(4, 8, bias=False)
        assert p.layer.bias is None
        x = torch.randn(2, 8)
        out = p(x)
        assert out.shape == (2, 4)

    def test_regularization_only_in_training(self):
        p = Perceptron(4, 8)
        p.regularization = nn.Dropout(0.99)
        p.eval()
        x = torch.randn(2, 8)
        out = p(x)
        assert not torch.isnan(out).any()


class TestPerceptronScales:
    def test_set_activation_scale_float(self):
        p = Perceptron(4, 8)
        p.set_activation_scale(2.5)
        assert p.activation_scale.item() == pytest.approx(2.5)

    def test_set_activation_scale_tensor(self):
        p = Perceptron(4, 8)
        p.set_activation_scale(torch.tensor(3.0))
        assert p.activation_scale.item() == pytest.approx(3.0)

    def test_set_parameter_scale(self):
        p = Perceptron(4, 8)
        p.set_parameter_scale(127.0)
        assert p.parameter_scale.item() == pytest.approx(127.0)

    def test_set_input_activation_scale(self):
        p = Perceptron(4, 8)
        p.set_input_activation_scale(0.5)
        assert p.input_activation_scale.item() == pytest.approx(0.5)


class TestPerceptronActivation:
    def test_set_activation(self):
        p = Perceptron(4, 8)
        ta = TransformedActivation(LeakyGradReLU(), [])
        p.set_activation(ta)
        assert isinstance(p.activation, TransformedActivation)

    def test_default_activation_is_leaky_grad_relu(self):
        p = Perceptron(4, 8)
        assert isinstance(p.activation, LeakyGradReLU)

    def test_default_base_activation(self):
        p = Perceptron(4, 8)
        assert isinstance(p.base_activation, LeakyGradReLU)
