"""Tests for tiny PerceptronFlow fixtures: forward, perceptrons, mapper repr."""

import pytest
import torch

from conftest import make_tiny_supermodel


class TestTinyPerceptronFlow:
    def test_forward_shape(self):
        m = make_tiny_supermodel(input_shape=(1, 8, 8), num_classes=4)
        m.eval()
        x = torch.randn(3, 1, 8, 8)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (3, 4)

    def test_get_perceptrons(self):
        m = make_tiny_supermodel()
        ps = m.get_perceptrons()
        assert len(ps) == 2

    def test_get_mapper_repr(self):
        m = make_tiny_supermodel()
        mr = m.get_mapper_repr()
        assert mr is not None

    def test_get_input_activation(self):
        m = make_tiny_supermodel()
        ia = m.get_input_activation()
        x = torch.randn(2, 1, 8, 8)
        assert torch.allclose(ia(x), x)

    def test_different_input_shapes(self):
        for shape in [(1, 4, 4), (3, 8, 8)]:
            m = make_tiny_supermodel(input_shape=shape, num_classes=2)
            m.eval()
            with torch.no_grad():
                out = m(torch.randn(2, *shape))
            assert out.shape == (2, 2)

    def test_first_perceptron_marked_encoding(self):
        m = make_tiny_supermodel()
        assert m.get_perceptrons()[0].is_encoding_layer is True
        assert m.get_perceptrons()[1].is_encoding_layer is False
