"""Tests for Supermodel: forward, perceptrons, mapper repr."""

import pytest
import torch

from conftest import make_tiny_supermodel


class TestSupermodel:
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

    def test_get_preprocessor(self):
        m = make_tiny_supermodel()
        pp = m.get_preprocessor()
        assert pp is not None

    def test_different_input_shapes(self):
        for shape in [(1, 4, 4), (3, 8, 8)]:
            m = make_tiny_supermodel(input_shape=shape, num_classes=2)
            m.eval()
            with torch.no_grad():
                out = m(torch.randn(2, *shape))
            assert out.shape == (2, 2)

    def test_in_act_clamps_to_01(self):
        m = make_tiny_supermodel()
        x_large = torch.ones(1, 1, 8, 8) * 5.0
        pp = m.get_preprocessor()
        preprocessed = pp(x_large)
        activated = m.in_act(preprocessed)
        assert activated.max() <= 1.0 + 1e-6
        assert activated.min() >= -1e-6
