"""Unit tests for :func:`mimarsinan.mapping.shape_probe.probe_module_io_shapes`."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.shape_probe import probe_module_io_shapes, ProbedShapes


class TestSingleInputShapes:
    def test_max_pool2d(self):
        out = probe_module_io_shapes(nn.MaxPool2d(kernel_size=2), (3, 32, 32))
        assert out.output_shape == (3, 16, 16)
        assert out.input_shape == (3, 32, 32)

    def test_avg_pool2d_stride(self):
        out = probe_module_io_shapes(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1), (4, 8, 8),
        )
        assert out.output_shape == (4, 4, 4)

    def test_adaptive_avg_pool2d(self):
        out = probe_module_io_shapes(nn.AdaptiveAvgPool2d((1, 1)), (4, 8, 8))
        assert out.output_shape == (4, 1, 1)

    def test_layer_norm(self):
        out = probe_module_io_shapes(nn.LayerNorm([16]), (8, 16))
        assert out.output_shape == (8, 16)

    def test_linear(self):
        out = probe_module_io_shapes(nn.Linear(20, 10), (20,))
        assert out.output_shape == (10,)

    def test_conv2d(self):
        out = probe_module_io_shapes(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            (3, 16, 16),
        )
        assert out.output_shape == (8, 16, 16)

    def test_custom_nn_module(self):
        class Halve(nn.Module):
            def forward(self, x):
                return x[:, ::2]

        out = probe_module_io_shapes(Halve(), (8,))
        assert out.output_shape == (4,)


class TestMultiInputShapes:
    def test_mhsa(self):
        out = probe_module_io_shapes(
            nn.MultiheadAttention(8, 2, batch_first=True),
            [(4, 8), (4, 8), (4, 8)],
            module_kwargs={"need_weights": False},
            output_index=0,
        )
        assert out.output_shape == (4, 8)
        assert out.input_shapes == ((4, 8), (4, 8), (4, 8))

    def test_two_input_add(self):
        class Add(nn.Module):
            def forward(self, a, b):
                return a + b

        out = probe_module_io_shapes(Add(), [(5,), (5,)])
        assert out.output_shape == (5,)


class TestProbedShapesDataclass:
    def test_input_shape_unary_only(self):
        out = probe_module_io_shapes(nn.Identity(), (4,))
        assert out.input_shape == (4,)

    def test_input_shape_raises_on_multi(self):
        out = probe_module_io_shapes(
            nn.MultiheadAttention(8, 2, batch_first=True),
            [(4, 8), (4, 8), (4, 8)],
            module_kwargs={"need_weights": False},
            output_index=0,
        )
        with pytest.raises(ValueError, match="unary"):
            _ = out.input_shape


class TestStatePreservation:
    def test_training_mode_restored(self):
        ln = nn.LayerNorm([4])
        ln.train()
        assert ln.training is True
        probe_module_io_shapes(ln, (4,))
        assert ln.training is True, "Probe must restore training mode"

    def test_eval_mode_preserved(self):
        ln = nn.LayerNorm([4])
        ln.eval()
        probe_module_io_shapes(ln, (4,))
        assert ln.training is False
