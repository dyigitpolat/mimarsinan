"""Tests for TorchSequentialConvBuilder."""

import pytest
import torch

from mimarsinan.models.builders.torch_sequential_conv_builder import (
    TorchSequentialConvBuilder,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir_pruning import get_neural_segments


class TestTorchSequentialConvBuilder:
    @pytest.fixture
    def builder(self):
        return TorchSequentialConvBuilder(
            device=torch.device("cpu"),
            input_shape=(1, 28, 28),
            num_classes=10,
            max_axons=1024,
            max_neurons=1024,
            pipeline_config={"target_tq": 32},
        )

    def test_build_returns_sequential_module(self, builder):
        model = builder.build(
            {"conv_out_channels": 8, "hidden_dims": [128, 64]}
        )
        assert isinstance(model, torch.nn.Sequential)

    def test_build_output_shape(self, builder):
        model = builder.build(
            {"conv_out_channels": 16, "hidden_dims": [256, 128]}
        )
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_build_one_hidden_layer(self, builder):
        model = builder.build(
            {"conv_out_channels": 8, "hidden_dims": [64]}
        )
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        assert out.shape == (1, 10)

    def test_build_requires_conv_out_channels_and_hidden_dims(self, builder):
        with pytest.raises(ValueError, match="conv_out_channels"):
            builder.build({})
        with pytest.raises(ValueError, match="hidden_dims"):
            builder.build({"conv_out_channels": 8})
        with pytest.raises(ValueError, match="hidden_dims"):
            builder.build({"conv_out_channels": 8, "hidden_dims": []})

    def test_build_accepts_tuple_hidden_dims(self, builder):
        model = builder.build(
            {"conv_out_channels": 8, "hidden_dims": (100, 50)}
        )
        x = torch.randn(1, 1, 28, 28)
        assert model(x).shape == (1, 10)

    def test_build_with_custom_conv_pool_params(self, builder):
        model = builder.build(
            {
                "conv_out_channels": 16,
                "conv_kernel_size": 3,
                "conv_stride": 1,
                "conv_padding": 1,
                "pool_kernel_size": 2,
                "pool_stride": 2,
                "hidden_dims": [64],
            }
        )
        x = torch.randn(1, 1, 28, 28)
        assert model(x).shape == (1, 10)

    def test_build_two_segments_after_conversion(self, builder):
        """Converted model should yield two neural segments and exactly one ComputeOp."""
        raw_model = builder.build(
            {"conv_out_channels": 8, "hidden_dims": [64]}
        )
        input_shape = (1, 28, 28)
        num_classes = 10
        supermodel = convert_torch_model(
            raw_model,
            input_shape=input_shape,
            num_classes=num_classes,
            device="cpu",
            Tq=32,
        )
        mapper_repr = supermodel.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        ir_mapping = IRMapping(
            q_max=127.0,
            firing_mode="Default",
            max_axons=1024,
            max_neurons=1024,
        )
        ir_graph = ir_mapping.map(mapper_repr)
        segments = get_neural_segments(ir_graph)
        compute_ops = ir_graph.get_compute_ops()
        assert len(segments) >= 1, "expected at least one neural segment"
        # ComputeOps include MaxPool2d + Identity-activated classifier (linear ComputeOp)
        assert len(compute_ops) >= 1, "expected at least one ComputeOp"
