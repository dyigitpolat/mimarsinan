"""Tests for TorchSequentialLinearBuilder."""

import pytest
import torch

from mimarsinan.models.builders.torch_sequential_linear_builder import (
    TorchSequentialLinearBuilder,
)


class TestTorchSequentialLinearBuilder:
    @pytest.fixture
    def builder(self):
        return TorchSequentialLinearBuilder(
            device=torch.device("cpu"),
            input_shape=(1, 28, 28),
            num_classes=10,
            pipeline_config={"target_tq": 32},
        )

    def test_build_returns_sequential_module(self, builder):
        model = builder.build({"hidden_dims": [256, 128]})
        assert isinstance(model, torch.nn.Sequential)

    def test_build_output_shape(self, builder):
        model = builder.build({"hidden_dims": [512, 256, 128]})
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_build_one_hidden_layer(self, builder):
        model = builder.build({"hidden_dims": [64]})
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        assert out.shape == (1, 10)

    def test_build_requires_hidden_dims(self, builder):
        with pytest.raises(ValueError, match="hidden_dims"):
            builder.build({})
        with pytest.raises(ValueError, match="hidden_dims"):
            builder.build({"hidden_dims": []})

    def test_build_accepts_tuple_hidden_dims(self, builder):
        model = builder.build({"hidden_dims": (100, 50)})
        x = torch.randn(1, 1, 28, 28)
        assert model(x).shape == (1, 10)
