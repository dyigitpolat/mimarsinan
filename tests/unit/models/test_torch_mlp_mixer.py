"""Unit tests for TorchMLPMixer and TorchMLPMixerBuilder."""

import pytest
import torch

from mimarsinan.models.torch_mlp_mixer import TorchMLPMixer
from mimarsinan.models.builders.torch_mlp_mixer_builder import TorchMLPMixerBuilder
from mimarsinan.pipelining.model_registry import ModelRegistry


class TestTorchMLPMixer:
    def test_forward_shape(self):
        model = TorchMLPMixer(
            input_shape=(3, 32, 32),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=32,
            fc_w_1=64,
            fc_w_2=64,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_mnist_shape(self):
        model = TorchMLPMixer(
            input_shape=(1, 28, 28),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=16,
            fc_w_1=32,
            fc_w_2=32,
        )
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)


class TestTorchMLPMixerBuilder:
    def test_build_returns_module(self):
        builder = TorchMLPMixerBuilder(
            device=torch.device("cpu"),
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=256,
            max_neurons=256,
            pipeline_config={"target_tq": 32},
        )
        config = {
            "patch_n_1": 4,
            "patch_m_1": 4,
            "patch_c_1": 32,
            "fc_w_1": 64,
            "fc_w_2": 64,
        }
        model = builder.build(config)
        assert isinstance(model, torch.nn.Module)
        assert not hasattr(model, "get_mapper_repr")

    def test_build_output_shape(self):
        builder = TorchMLPMixerBuilder(
            device=torch.device("cpu"),
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=256,
            max_neurons=256,
            pipeline_config={"target_tq": 32},
        )
        config = {
            "patch_n_1": 4,
            "patch_m_1": 4,
            "patch_c_1": 32,
            "fc_w_1": 64,
            "fc_w_2": 64,
        }
        model = builder.build(config)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_mlp_mixer_registered_torch_category(self):
        assert ModelRegistry.get_category("mlp_mixer") == "torch"

    def test_get_nas_search_options(self):
        opts = TorchMLPMixerBuilder.get_nas_search_options((1, 28, 28))
        assert "patch_n_1" in opts
        assert "fc_w_1" in opts
        assert 4 in opts["patch_n_1"]
        assert 7 in opts["patch_n_1"]

    def test_validate_config(self):
        assert TorchMLPMixerBuilder.validate_config(
            {"patch_n_1": 4, "patch_m_1": 4},
            {},
            (1, 28, 28),
        ) is True
        assert TorchMLPMixerBuilder.validate_config(
            {"patch_n_1": 5, "patch_m_1": 4},
            {},
            (1, 28, 28),
        ) is False
