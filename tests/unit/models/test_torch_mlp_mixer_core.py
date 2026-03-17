"""Unit tests for TorchMLPMixerCore and TorchMLPMixerCoreBuilder."""

import pytest
import torch

from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.models.builders.torch_mlp_mixer_core_builder import TorchMLPMixerCoreBuilder
from mimarsinan.pipelining.model_registry import ModelRegistry
from mimarsinan.mapping.mappers.base import is_chip_supported_activation


class TestTorchMLPMixerCore:
    def test_forward_shape(self):
        model = TorchMLPMixerCore(
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
        model = TorchMLPMixerCore(
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


class TestTorchMLPMixerCoreConversion:
    """Verify conversion fidelity and that all mixer FCs are chip-packaged perceptrons."""

    def test_supermodel_forward_matches_original(self):
        """Supermodel output must numerically match the original model."""
        from mimarsinan.torch_mapping.converter import convert_torch_model

        model = TorchMLPMixerCore(
            input_shape=(1, 28, 28),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=32,
            fc_w_1=64,
            fc_w_2=64,
        )
        model.eval()
        supermodel = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        supermodel.eval()

        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            orig_out = model(x)
            super_out = supermodel.perceptron_flow(x)

        assert orig_out.shape == super_out.shape
        assert torch.allclose(orig_out, super_out, atol=1e-3), (
            f"Output mismatch — max diff: {(orig_out - super_out).abs().max().item():.6f}. "
            "The Supermodel forward pass does not faithfully reproduce the original model."
        )

    def test_all_mixer_fc_perceptrons_chip_supported(self):
        """Every mixer FC (8 total) plus the patch embed must have chip-supported activation.

        get_perceptrons() includes: patch embed (Conv mapper absorbs BN+ReLU, chip-supported),
        then 8 mixer FCs (all chip-supported for Core). Classifier is Identity and excluded by
        PerceptronMapper.owned_perceptron_groups (not chip-supported). So we expect 9 total,
        all 9 chip-supported.
        """
        from mimarsinan.torch_mapping.converter import convert_torch_model

        model = TorchMLPMixerCore(
            input_shape=(1, 28, 28),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=16,
            fc_w_1=32,
            fc_w_2=32,
        )
        model.eval()
        supermodel = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        perceptrons = supermodel.get_perceptrons()

        chip_supported = [p for p in perceptrons if is_chip_supported_activation(p)]

        assert len(perceptrons) == 9, (
            f"Expected 9 perceptrons (1 patch + 8 mixer FCs). Got {len(perceptrons)}."
        )
        assert len(chip_supported) == 9, (
            f"Expected 9 chip-supported perceptrons (patch+mixer FCs). Got {len(chip_supported)}."
        )


class TestTorchMLPMixerCoreBuilder:
    def test_build_returns_module(self):
        builder = TorchMLPMixerCoreBuilder(
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
        assert isinstance(model, TorchMLPMixerCore)

    def test_build_output_shape(self):
        builder = TorchMLPMixerCoreBuilder(
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

    def test_mlp_mixer_core_registered_torch_category(self):
        assert ModelRegistry.get_category("mlp_mixer_core") == "torch"

    def test_get_nas_search_options(self):
        opts = TorchMLPMixerCoreBuilder.get_nas_search_options((1, 28, 28))
        assert "patch_n_1" in opts
        assert "fc_w_1" in opts
        assert 4 in opts["patch_n_1"]

    def test_validate_config(self):
        assert TorchMLPMixerCoreBuilder.validate_config(
            {"patch_n_1": 4, "patch_m_1": 4},
            {},
            (1, 28, 28),
        ) is True
        assert TorchMLPMixerCoreBuilder.validate_config(
            {"patch_n_1": 5, "patch_m_1": 4},
            {},
            (1, 28, 28),
        ) is False
