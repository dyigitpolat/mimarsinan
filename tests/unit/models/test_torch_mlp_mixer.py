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


class TestTorchMLPMixerConversion:
    """Verify that convert_torch_model preserves forward-pass fidelity.

    The Supermodel's forward pass (used by TorchMappingStep validation) must
    agree with the original PyTorch model.  A failure here manifests as ~10%
    (chance-level) accuracy right after the Torch Mapping step.
    """

    def test_supermodel_forward_matches_original(self):
        """Supermodel output must numerically match the original model."""
        from mimarsinan.torch_mapping.converter import convert_torch_model

        model = TorchMLPMixer(
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
            # Compare perceptron_flow directly (bypasses InputCQ preprocessing)
            # to isolate the mapper DAG fidelity from input quantization.
            super_out = supermodel.perceptron_flow(x)

        assert orig_out.shape == super_out.shape, (
            f"Shape mismatch: {orig_out.shape} vs {super_out.shape}"
        )
        assert torch.allclose(orig_out, super_out, atol=1e-3), (
            f"Output mismatch — max diff: {(orig_out - super_out).abs().max().item():.6f}. "
            "The Supermodel forward pass does not faithfully reproduce the original model."
        )

    def test_supermodel_argmax_matches_original(self):
        """Class predictions from the Supermodel must match the original model."""
        from mimarsinan.torch_mapping.converter import convert_torch_model

        torch.manual_seed(42)
        model = TorchMLPMixer(
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

        x = torch.randn(16, 1, 28, 28)
        with torch.no_grad():
            orig_pred = model(x).argmax(dim=1)
            # Compare perceptron_flow directly (bypasses InputCQ preprocessing)
            super_pred = supermodel.perceptron_flow(x).argmax(dim=1)

        agreement = (orig_pred == super_pred).float().mean().item()
        assert agreement == 1.0, (
            f"Class predictions disagree on {int((1 - agreement) * 16)}/16 samples. "
            "TorchMappingStep validation will show chance-level accuracy."
        )


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
