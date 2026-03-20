"""End-to-end numerical assurance: original model vs converted Supermodel.

Tests that the full torch conversion pipeline (FX trace → representability →
mapper DAG → Supermodel) preserves forward-pass fidelity for each supported
model architecture. Failures here manifest as catastrophic accuracy loss
after the TorchMappingStep in the deployment pipeline.

These tests bypass InputCQ by calling supermodel.perceptron_flow(x) directly,
isolating the mapper DAG from input quantization.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper
from mimarsinan.torch_mapping.converter import convert_torch_model


# ── Test models ─────────────────────────────────────────────────────────


class LinearBNReLU(nn.Module):
    """Simple: Linear -> BN -> ReLU -> Linear."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


class ConvModel(nn.Module):
    """Conv2d -> BN -> ReLU -> Flatten -> Linear (no activation on last layer)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        return self.fc(x)


class MultiLayerConvModel(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> LeakyReLU -> Flatten -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.act2 = nn.LeakyReLU()
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)


class IdentityActivationModel(nn.Module):
    """Linear -> Linear (no activations at all). Tests Identity preservation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc2(self.fc1(x))


# ── Parametric test ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_cls,input_shape,num_classes",
    [
        (LinearBNReLU, (1, 8, 8), 10),
        (ConvModel, (1, 8, 8), 10),
        (MultiLayerConvModel, (1, 8, 8), 10),
        (IdentityActivationModel, (1, 8, 8), 10),
    ],
    ids=["linear_bn_relu", "conv_bn_relu", "multi_conv", "identity_only"],
)
class TestE2ENumericalAssurance:
    def test_forward_numerical_match(self, model_cls, input_shape, num_classes):
        """Converted model output must numerically match original (atol=1e-3)."""
        torch.manual_seed(42)
        model = model_cls()
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, *input_shape))  # warmup BN

        supermodel = convert_torch_model(model, input_shape=input_shape, num_classes=num_classes)
        supermodel.eval()

        x = torch.randn(8, *input_shape)
        with torch.no_grad():
            orig = model(x)
            converted = supermodel.perceptron_flow(x)

        assert orig.shape == converted.shape
        diff = (orig - converted).abs().max().item()
        assert diff < 1e-3, (
            f"{model_cls.__name__}: max diff {diff:.6f}. "
            "Converted forward does not match original."
        )

    def test_argmax_agreement(self, model_cls, input_shape, num_classes):
        """Class predictions must agree 100% between original and converted."""
        torch.manual_seed(42)
        model = model_cls()
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, *input_shape))

        supermodel = convert_torch_model(model, input_shape=input_shape, num_classes=num_classes)
        supermodel.eval()

        x = torch.randn(16, *input_shape)
        with torch.no_grad():
            orig_pred = model(x).argmax(dim=1)
            conv_pred = supermodel.perceptron_flow(x).argmax(dim=1)

        agreement = (orig_pred == conv_pred).float().mean().item()
        assert agreement == 1.0, (
            f"{model_cls.__name__}: {int((1 - agreement) * 16)}/16 predictions disagree."
        )

    def test_every_perceptron_has_real_activation(self, model_cls, input_shape, num_classes):
        """Every PerceptronMapper must have a non-Identity activation.

        Activation-less layers (e.g. final classifier) should be
        ModuleComputeMapper, not PerceptronMapper with a fake Identity.
        """
        from mimarsinan.mapping.mappers.perceptron import PerceptronMapper, ModuleComputeMapper

        torch.manual_seed(42)
        model = model_cls()
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, *input_shape))

        supermodel = convert_torch_model(model, input_shape=input_shape, num_classes=num_classes)
        mapper_repr = supermodel.perceptron_flow.get_mapper_repr()
        mapper_repr._ensure_exec_graph()

        perceptron_mappers = [
            n for n in mapper_repr._exec_order if isinstance(n, PerceptronMapper)
        ]
        for pm in perceptron_mappers:
            assert pm.perceptron.base_activation_name != "Identity", (
                f"{model_cls.__name__}: PerceptronMapper '{pm.perceptron.name}' has "
                f"Identity activation — it should be a ModuleComputeMapper instead."
            )


# ── TorchMLPMixer specific test ────────────────────────────────────────


class TestTorchMLPMixerE2E:
    """Ensure the full TorchMLPMixer conversion pipeline preserves fidelity."""

    def test_forward_match(self):
        from mimarsinan.models.torch_mlp_mixer import TorchMLPMixer

        torch.manual_seed(42)
        model = TorchMLPMixer(
            input_shape=(1, 28, 28), num_classes=10,
            patch_n_1=4, patch_m_1=4, patch_c_1=16,
            fc_w_1=32, fc_w_2=32,
        )
        model.eval()

        supermodel = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        supermodel.eval()

        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            orig = model(x)
            converted = supermodel.perceptron_flow(x)

        diff = (orig - converted).abs().max().item()
        assert diff < 1e-3, f"TorchMLPMixer max diff {diff:.6f}"

    def test_conv_patch_embedding_is_compute_op(self):
        """Conv2d patch embedding (no activation) should be ModuleComputeMapper, not a perceptron."""
        from mimarsinan.mapping.mappers.perceptron import ModuleComputeMapper
        from mimarsinan.models.torch_mlp_mixer import TorchMLPMixer

        model = TorchMLPMixer(
            input_shape=(1, 28, 28), num_classes=10,
            patch_n_1=4, patch_m_1=4, patch_c_1=16,
            fc_w_1=32, fc_w_2=32,
        )
        model.eval()

        supermodel = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        repr_ = supermodel.get_mapper_repr()
        repr_._ensure_exec_graph()

        # Conv2d with no activation → ModuleComputeMapper, not Conv2DPerceptronMapper
        conv_mappers = [n for n in repr_._exec_order if isinstance(n, Conv2DPerceptronMapper)]
        assert not conv_mappers, (
            "Patch embedding Conv2d should be ModuleComputeMapper, "
            f"but found {len(conv_mappers)} Conv2DPerceptronMapper(s)"
        )

        # Should exist as a ModuleComputeMapper instead
        compute_mappers = [n for n in repr_._exec_order if isinstance(n, ModuleComputeMapper)]
        assert compute_mappers, "No ModuleComputeMapper found for patch embedding"
