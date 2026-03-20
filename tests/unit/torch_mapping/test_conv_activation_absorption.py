"""Test that Conv2d/Conv1d conversion correctly absorbs (or omits) activations.

Regression guard for the bug where _convert_conv2d did not look for absorbed
activation followers, causing Perceptron to default to LeakyGradReLU even when
the original model had no activation after the Conv layer.

Note: Conv perceptrons with Identity activation are *host-side* (not chip-targeted) and
are therefore excluded from ``model.get_perceptrons()``.  These tests access such
perceptrons directly via the mapper graph to verify the converter's absorption logic,
independently of whether the perceptron is exposed to pipeline tuning.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper
from mimarsinan.torch_mapping.converter import convert_torch_model


# ── Minimal test models ────────────────────────────────────────────────


class ConvBNOnly(nn.Module):
    """Conv2d -> BatchNorm2d -> Flatten -> Linear (no activation after conv)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> Flatten -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ConvBNGELU(nn.Module):
    """Conv2d -> BatchNorm2d -> GELU -> Flatten -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ConvBNLeakyReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> LeakyReLU -> Flatten -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.act = nn.LeakyReLU()
        self.fc = nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ── Helpers ──────────────────────────────────────────────────────────────


def _convert(model_cls):
    model = model_cls()
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 1, 8, 8))  # warmup BN
    return convert_torch_model(model, input_shape=(1, 8, 8), num_classes=4)


def _get_conv_perceptron_activation_name(model_cls):
    """Convert model, return the conv perceptron's base_activation_name.

    Accesses the perceptron directly from its Conv2DPerceptronMapper node in
    the mapper graph rather than through ``get_perceptrons()``.  Identity conv
    perceptrons are host-side and intentionally excluded from the pipeline-
    visible perceptron list, so direct traversal is required to test absorption.
    """
    supermodel = _convert(model_cls)
    repr_ = supermodel.get_mapper_repr()
    repr_._ensure_exec_graph()
    conv_mappers = [n for n in repr_._exec_order if isinstance(n, Conv2DPerceptronMapper)]
    assert conv_mappers, "No Conv2DPerceptronMapper found in converted model"
    return conv_mappers[0].perceptron.base_activation_name


# ── Tests ───────────────────────────────────────────────────────────────


class TestConv2dActivationAbsorption:
    def test_no_activation_gives_identity(self):
        """Conv2d -> BN (no activation) must produce Identity, not ReLU."""
        name = _get_conv_perceptron_activation_name(ConvBNOnly)
        assert name == "Identity", (
            f"Conv perceptron got activation '{name}' instead of 'Identity'. "
            "Spurious ReLU would zero ~50% of BN outputs."
        )

    def test_relu_absorbed(self):
        name = _get_conv_perceptron_activation_name(ConvBNReLU)
        assert name == "ReLU"

    def test_gelu_absorbed(self):
        name = _get_conv_perceptron_activation_name(ConvBNGELU)
        assert name == "GELU"

    def test_leakyrelu_absorbed(self):
        name = _get_conv_perceptron_activation_name(ConvBNLeakyReLU)
        assert name == "LeakyReLU"


class TestConv2dNumericalFidelity:
    """Converted model output must match the original for each activation config."""

    @pytest.mark.parametrize("model_cls", [ConvBNOnly, ConvBNReLU, ConvBNGELU, ConvBNLeakyReLU])
    def test_forward_matches_original(self, model_cls):
        torch.manual_seed(99)
        model = model_cls()
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 1, 8, 8))  # warmup BN

        supermodel = convert_torch_model(model, input_shape=(1, 8, 8), num_classes=4)
        supermodel.eval()

        x = torch.randn(4, 1, 8, 8)
        with torch.no_grad():
            orig = model(x)
            converted = supermodel.perceptron_flow(x)

        diff = (orig - converted).abs().max().item()
        assert diff < 1e-3, (
            f"{model_cls.__name__}: max diff {diff:.6f} >= 1e-3 between original and converted."
        )
