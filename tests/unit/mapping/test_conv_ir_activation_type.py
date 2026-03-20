"""Test that Conv2DPerceptronMapper creates NeuralCores with correct activation_type.

Conv2DPerceptronMapper always creates NeuralCores (it always has a real activation).
Conv2d without activation goes through ModuleComputeMapper → generic ComputeOp.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.perceptron import ModuleComputeMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.ir import NeuralCore, ComputeOp
from mimarsinan.mapping.ir_mapping import IRMapping


class TestConvIRActivationType:
    """Verify Conv2DPerceptronMapper creates NeuralCores for all activations."""

    @pytest.mark.parametrize("act_name", ["ReLU", "LeakyReLU", "GELU"])
    def test_conv_creates_neural_cores(self, act_name):
        inp = InputMapper((1, 4, 4))
        conv = Conv2DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name=act_name,
        )
        ir_mapping = IRMapping(q_max=1, firing_mode="TTFS", max_axons=256, max_neurons=256)
        conv.map_to_ir(ir_mapping)

        neural_cores = [n for n in ir_mapping.nodes if isinstance(n, NeuralCore)]
        assert len(neural_cores) > 0, f"{act_name} Conv should create NeuralCores"

    def test_relu_conv_has_leaky_grad_relu_type(self):
        """ReLU Conv NeuralCores should have LeakyGradReLU activation_type."""
        inp = InputMapper((1, 4, 4))
        conv = Conv2DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="ReLU",
        )
        ir_mapping = IRMapping(q_max=1, firing_mode="TTFS", max_axons=256, max_neurons=256)
        conv.map_to_ir(ir_mapping)

        neural_cores = [n for n in ir_mapping.nodes if isinstance(n, NeuralCore)]
        for core in neural_cores:
            assert core.activation_type == "LeakyGradReLU"

    @pytest.mark.parametrize("act_type,expected_fn", [
        ("LeakyGradReLU", "relu"),
        ("ReLU", "relu"),
    ])
    def test_spiking_uses_correct_activation(self, act_type, expected_fn):
        """SpikingUnifiedCoreFlow resolves activations correctly."""
        from mimarsinan.models.unified_core_flow import _ttfs_activation_from_type
        import torch.nn.functional as F

        act_fn = _ttfs_activation_from_type(act_type)
        x = torch.randn(1, 10)
        expected = getattr(F, expected_fn)(x)
        result = act_fn(x)
        assert torch.allclose(result, expected)


class TestConvComputeOpViaModuleMapper:
    """Conv2d without activation → ModuleComputeMapper → generic ComputeOp."""

    def test_bare_conv_creates_compute_op(self):
        """Conv2d wrapped in ModuleComputeMapper creates a module ComputeOp."""
        inp = InputMapper((1, 4, 4))
        conv = nn.Conv2d(1, 2, kernel_size=2, stride=2, padding=0)
        mapper = ModuleComputeMapper(inp, conv, input_shape=(1, 4, 4), name="test_conv")

        ir_mapping = IRMapping(q_max=1, firing_mode="TTFS", max_axons=256, max_neurons=256)
        mapper.map_to_ir(ir_mapping)

        neural_cores = [n for n in ir_mapping.nodes if isinstance(n, NeuralCore)]
        compute_ops = [n for n in ir_mapping.nodes if isinstance(n, ComputeOp)]
        assert len(neural_cores) == 0, "Bare Conv should NOT create NeuralCores"
        assert len(compute_ops) >= 1, "Bare Conv should create ComputeOps"
