"""Test that Conv2DPerceptronMapper passes activation_type to IR NeuralCores.

Regression guard: Conv2DPerceptronMapper._map_to_ir previously did NOT pass
activation_type to add_shared_neural_core, causing all conv NeuralCores to
default to activation_type=None → F.relu in the spiking simulation.  This
silently applied ReLU even when the conv layer had Identity or GELU activation,
destroying negative activations and causing catastrophic accuracy loss.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping import IRMapping


class TestConvIRActivationType:
    """Verify Conv2DPerceptronMapper creates correct IR nodes based on activation."""

    def test_relu_conv_creates_neural_cores(self):
        """Conv with ReLU creates NeuralCores (chip-supported)."""
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
        assert len(neural_cores) > 0, "ReLU Conv should create NeuralCores"
        for core in neural_cores:
            assert core.activation_type == "LeakyGradReLU"

    @pytest.mark.parametrize("act_name", ["Identity", "GELU"])
    def test_unsupported_activation_creates_compute_ops(self, act_name):
        """Conv with non-chip activation (Identity, GELU) creates ComputeOps."""
        from mimarsinan.mapping.ir import ComputeOp
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
        linear_ops = [n for n in ir_mapping.nodes if isinstance(n, ComputeOp) and n.op_type == "linear"]
        assert len(neural_cores) == 0, f"{act_name} Conv should NOT create NeuralCores"
        assert len(linear_ops) > 0, f"{act_name} Conv should create linear ComputeOps"

    @pytest.mark.parametrize("act_type,expected_fn", [
        ("LeakyGradReLU", "relu"),
        ("ReLU", "relu"),
    ])
    def test_spiking_uses_correct_activation(self, act_type, expected_fn):
        """SpikingUnifiedCoreFlow resolves chip-supported activations correctly."""
        from mimarsinan.models.unified_core_flow import _ttfs_activation_from_type
        import torch.nn.functional as F

        act_fn = _ttfs_activation_from_type(act_type)
        x = torch.randn(1, 10)
        expected = getattr(F, expected_fn)(x)
        result = act_fn(x)
        assert torch.allclose(result, expected)


class TestConvIRNumericalEquivalence:
    """Verify Conv2D with Identity activation preserves negative values via ComputeOp."""

    def test_identity_conv_preserves_negatives(self):
        """Conv with Identity activation creates ComputeOps that preserve negatives."""
        from mimarsinan.mapping.model_representation import ModelRepresentation
        from mimarsinan.mapping.per_source_scales import compute_per_source_scales
        from mimarsinan.mapping.ir import ComputeOp

        inp = InputMapper((1, 4, 4))
        conv = Conv2DPerceptronMapper(
            inp,
            in_channels=1,
            out_channels=2,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            use_batchnorm=False,
            base_activation_name="Identity",
        )
        repr_ = ModelRepresentation(conv)

        # Set conv weights to produce negative outputs
        with torch.no_grad():
            conv.perceptron.layer.weight.fill_(-0.5)
            conv.perceptron.layer.bias.fill_(0.0)

        compute_per_source_scales(repr_)

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=256, max_neurons=256,
        )
        ir_graph = ir_mapping.map(repr_)

        # Identity conv should produce ComputeOps, not NeuralCores
        linear_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp) and n.op_type == "linear"]
        assert len(linear_ops) >= 1, "Identity conv should create linear ComputeOps"

        # Execute via IR directly and verify negatives are preserved
        x = torch.ones(1, 1, 4, 4) * 0.5
        x_flat = x.view(1, -1)
        buffers = {}
        for node in ir_graph.nodes:
            buffers[node.id] = node.execute(x_flat, buffers)

        # At least one ComputeOp should have negative outputs
        any_negative = any((buffers[n.id] < 0).any() for n in linear_ops)
        assert any_negative, (
            "Identity conv ComputeOp should preserve negative values "
            "(negative weights × positive input = negative output)"
        )
