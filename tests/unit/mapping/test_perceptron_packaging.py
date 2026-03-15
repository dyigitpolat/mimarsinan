"""Perceptron packaging: Identity layers → ComputeOps, ReLU layers → NeuralCores.

The chip only supports ReLU activation. Layers packaged as Perceptrons
(matmul + bn + chip_activation) map to NeuralCores. Layers with Identity
activation cannot run on the crossbar and become host-side ComputeOps.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.mappers.structural import (
    InputMapper, EinopsRearrangeMapper,
)
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ComputeOp
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


def _map_to_ir(mapper_repr, input_shape):
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(
        q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
    )
    return ir_mapping.map(mapper_repr)


class TestIdentityFCBecomesComputeOp:
    """FC layer with Identity activation → ComputeOp (host-side linear)."""

    def test_identity_fc_creates_compute_op(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(),
                       base_activation_name="Identity")
        fc = PerceptronMapper(flat, p)
        repr_ = ModelRepresentation(fc)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))
        compute_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp)]
        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]

        assert len(compute_ops) >= 1, "Identity FC should create at least 1 ComputeOp"
        linear_ops = [op for op in compute_ops if op.op_type == "linear"]
        assert len(linear_ops) == 1, f"Expected 1 linear ComputeOp, got {len(linear_ops)}"
        assert len(neural_cores) == 0, f"Identity FC should NOT create NeuralCores, got {len(neural_cores)}"


class TestReLUFCStaysNeuralCore:
    """FC layer with ReLU activation → NeuralCore (on chip)."""

    def test_relu_fc_creates_neural_core(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(),
                       base_activation_name="ReLU")
        fc = PerceptronMapper(flat, p)
        repr_ = ModelRepresentation(fc)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))
        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
        linear_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp) and n.op_type == "linear"]

        assert len(neural_cores) >= 1, "ReLU FC should create NeuralCores"
        assert len(linear_ops) == 0, "ReLU FC should NOT create linear ComputeOps"


class TestLinearComputeOpPreservesNegatives:
    """The linear ComputeOp must not clip negative values."""

    def test_negatives_preserved(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(),
                       base_activation_name="Identity")
        fc = PerceptronMapper(flat, p)
        repr_ = ModelRepresentation(fc)

        # Set weights to produce negative outputs
        with torch.no_grad():
            p.layer.weight.fill_(-0.5)
            p.layer.bias.fill_(0.0)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))

        # Execute via IR
        x = torch.ones(1, 16) * 0.5
        buffers = {}
        for node in ir_graph.nodes:
            buffers[node.id] = node.execute(x, buffers)

        final = buffers[ir_graph.nodes[-1].id]
        assert (final < 0).any(), (
            f"Linear ComputeOp should preserve negatives, got all non-negative: {final}"
        )


class TestIdentityConvBecomesComputeOp:
    """Conv2D with Identity activation → per-position linear ComputeOps."""

    def test_identity_conv_creates_compute_ops(self):
        inp = InputMapper((1, 4, 4))
        conv = Conv2DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="Identity",
        )
        flat = EinopsRearrangeMapper(conv, "... c h w -> ... (c h w)")
        p = Perceptron(4, 8, normalization=nn.Identity(),
                       base_activation_name="ReLU")
        fc = PerceptronMapper(flat, p)
        repr_ = ModelRepresentation(fc)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))
        linear_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp) and n.op_type == "linear"]
        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]

        # Conv with 4x4 input, 2x2 kernel, stride 2 → 2x2 output → 4 positions
        assert len(linear_ops) == 4, f"Expected 4 per-position linear ComputeOps, got {len(linear_ops)}"
        # FC with ReLU should still be NeuralCore
        assert len(neural_cores) >= 1, "ReLU FC should still be NeuralCore"


class TestMixedModelForwardEquivalence:
    """Model with both ReLU and Identity layers: IR forward matches model forward."""

    def test_forward_matches(self):
        torch.manual_seed(42)
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")

        # ReLU layer → NeuralCore
        p1 = Perceptron(8, 16, normalization=nn.Identity(),
                        base_activation_name="ReLU")
        fc1 = PerceptronMapper(flat, p1)

        # Identity layer → ComputeOp
        p2 = Perceptron(4, 8, normalization=nn.Identity(),
                        base_activation_name="Identity")
        fc2 = PerceptronMapper(fc1, p2)
        repr_ = ModelRepresentation(fc2)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))

        # Set threshold=1 for NeuralCores (no quantization)
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                node.threshold = 1.0
                node.parameter_scale = torch.tensor(1.0)

        x = torch.rand(4, 1, 4, 4)
        with torch.no_grad():
            model_out = repr_(x)

        # Execute IR graph
        x_flat = x.view(x.shape[0], -1)
        buffers = {}
        for node in ir_graph.nodes:
            buffers[node.id] = node.execute(x_flat, buffers)

        # Gather final output
        from mimarsinan.mapping.ir import IRSource
        ir_out = torch.zeros(4, len(ir_graph.output_sources.flatten()))
        for idx, src in enumerate(ir_graph.output_sources.flatten()):
            if isinstance(src, IRSource) and src.node_id >= 0:
                ir_out[:, idx] = buffers[src.node_id][:, src.index]

        max_diff = (model_out - ir_out).abs().max().item()
        assert max_diff < 1e-4, (
            f"IR forward should match model forward, max diff: {max_diff:.6f}"
        )
