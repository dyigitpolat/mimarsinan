"""Perceptron packaging: activation detection determines NeuralCore vs ComputeOp.

The perceptron packaging rule: MM+ + BN? + ACT → perceptron → NeuralCore.
When no activation is detected (Identity), the layer is a linear compute op.
Any detected nonlinearity (ReLU, GELU, LeakyReLU, etc.) qualifies as a
perceptron — the adaptation pipeline converts all of them to LeakyGradReLU.

Mapper eligibility contract
-----------------------------
All mapper types (PerceptronMapper, Conv2DPerceptronMapper, Conv1DPerceptronMapper)
must expose the same semantics through ``owned_perceptron_groups()``:
  - Identity activation → returns [] (excluded from pipeline processing)
  - Any nonlinear activation (ReLU, LeakyReLU, GELU, …) → returns the perceptron
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.mappers.structural import (
    InputMapper, EinopsRearrangeMapper,
)
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper, Conv1DPerceptronMapper
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


class TestGELUFCBecomesNeuralCore:
    """FC layer with GELU activation → NeuralCore (any nonlinearity is a perceptron)."""

    def test_gelu_fc_creates_neural_core(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(),
                       base_activation_name="GELU")
        fc = PerceptronMapper(flat, p)
        repr_ = ModelRepresentation(fc)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))
        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
        linear_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp) and n.op_type == "linear"]

        assert len(neural_cores) >= 1, "GELU FC should create NeuralCores"
        assert len(linear_ops) == 0, "GELU FC should NOT create linear ComputeOps"


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


# ---------------------------------------------------------------------------
# Mapper eligibility contract tests
# ---------------------------------------------------------------------------

class TestPerceptronMapperEligibilityContract:
    """PerceptronMapper.owned_perceptron_groups() must apply is_perceptron_activation.

    Identity perceptrons are linear compute ops and must be excluded from the
    pipeline-visible perceptron list. All nonlinear activations are perceptrons.
    """

    def test_identity_fc_excluded_from_owned_groups(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(),
                       base_activation_name="Identity")
        mapper = PerceptronMapper(flat, p)
        assert mapper.owned_perceptron_groups() == [], (
            "PerceptronMapper with Identity activation must return [] "
            "from owned_perceptron_groups()."
        )

    @pytest.mark.parametrize("act_name", ["ReLU", "LeakyReLU", "GELU"])
    def test_nonlinear_fc_included_in_owned_groups(self, act_name):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(),
                       base_activation_name=act_name)
        mapper = PerceptronMapper(flat, p)
        groups = mapper.owned_perceptron_groups()
        assert len(groups) == 1 and p in groups[0], (
            f"PerceptronMapper with {act_name} activation must expose the perceptron."
        )


class TestConv2DMapperEligibilityContract:
    """Conv2DPerceptronMapper.owned_perceptron_groups() must mirror PerceptronMapper."""

    def test_identity_conv2d_excluded_from_owned_groups(self):
        inp = InputMapper((1, 4, 4))
        conv = Conv2DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=False, use_batchnorm=False,
            base_activation_name="Identity",
        )
        assert conv.owned_perceptron_groups() == [], (
            "Conv2DPerceptronMapper with Identity activation must return [] "
            "from owned_perceptron_groups()."
        )

    @pytest.mark.parametrize("act_name", ["ReLU", "LeakyReLU", "GELU"])
    def test_nonlinear_conv2d_included_in_owned_groups(self, act_name):
        inp = InputMapper((1, 4, 4))
        conv = Conv2DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=False, use_batchnorm=False,
            base_activation_name=act_name,
        )
        groups = conv.owned_perceptron_groups()
        assert len(groups) == 1 and conv.perceptron in groups[0], (
            f"Conv2DPerceptronMapper with {act_name} activation must expose the perceptron."
        )


class TestConv1DMapperEligibilityContract:
    """Conv1DPerceptronMapper.owned_perceptron_groups() must mirror PerceptronMapper."""

    def test_identity_conv1d_excluded_from_owned_groups(self):
        inp = InputMapper((1, 8))
        conv = Conv1DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=False, use_batchnorm=False,
            base_activation_name="Identity",
        )
        assert conv.owned_perceptron_groups() == [], (
            "Conv1DPerceptronMapper with Identity activation must return [] "
            "from owned_perceptron_groups()."
        )

    @pytest.mark.parametrize("act_name", ["ReLU", "LeakyReLU", "GELU"])
    def test_nonlinear_conv1d_included_in_owned_groups(self, act_name):
        inp = InputMapper((1, 8))
        conv = Conv1DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=False, use_batchnorm=False,
            base_activation_name=act_name,
        )
        groups = conv.owned_perceptron_groups()
        assert len(groups) == 1 and conv.perceptron in groups[0], (
            f"Conv1DPerceptronMapper with {act_name} activation must expose the perceptron."
        )


class TestGetPerceptronsExcludesIdentity:
    """ModelRepresentation.get_perceptrons() must return only perceptrons with
    nonlinear activations. Identity perceptrons are invisible to the pipeline.
    """

    def test_identity_fc_not_in_get_perceptrons(self):
        """FC Identity perceptron must not appear in get_perceptrons()."""
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p_id = Perceptron(8, 16, normalization=nn.Identity(),
                          base_activation_name="Identity")
        p_relu = Perceptron(4, 8, normalization=nn.Identity(),
                            base_activation_name="ReLU")
        m1 = PerceptronMapper(flat, p_id)
        m2 = PerceptronMapper(m1, p_relu)
        repr_ = ModelRepresentation(m2)

        perceptrons = repr_.get_perceptrons()
        assert p_id not in perceptrons, "Identity FC must be excluded from get_perceptrons()"
        assert p_relu in perceptrons, "ReLU FC must be included in get_perceptrons()"

    def test_identity_conv2d_not_in_get_perceptrons(self):
        """Conv2D Identity perceptron must not appear in get_perceptrons()."""
        inp = InputMapper((1, 4, 4))
        conv_id = Conv2DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=False, use_batchnorm=False,
            base_activation_name="Identity",
        )
        flat = EinopsRearrangeMapper(conv_id, "... c h w -> ... (c h w)")
        p_relu = Perceptron(4, 8, normalization=nn.Identity(),
                            base_activation_name="ReLU")
        fc = PerceptronMapper(flat, p_relu)
        repr_ = ModelRepresentation(fc)

        perceptrons = repr_.get_perceptrons()
        assert conv_id.perceptron not in perceptrons, (
            "Identity Conv2D perceptron must be excluded from get_perceptrons()"
        )
        assert p_relu in perceptrons, "ReLU FC must still be in get_perceptrons()"

    def test_torch_mapped_mixed_model_excludes_identity_conv(self):
        """Torch-mapped model with Conv(Identity)+FC(LeakyReLU): only FC is in get_perceptrons().

        This mirrors the MLP-Mixer deployment scenario where the patch-embedding
        Conv2d has no activation (→ Identity) but the FC mixing layers do.
        """
        import torch.nn as nn
        from mimarsinan.torch_mapping.converter import convert_torch_model

        class ConvIdentityFCLeakyReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, padding=0)
                self.bn = nn.BatchNorm2d(4)
                self.fc = nn.Linear(4 * 4 * 4, 8)
                self.act = nn.LeakyReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = x.flatten(1)
                x = self.act(self.fc(x))
                return x

        torch.manual_seed(0)
        model = ConvIdentityFCLeakyReLU()
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 1, 8, 8))

        supermodel = convert_torch_model(model, input_shape=(1, 8, 8), num_classes=8)
        perceptrons = supermodel.get_perceptrons()

        # The Conv2d has no activation → Identity → must NOT be in get_perceptrons()
        act_names = [type(p.base_activation).__name__ for p in perceptrons]
        assert "Identity" not in act_names, (
            f"Identity conv perceptron must not be in get_perceptrons(). "
            f"Found activations: {act_names}"
        )
        # The FC with LeakyReLU must be included
        assert len(perceptrons) >= 1, "At least one perceptron expected"
