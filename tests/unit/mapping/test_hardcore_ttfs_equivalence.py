"""Hard-core TTFS equivalence: SpikingHybridCoreFlow must match SpikingUnifiedCoreFlow.

Tests that the hard-core mapping pipeline (segment splitting, compaction, HardCore
packing) produces the same TTFS continuous outputs as the soft-core unified path.

The chip only supports ReLU activation. All activations must be adapted to
ReLU-compatible forms (via Clamp Adaptation) before reaching deployment.
These tests use ReLU activations throughout to match deployed behavior.

NOTE: SpikingHybridCoreFlow._forward_ttfs multiplies output by T (simulation_length)
while SpikingUnifiedCoreFlow does not. Tests normalize by dividing hard_out by T.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.mappers.structural import (
    InputMapper, PermuteMapper, MeanMapper, EinopsRearrangeMapper,
)
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_latency import IRLatency
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow


CORES_CONFIG = [{"count": 512, "max_axons": 256, "max_neurons": 256}]
SIM_LENGTH = 32


def _build_ir_and_flows(mapper_repr, input_shape):
    """Build IR graph, soft-core flow, and hard-core flow from a ModelRepresentation."""
    compute_per_source_scales(mapper_repr)

    ir_mapping = IRMapping(
        q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
    )
    ir_graph = ir_mapping.map(mapper_repr)

    # No weight quantization: threshold=1.0, parameter_scale=1.0
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.threshold = 1.0
            node.parameter_scale = torch.tensor(1.0)

    # Assign latencies (required for correct hard-core packing)
    IRLatency(ir_graph).calculate()

    soft_flow = SpikingUnifiedCoreFlow(
        input_shape, ir_graph, SIM_LENGTH, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs",
    )
    soft_flow.eval()

    hybrid_mapping = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=CORES_CONFIG,
    )

    hard_flow = SpikingHybridCoreFlow(
        input_shape, hybrid_mapping, SIM_LENGTH, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs",
    )
    hard_flow.eval()

    return soft_flow, hard_flow, ir_graph


class TestHardCoreTTFSSimpleLinear:
    """Two-layer MLP: flatten → Linear+ReLU → Linear+ReLU (chip-compatible activations)."""

    def test_hard_matches_soft(self):
        torch.manual_seed(42)
        input_shape = (1, 4, 4)

        p1 = Perceptron(8, 16, normalization=nn.Identity(),
                        base_activation_name="ReLU")
        p2 = Perceptron(4, 8, normalization=nn.Identity(),
                        base_activation_name="ReLU")

        inp = InputMapper(input_shape)
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        m1 = PerceptronMapper(flat, p1)
        m2 = PerceptronMapper(m1, p2)
        repr_ = ModelRepresentation(m2)

        soft_flow, hard_flow, _ = _build_ir_and_flows(repr_, input_shape)

        x = torch.rand(8, *input_shape)
        with torch.no_grad():
            soft_out = soft_flow(x)
            hard_out = hard_flow(x) / SIM_LENGTH

        max_diff = (soft_out - hard_out).abs().max().item()
        assert max_diff < 1e-4, (
            f"Hard-core vs soft-core max diff {max_diff:.6f}. "
            f"soft sample: {soft_out[0]}, hard sample: {hard_out[0]}"
        )


class TestHardCoreTTFSConvPlusFC:
    """Conv2D (ReLU) → flatten → FC (ReLU), no ComputeOps."""

    def test_conv_relu_hard_matches_soft(self):
        torch.manual_seed(42)
        input_shape = (1, 4, 4)

        conv = Conv2DPerceptronMapper(
            InputMapper(input_shape),
            in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="ReLU",
        )
        flat = EinopsRearrangeMapper(conv, "... c h w -> ... (c h w)")
        p_out = Perceptron(4, 8, normalization=nn.Identity(),
                           base_activation_name="ReLU")
        fc = PerceptronMapper(flat, p_out)
        repr_ = ModelRepresentation(fc)

        soft_flow, hard_flow, _ = _build_ir_and_flows(repr_, input_shape)

        x = torch.rand(8, *input_shape)
        with torch.no_grad():
            soft_out = soft_flow(x)
            hard_out = hard_flow(x) / SIM_LENGTH

        max_diff = (soft_out - hard_out).abs().max().item()
        assert max_diff < 1e-4, (
            f"Conv+FC hard-core vs soft-core max diff {max_diff:.6f}"
        )


class TestHardCoreTTFSMiniMixer:
    """Mini mixer with ComputeOp: Conv → permute → FC → mean → classifier.

    This is the critical test: the mean ComputeOp splits cores into 2 segments.
    All activations use ReLU (chip-compatible).
    """

    def test_mini_mixer_hard_matches_soft(self):
        torch.manual_seed(42)
        input_shape = (1, 4, 4)
        num_patches = 4
        patch_dim = 2

        conv = Conv2DPerceptronMapper(
            InputMapper(input_shape),
            in_channels=1, out_channels=patch_dim,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="ReLU",
        )
        flat = EinopsRearrangeMapper(conv, "... c h w -> ... c (h w)")
        perm1 = PermuteMapper(flat, (0, 2, 1))

        # Token mixing
        perm2 = PermuteMapper(perm1, (0, 2, 1))
        p_tok = Perceptron(num_patches, num_patches, normalization=nn.Identity(),
                           base_activation_name="ReLU")
        fc_tok = PerceptronMapper(perm2, p_tok)
        perm3 = PermuteMapper(fc_tok, (0, 2, 1))

        # Mean pool over patches
        mean = MeanMapper(perm3, dim=1)

        # Classifier (ReLU — adapted from Identity via Clamp Adaptation)
        p_cls = Perceptron(3, patch_dim, normalization=nn.Identity(),
                           base_activation_name="ReLU")
        classifier = PerceptronMapper(mean, p_cls)
        repr_ = ModelRepresentation(classifier)

        soft_flow, hard_flow, ir_graph = _build_ir_and_flows(repr_, input_shape)

        # Verify we actually have 2 segments (ComputeOp splits the graph)
        hybrid_mapping = hard_flow.hybrid_mapping
        neural_segs = hybrid_mapping.get_neural_segments()
        compute_ops = hybrid_mapping.get_compute_ops()
        assert len(neural_segs) == 2, f"Expected 2 neural segments, got {len(neural_segs)}"
        assert len(compute_ops) == 1, f"Expected 1 compute op, got {len(compute_ops)}"

        x = torch.rand(16, *input_shape)
        with torch.no_grad():
            soft_out = soft_flow(x)
            hard_out = hard_flow(x) / SIM_LENGTH

        max_diff = (soft_out - hard_out).abs().max().item()
        assert max_diff < 1e-4, (
            f"Mini-mixer hard-core vs soft-core max diff {max_diff:.6f}. "
            f"soft sample: {soft_out[0]}, hard sample: {hard_out[0]}"
        )

    def test_mini_mixer_argmax_agreement(self):
        """Hard-core argmax must agree with soft-core argmax."""
        torch.manual_seed(42)
        input_shape = (1, 4, 4)
        num_patches = 4
        patch_dim = 2

        conv = Conv2DPerceptronMapper(
            InputMapper(input_shape),
            in_channels=1, out_channels=patch_dim,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="ReLU",
        )
        flat = EinopsRearrangeMapper(conv, "... c h w -> ... c (h w)")
        perm1 = PermuteMapper(flat, (0, 2, 1))
        mean = MeanMapper(perm1, dim=1)
        p_cls = Perceptron(3, patch_dim, normalization=nn.Identity(),
                           base_activation_name="ReLU")
        classifier = PerceptronMapper(mean, p_cls)
        repr_ = ModelRepresentation(classifier)

        soft_flow, hard_flow, _ = _build_ir_and_flows(repr_, input_shape)

        x = torch.rand(32, *input_shape)
        with torch.no_grad():
            soft_out = soft_flow(x)
            hard_out = hard_flow(x)

        soft_pred = soft_out.argmax(dim=1)
        hard_pred = hard_out.argmax(dim=1)
        agreement = (soft_pred == hard_pred).float().mean().item()
        assert agreement == 1.0, (
            f"Hard vs soft argmax agreement {agreement:.0%}. "
            f"soft sample: {soft_out[0]}, hard sample: {hard_out[0]}"
        )


class TestHardCoreTTFSFullMixer:
    """Full TorchMLPMixer converted through the torch_mapping pipeline."""

    def test_full_mixer_hard_matches_soft(self):
        from mimarsinan.models.torch_mlp_mixer import TorchMLPMixer
        from mimarsinan.torch_mapping.converter import convert_torch_model

        torch.manual_seed(42)
        input_shape = (1, 8, 8)
        num_classes = 4

        model = TorchMLPMixer(
            input_shape=input_shape,
            num_classes=num_classes,
            patch_n_1=2,
            patch_m_1=2,
            patch_c_1=4,
            fc_w_1=4,
            fc_w_2=4,
            base_activation="ReLU",
        )
        model.eval()

        supermodel = convert_torch_model(model, input_shape, num_classes, device="cpu", Tq=4)

        # Fuse BN
        from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
        pt = PerceptronTransformer()
        for p in supermodel.get_perceptrons():
            if isinstance(p.normalization, nn.Identity):
                continue
            u, beta, mean = pt._get_u_beta_mean(p.normalization)
            W = p.layer.weight.data
            b = p.layer.bias.data if p.layer.bias is not None else torch.zeros(W.shape[0], device=W.device)
            fused_W = W * u.unsqueeze(-1)
            fused_b = (b - mean) * u + beta
            p.layer = nn.Linear(p.input_features, p.output_channels, bias=True)
            p.layer.weight.data = fused_W
            p.layer.bias.data = fused_b
            p.normalization = nn.Identity()

        supermodel.eval()
        mapper_repr = supermodel.get_mapper_repr()
        compute_per_source_scales(mapper_repr)

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        ir_graph = ir_mapping.map(mapper_repr)

        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                node.threshold = 1.0
                node.parameter_scale = torch.tensor(1.0)
                # Simulate post-adaptation state: all activations become ReLU-
                # compatible after Clamp Adaptation runs in the real pipeline.
                node.activation_type = "LeakyGradReLU"

        IRLatency(ir_graph).calculate()

        soft_flow = SpikingUnifiedCoreFlow(
            input_shape, ir_graph, SIM_LENGTH, nn.Identity(),
            "TTFS", "TTFS", "<=", spiking_mode="ttfs",
        )
        soft_flow.eval()

        hybrid_mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph,
            cores_config=CORES_CONFIG,
        )
        hard_flow = SpikingHybridCoreFlow(
            input_shape, hybrid_mapping, SIM_LENGTH, nn.Identity(),
            "TTFS", "TTFS", "<=", spiking_mode="ttfs",
        )
        hard_flow.eval()

        x = torch.rand(8, *input_shape)
        with torch.no_grad():
            soft_out = soft_flow(x)
            hard_out = hard_flow(x) / SIM_LENGTH

        max_diff = (soft_out - hard_out).abs().max().item()
        assert max_diff < 1e-3, (
            f"Full mixer hard-core vs soft-core max diff {max_diff:.6f}. "
            f"soft sample: {soft_out[0]}, hard sample: {hard_out[0]}"
        )

        # Argmax must agree
        soft_pred = soft_out.argmax(dim=1)
        hard_pred = hard_out.argmax(dim=1)
        agreement = (soft_pred == hard_pred).float().mean().item()
        assert agreement == 1.0, (
            f"Full mixer hard vs soft argmax agreement {agreement:.0%}"
        )
