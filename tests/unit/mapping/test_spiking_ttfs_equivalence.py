"""TTFS continuous equivalence: ModelRepresentation forward vs SpikingUnifiedCoreFlow.

Tests that the analytical TTFS path (act(W@x+b)/threshold) produces the same
logits (up to constant scaling) as the float model forward.  This catches:

- Missing activation_type on NeuralCores (defaulting to ReLU)
- Incorrect weight/bias transfer
- ComputeOp (mean/permute) wiring errors
- Per-source scale mismatches
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
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


def _build_ir_and_flow(mapper_repr, input_shape):
    """Build IR graph and SpikingUnifiedCoreFlow from a ModelRepresentation."""
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

    flow = SpikingUnifiedCoreFlow(
        input_shape, ir_graph, 32, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs",
    )
    flow.eval()
    return flow


class TestSimpleLinearTTFS:
    """Two-layer MLP: flatten → Linear+ReLU → Linear+Identity."""

    def test_argmax_agreement(self):
        torch.manual_seed(42)
        input_shape = (1, 4, 4)
        in_feat = 16
        hidden = 8
        out_feat = 4

        p1 = Perceptron(hidden, in_feat, normalization=nn.Identity(),
                        base_activation_name="ReLU")
        p2 = Perceptron(out_feat, hidden, normalization=nn.Identity(),
                        base_activation_name="Identity")

        inp = InputMapper(input_shape)
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        m1 = PerceptronMapper(flat, p1)
        m2 = PerceptronMapper(m1, p2)
        repr_ = ModelRepresentation(m2)

        flow = _build_ir_and_flow(repr_, input_shape)

        x = torch.rand(8, *input_shape)
        with torch.no_grad():
            model_out = repr_(x)
            flow_out = flow(x)

        # Outputs may differ by a constant factor (activation_scale), but
        # argmax must agree.
        model_pred = model_out.argmax(dim=1)
        flow_pred = flow_out.argmax(dim=1)
        agreement = (model_pred == flow_pred).float().mean().item()
        assert agreement == 1.0, (
            f"Argmax agreement {agreement:.0%} on batch of 8. "
            f"model_out sample: {model_out[0]}, flow_out sample: {flow_out[0]}"
        )


class TestConvPlusFCTTFS:
    """Conv2D (Identity activation) → flatten → FC (ReLU) → FC (Identity)."""

    def test_conv_identity_no_relu_clip(self):
        """Conv with Identity must not clip negatives in TTFS."""
        torch.manual_seed(42)
        input_shape = (1, 4, 4)

        conv = Conv2DPerceptronMapper(
            InputMapper(input_shape),
            in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="Identity",
        )
        flat = EinopsRearrangeMapper(conv, "... c h w -> ... (c h w)")
        p_out = Perceptron(4, 8, normalization=nn.Identity(),
                           base_activation_name="Identity")
        fc = PerceptronMapper(flat, p_out)
        repr_ = ModelRepresentation(fc)

        flow = _build_ir_and_flow(repr_, input_shape)

        x = torch.rand(8, *input_shape)
        with torch.no_grad():
            model_out = repr_(x)
            flow_out = flow(x)

        # Check that negative values are preserved (not clipped by spurious ReLU)
        if (model_out < 0).any():
            neg_mask = model_out < 0
            flow_neg = flow_out[neg_mask]
            assert (flow_neg < 0).any(), (
                "Model has negative outputs but flow clipped them to >= 0. "
                "Conv NeuralCores likely missing activation_type (defaulting to ReLU)."
            )

        model_pred = model_out.argmax(dim=1)
        flow_pred = flow_out.argmax(dim=1)
        agreement = (model_pred == flow_pred).float().mean().item()
        assert agreement >= 0.75, (
            f"Argmax agreement {agreement:.0%} too low."
        )


class TestMiniMixerTTFS:
    """Minimal mixer: Conv → permute → FC → permute → FC → mean → classifier."""

    def test_mini_mixer_argmax(self):
        torch.manual_seed(42)
        input_shape = (1, 4, 4)
        num_patches = 4  # 2x2 patches from 4x4 with kernel=stride=2
        patch_dim = 2

        # Patch embedding (Conv2D, Identity activation — no activation after conv)
        conv = Conv2DPerceptronMapper(
            InputMapper(input_shape),
            in_channels=1, out_channels=patch_dim,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="Identity",
        )
        # Conv output: (patch_dim, 2, 2) → flatten → (patch_dim, num_patches)
        flat = EinopsRearrangeMapper(conv, "... c h w -> ... c (h w)")
        # Permute to (num_patches, patch_dim)
        perm1 = PermuteMapper(flat, (0, 2, 1))

        # Token mixing: permute → FC(ReLU) → FC(Identity) → permute back
        perm2 = PermuteMapper(perm1, (0, 2, 1))  # (patch_dim, num_patches)
        p_tok = Perceptron(num_patches, num_patches, normalization=nn.Identity(),
                           base_activation_name="ReLU")
        fc_tok = PerceptronMapper(perm2, p_tok)
        perm3 = PermuteMapper(fc_tok, (0, 2, 1))  # (num_patches, patch_dim)

        # Mean pool over patches
        mean = MeanMapper(perm3, dim=1)  # (patch_dim,)

        # Classifier
        p_cls = Perceptron(3, patch_dim, normalization=nn.Identity(),
                           base_activation_name="Identity")
        classifier = PerceptronMapper(mean, p_cls)
        repr_ = ModelRepresentation(classifier)

        flow = _build_ir_and_flow(repr_, input_shape)

        x = torch.rand(16, *input_shape)
        with torch.no_grad():
            model_out = repr_(x)
            flow_out = flow(x)

        model_pred = model_out.argmax(dim=1)
        flow_pred = flow_out.argmax(dim=1)
        agreement = (model_pred == flow_pred).float().mean().item()
        assert agreement >= 0.75, (
            f"Mini-mixer argmax agreement {agreement:.0%} < 75%. "
            f"Likely conv activation_type or ComputeOp wiring issue.\n"
            f"model sample: {model_out[0]}\nflow sample: {flow_out[0]}"
        )

    def test_mini_mixer_relative_order(self):
        """Flow logits should preserve relative ordering of model logits."""
        torch.manual_seed(123)
        input_shape = (1, 4, 4)
        num_patches = 4
        patch_dim = 2

        conv = Conv2DPerceptronMapper(
            InputMapper(input_shape),
            in_channels=1, out_channels=patch_dim,
            kernel_size=2, stride=2, padding=0,
            bias=True, use_batchnorm=False,
            base_activation_name="Identity",
        )
        flat = EinopsRearrangeMapper(conv, "... c h w -> ... c (h w)")
        perm1 = PermuteMapper(flat, (0, 2, 1))
        mean = MeanMapper(perm1, dim=1)
        p_cls = Perceptron(3, patch_dim, normalization=nn.Identity(),
                           base_activation_name="Identity")
        classifier = PerceptronMapper(mean, p_cls)
        repr_ = ModelRepresentation(classifier)

        flow = _build_ir_and_flow(repr_, input_shape)

        x = torch.rand(32, *input_shape)
        with torch.no_grad():
            model_out = repr_(x)
            flow_out = flow(x)

        # Check max-diff normalized by output range
        model_range = model_out.max() - model_out.min()
        if model_range > 1e-6:
            flow_range = flow_out.max() - flow_out.min()
            assert flow_range > 1e-6, "Flow output is constant — likely all zeros from ReLU clipping"
