"""End-to-end tests for the unified per-source-scale wrap rule.

``compute_per_source_scales`` must stamp ``per_source_scales`` /
``output_scale`` on a ``ComputeOpMapper`` iff its input scales are
heterogeneous — either across multiple sources or within one source's
channel vector.  The wrapper then materialises at IR emission time.
"""

from __future__ import annotations

import operator
from typing import Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.compute_modules import ComputeAdapter, ScaleNormalizingWrapper
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, EinopsRearrangeMapper, InputMapper
from mimarsinan.mapping.mappers.leading_dim import Ensure2DMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.mappers.scale_propagation import (
    _all_sources_uniform,
    _is_per_channel_heterogeneous,
    apply_compute_op_scale_policy as _apply_compute_op_scale_policy,
)
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class _StubMapper:
    """Stand-in for a mapper, just enough to plug into compute_per_source_scales."""
    def __init__(self):
        self.source_mapper = None

    def get_source_mappers(self):
        return []


def _make_compute_op_mapper(num_sources: int = 2) -> ComputeOpMapper:
    sources = [InputMapper((1,)) for _ in range(num_sources)]
    return ComputeOpMapper(sources, ComputeAdapter(operator.add))


class TestScaleHeterogeneityPredicates:
    def test_uniform_vector_is_not_heterogeneous(self):
        assert not _is_per_channel_heterogeneous(torch.full((8,), 2.0))

    def test_scalar_is_not_heterogeneous(self):
        assert not _is_per_channel_heterogeneous(torch.tensor([3.0]))

    def test_mixed_vector_is_heterogeneous(self):
        assert _is_per_channel_heterogeneous(torch.tensor([2.0, 4.0, 2.0, 4.0]))

    def test_all_sources_uniform_when_equal(self):
        s = torch.tensor([2.0, 2.0])
        assert _all_sources_uniform([s, s.clone(), s.clone()])

    def test_all_sources_uniform_false_when_differ(self):
        assert not _all_sources_uniform(
            [torch.tensor([2.0, 2.0]), torch.tensor([4.0, 4.0])]
        )


class TestPolicyMultiInput:
    def test_uniform_sources_no_wrap(self):
        node = _make_compute_op_mapper(num_sources=2)
        scales = [torch.tensor([2.0]), torch.tensor([2.0])]
        out = _apply_compute_op_scale_policy(node, scales)
        assert torch.allclose(out, torch.tensor([2.0]))
        assert node.per_source_scales is None
        assert node.output_scale is None

    def test_divergent_sources_trigger_wrap(self):
        node = _make_compute_op_mapper(num_sources=2)
        scales = [torch.tensor([2.0]), torch.tensor([4.0])]
        out = _apply_compute_op_scale_policy(node, scales)
        # mean policy: (2 + 4) / 2 = 3.0
        assert torch.allclose(out, torch.tensor([3.0]))
        assert node.per_source_scales is not None
        assert node.output_scale is not None
        assert torch.allclose(node.output_scale, torch.tensor([3.0]))


class TestPolicyUnaryHeterogeneous:
    def test_unary_uniform_no_wrap(self):
        node = _make_compute_op_mapper(num_sources=1)
        scales = [torch.full((8,), 2.0)]
        out = _apply_compute_op_scale_policy(node, scales)
        assert torch.allclose(out, scales[0])
        assert node.per_source_scales is None

    def test_unary_heterogeneous_triggers_wrap(self):
        node = _make_compute_op_mapper(num_sources=1)
        scales = [torch.tensor([2.0, 2.0, 4.0, 4.0])]
        out = _apply_compute_op_scale_policy(node, scales)
        # combine_source_scales over a single source = the source itself.
        assert torch.allclose(out, scales[0])
        assert node.per_source_scales is not None
        assert len(node.per_source_scales) == 1
        assert torch.allclose(node.per_source_scales[0], scales[0])


class TestEndToEndConcatPlusComputeOp:
    """``ConcatMapper`` propagates a per-channel scale vector; a downstream
    ``ComputeOpMapper`` consuming heterogeneous channels gets wrapped."""

    def test_layernorm_after_concat_of_diverging_scales_wraps(self):
        a_perc = Perceptron(2, 2, normalization=nn.Identity(), base_activation_name="ReLU")
        a_perc.activation_scale = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        b_perc = Perceptron(2, 2, normalization=nn.Identity(), base_activation_name="ReLU")
        b_perc.activation_scale = nn.Parameter(torch.tensor(4.0), requires_grad=False)

        inp = InputMapper((1, 4))
        flat = EinopsRearrangeMapper(inp, "... c f -> ... (c f)")
        flat_a = PerceptronMapper(Ensure2DMapper(flat), a_perc)
        flat_b = PerceptronMapper(Ensure2DMapper(flat), b_perc)

        cat = ConcatMapper([flat_a, flat_b], dim=1)
        layer_norm = ComputeOpMapper(cat, nn.LayerNorm([4]), name="ln")

        repr_ = ModelRepresentation(layer_norm)
        compute_per_source_scales(repr_)

        # LayerNorm sees a (2,2)-style heterogeneous concat → must be wrapped.
        assert layer_norm.per_source_scales is not None
        merged = layer_norm.per_source_scales[0]
        # Concatenation of [2,2] and [4,4] → [2,2,4,4]
        assert torch.allclose(merged, torch.tensor([2.0, 2.0, 4.0, 4.0]))

    def test_layernorm_after_concat_of_matching_scales_no_wrap(self):
        a_perc = Perceptron(2, 2, normalization=nn.Identity(), base_activation_name="ReLU")
        a_perc.activation_scale = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        b_perc = Perceptron(2, 2, normalization=nn.Identity(), base_activation_name="ReLU")
        b_perc.activation_scale = nn.Parameter(torch.tensor(3.0), requires_grad=False)

        inp = InputMapper((1, 4))
        flat = EinopsRearrangeMapper(inp, "... c f -> ... (c f)")
        flat_a = PerceptronMapper(Ensure2DMapper(flat), a_perc)
        flat_b = PerceptronMapper(Ensure2DMapper(flat), b_perc)

        cat = ConcatMapper([flat_a, flat_b], dim=1)
        layer_norm = ComputeOpMapper(cat, nn.LayerNorm([4]), name="ln")

        repr_ = ModelRepresentation(layer_norm)
        compute_per_source_scales(repr_)

        assert layer_norm.per_source_scales is None


class TestEndToEndIREmission:
    """When the policy stamps slots, IR emission must wrap the module."""

    def test_wrapper_appears_in_emitted_compute_op(self):
        a = InputMapper((4,))
        b = InputMapper((4,))
        mapper = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        mapper.per_source_scales = [
            torch.tensor([2.0, 2.0, 2.0, 2.0]),
            torch.tensor([4.0, 4.0, 4.0, 4.0]),
        ]
        mapper.output_scale = torch.tensor([3.0, 3.0, 3.0, 3.0])

        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        mapper.map_to_ir(ir)
        op = next(n for n in ir.nodes if isinstance(n, ComputeOp))
        assert isinstance(op.params["module"], ScaleNormalizingWrapper)

    def test_wrapper_math_matches_policy_intent(self):
        """For Add with two divergent scales, the wrapper output equals
        ``(s_a · a + s_b · b) / s_out`` channel-wise."""
        s_a = torch.tensor([2.0, 4.0])
        s_b = torch.tensor([6.0, 8.0])
        s_out = (s_a + s_b) / 2.0
        wrapper = ScaleNormalizingWrapper(
            ComputeAdapter(operator.add), [s_a, s_b], s_out,
        )

        a = torch.tensor([[0.5, 0.25]])
        b = torch.tensor([[0.3, 0.1]])
        out = wrapper(a, b)

        expected = (s_a * a + s_b * b) / s_out
        assert torch.allclose(out, expected, atol=1e-6)


class TestForwardScaleNormalized:
    """The NF-twin of the emitted wrapper: ``forward_scale_normalized`` must run
    the same composition the deployed ComputeOp runs, and stay the plain
    ``forward`` when no scales are armed (the byte-identical scalar path)."""

    def test_unarmed_mapper_is_identity_to_forward(self):
        mapper = _make_compute_op_mapper(num_sources=2)
        a = torch.tensor([[0.5, 0.25]])
        b = torch.tensor([[0.3, 0.1]])
        assert torch.equal(
            mapper.forward_scale_normalized((a, b)), mapper.forward((a, b))
        )

    def test_armed_mapper_matches_the_emitted_wrapper(self):
        mapper = _make_compute_op_mapper(num_sources=2)
        s_a = torch.tensor([2.0, 4.0])
        s_b = torch.tensor([6.0, 8.0])
        s_out = (s_a + s_b) / 2.0
        mapper.per_source_scales = [s_a, s_b]
        mapper.output_scale = s_out

        a = torch.tensor([[0.5, 0.25]])
        b = torch.tensor([[0.3, 0.1]])
        wrapper = ScaleNormalizingWrapper(mapper.module, [s_a, s_b], s_out)
        assert torch.allclose(
            mapper.forward_scale_normalized((a, b)), wrapper(a, b), atol=1e-6,
        )

    def test_armed_unary_mapper_decodes_and_renormalizes(self):
        source = InputMapper((1,))
        mapper = ComputeOpMapper(source, nn.Linear(2, 3))
        theta = torch.tensor([2.0, 4.0])
        mapper.per_source_scales = [theta]
        mapper.output_scale = theta

        wire = torch.tensor([[0.5, 0.25]])
        expected = ScaleNormalizingWrapper(mapper.module, [theta], theta)(wire)
        assert torch.allclose(
            mapper.forward_scale_normalized(wire), expected, atol=1e-6,
        )
