"""V6: per-node decisions are polymorphic ``Mapper`` methods, not isinstance chains.

These lock the dispatch/precedence rules of the three migrated call-sites
(``compute_per_source_scales``, ``propagate_boundary_input_scales``, the softcore
flowchart) at the mapper-method seam, and prove the open-closed property: a NEW
mapper kind gets correct behavior in ALL THREE sites by inheriting the base
defaults — zero edits at any call-site.
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.flowchart import FlowchartFCSpec, FlowchartNodeEstimate
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import (
    ConcatMapper,
    InputMapper,
    StackMapper,
)
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


def _perc(out_c, in_f, scale):
    p = Perceptron(out_c, in_f, normalization=nn.Identity(), base_activation_name="ReLU")
    p.activation_scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)
    return p


class TestSourceScaleDispatch:
    def test_input_mapper_emits_unit_vector(self):
        node = InputMapper((5,))
        out = node.propagate_source_scale([], {})
        assert torch.allclose(out, torch.ones(5))

    def test_perceptron_emits_activation_scale_vector(self):
        node = PerceptronMapper(InputMapper((4,)), _perc(3, 4, 2.0))
        out = node.propagate_source_scale([], {})
        assert torch.allclose(out, torch.full((3,), 2.0))

    def test_concat_concatenates_present_sources(self):
        a, b = InputMapper((2,)), InputMapper((2,))
        node = ConcatMapper([a, b])
        out_scales = {a: torch.tensor([2.0, 2.0]), b: torch.tensor([4.0, 4.0])}
        out = node.propagate_source_scale([a, b], out_scales)
        assert torch.allclose(out, torch.tensor([2.0, 2.0, 4.0, 4.0]))

    def test_concat_with_no_present_sources_is_none(self):
        a = InputMapper((2,))
        assert ConcatMapper([a]).propagate_source_scale([a], {}) is None

    def test_compute_op_uniform_returns_first_without_wrap(self):
        a, b = InputMapper((1,)), InputMapper((1,))
        node = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        out_scales = {a: torch.tensor([2.0]), b: torch.tensor([2.0])}
        out = node.propagate_source_scale([a, b], out_scales)
        assert torch.allclose(out, torch.tensor([2.0]))
        assert node.per_source_scales is None and node.output_scale is None

    def test_compute_op_divergent_triggers_wrap(self):
        a, b = InputMapper((1,)), InputMapper((1,))
        node = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        out_scales = {a: torch.tensor([2.0]), b: torch.tensor([4.0])}
        out = node.propagate_source_scale([a, b], out_scales)
        assert torch.allclose(out, torch.tensor([3.0]))
        assert node.per_source_scales is not None
        assert node.output_scale is not None

    def test_transparent_default_passes_first_source_through(self):
        a = InputMapper((2,))
        node = StackMapper([a])  # no source-scale override
        out_scales = {a: torch.tensor([7.0, 7.0])}
        out = node.propagate_source_scale([a], out_scales)
        assert torch.allclose(out, torch.tensor([7.0, 7.0]))


class TestBoundaryScaleDispatch:
    def test_input_mapper_returns_default(self):
        assert InputMapper((5,)).propagate_boundary_scale([], {}, 0.5) == 0.5

    def test_perceptron_sets_input_scale_and_returns_theta_out(self):
        p = _perc(3, 4, 2.0)
        node = PerceptronMapper(InputMapper((4,)), p)
        a = InputMapper((4,))
        out = node.propagate_boundary_scale([a], {a: 0.75}, 1.0)
        assert out == 2.0
        assert float(p.input_activation_scale) == 0.75

    def test_perceptron_input_scale_defaults_when_no_source(self):
        p = _perc(3, 4, 2.0)
        node = PerceptronMapper(InputMapper((4,)), p)
        node.propagate_boundary_scale([], {}, 0.3)
        assert float(p.input_activation_scale) == pytest.approx(0.3)

    def test_compute_op_means_present_sources(self):
        a, b = InputMapper((1,)), InputMapper((1,))
        node = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        out = node.propagate_boundary_scale([a, b], {a: 2.0, b: 4.0}, 1.0)
        assert out == 3.0

    def test_transparent_default_means_present_or_none(self):
        a = InputMapper((1,))
        node = StackMapper([a])
        assert node.propagate_boundary_scale([a], {a: 5.0}, 1.0) == 5.0
        assert node.propagate_boundary_scale([], {}, 1.0) is None


class TestFlowchartDispatch:
    def test_perceptron_fc_spec(self):
        node = PerceptronMapper(InputMapper((4,)), _perc(3, 4, 2.0))
        est = node.flowchart_node_estimate(out_shape=(3,))
        assert est.fc_spec == FlowchartFCSpec(4, 3, 1, has_bias=True)
        assert "in_features=4, out_features=3" in est.sw_text

    def test_conv2d_instances_from_out_shape(self):
        c = Conv2DPerceptronMapper(
            InputMapper((3, 8, 8)), in_channels=3, out_channels=4,
            kernel_size=3, padding=1, use_batchnorm=False,
        )
        est = c.flowchart_node_estimate(out_shape=(4, 8, 8))
        assert est.fc_spec == FlowchartFCSpec(27, 4, 64, has_bias=True)

    def test_conv1d_instances_from_out_shape(self):
        c = Conv1DPerceptronMapper(
            InputMapper((3, 8)), in_channels=3, out_channels=4,
            kernel_size=3, padding=1, use_batchnorm=False,
        )
        est = c.flowchart_node_estimate(out_shape=(4, 8))
        assert est.fc_spec == FlowchartFCSpec(9, 4, 8, has_bias=True)

    def test_add_compute_op_estimates_linear_op(self):
        a, b = InputMapper((4,)), InputMapper((4,))
        node = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        est = node.flowchart_node_estimate(out_shape=(4,))
        assert est.fc_spec == FlowchartFCSpec(8, 4, 1, has_bias=False)
        assert est.sw_text == "SW perceptrons=0 (Add op)"

    def test_non_add_compute_op_has_no_fc_spec(self):
        node = ComputeOpMapper(InputMapper((4,)), nn.LayerNorm([4]))
        est = node.flowchart_node_estimate(out_shape=(4,))
        assert est.fc_spec is None

    def test_add_compute_op_without_unit_out_shape_no_spec(self):
        a, b = InputMapper((4,)), InputMapper((4,))
        node = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        est = node.flowchart_node_estimate(out_shape=(2, 4))
        assert est.fc_spec is None
        assert est.sw_text == "SW: n/a"

    def test_stack_is_host_side(self):
        est = StackMapper([InputMapper((4,))]).flowchart_node_estimate(out_shape=(4,))
        assert est.fc_spec is None
        assert est.sw_text == "SW stack (host-side)"

    def test_input_mapper_default(self):
        est = InputMapper((4,)).flowchart_node_estimate(out_shape=(4,))
        assert est == FlowchartNodeEstimate()


class _NovelMapper(Mapper):
    """A brand-new mapper kind that overrides NOTHING of the V6 methods."""

    def __init__(self, source):
        super().__init__(source)

    def _map_to_ir(self, ir_mapping):  # pragma: no cover - not exercised
        return self.source_mapper.map_to_ir(ir_mapping)

    def _forward_impl(self, x):  # pragma: no cover - not exercised
        return x


class TestOpenClosedExtensibility:
    """A new mapper kind inherits sane defaults in all three sites — no edits."""

    def test_source_scale_passes_through(self):
        a = InputMapper((2,))
        node = _NovelMapper(a)
        assert torch.allclose(
            node.propagate_source_scale([a], {a: torch.tensor([3.0, 3.0])}),
            torch.tensor([3.0, 3.0]),
        )

    def test_boundary_scale_means_sources(self):
        a, b = InputMapper((1,)), InputMapper((1,))
        node = _NovelMapper(a)
        assert node.propagate_boundary_scale([a, b], {a: 2.0, b: 6.0}, 1.0) == 4.0

    def test_flowchart_default_no_estimate(self):
        node = _NovelMapper(InputMapper((4,)))
        assert node.flowchart_node_estimate(out_shape=(4,)) == FlowchartNodeEstimate()
