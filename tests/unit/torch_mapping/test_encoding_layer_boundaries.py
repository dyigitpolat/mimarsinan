"""Tests for ``mark_encoding_layers`` boundary detection.

A perceptron is marked as an encoding layer iff its upstream chain hits either
the raw network input or a ``ComputeOpMapper`` wrapping a bare Linear / Conv
(signed, unbounded output).  Bounded-output ComputeOps (LayerNorm, GELU, ...)
and upstream perceptrons are transparent.
"""

from __future__ import annotations

import operator

import torch
import torch.nn as nn

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import (
    EinopsRearrangeMapper,
    InputMapper,
)
from mimarsinan.mapping.mappers.leading_dim import Ensure2DMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import (
    _is_encoding_segment_start,
    _wraps_unbounded_raw_linear_or_conv,
    mark_encoding_layers,
)


def _perc(out_features, in_features, *, name="p"):
    return Perceptron(
        out_features, in_features,
        normalization=nn.Identity(),
        base_activation_name="ReLU",
        name=name,
    )


def _flatten_path(input_shape):
    inp = InputMapper(input_shape)
    flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
    return Ensure2DMapper(flat)


class TestUnboundedRawDetector:
    def test_bare_linear_is_unbounded(self):
        mapper = ComputeOpMapper(InputMapper((4,)), nn.Linear(4, 4))
        assert _wraps_unbounded_raw_linear_or_conv(mapper)

    def test_bare_conv2d_is_unbounded(self):
        mapper = ComputeOpMapper(InputMapper((3, 8, 8)), nn.Conv2d(3, 4, 3, padding=1))
        assert _wraps_unbounded_raw_linear_or_conv(mapper)

    def test_sequential_starting_with_linear_is_unbounded(self):
        seq = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        mapper = ComputeOpMapper(InputMapper((4,)), seq)
        assert _wraps_unbounded_raw_linear_or_conv(mapper)

    def test_layernorm_is_bounded(self):
        mapper = ComputeOpMapper(InputMapper((4,)), nn.LayerNorm([4]))
        assert not _wraps_unbounded_raw_linear_or_conv(mapper)

    def test_gelu_is_bounded(self):
        mapper = ComputeOpMapper(InputMapper((4,)), nn.GELU())
        assert not _wraps_unbounded_raw_linear_or_conv(mapper)

    def test_compute_adapter_is_bounded(self):
        adapter = ComputeAdapter(torch.mean, kwargs={"dim": 0})
        mapper = ComputeOpMapper(InputMapper((4,)), adapter)
        assert not _wraps_unbounded_raw_linear_or_conv(mapper)


class TestEncodingSegmentStart:
    def test_first_perceptron_from_raw_input_is_encoding(self):
        source = _flatten_path((1, 4, 4))
        p_mapper = PerceptronMapper(source, _perc(8, 16))
        assert _is_encoding_segment_start(p_mapper)

    def test_perceptron_after_perceptron_is_not_encoding(self):
        source = _flatten_path((1, 4, 4))
        p1 = PerceptronMapper(source, _perc(8, 16, name="p1"))
        p2 = PerceptronMapper(p1, _perc(4, 8, name="p2"))
        assert _is_encoding_segment_start(p1)
        assert not _is_encoding_segment_start(p2)

    def test_perceptron_after_layernorm_transparent(self):
        source = _flatten_path((1, 4, 4))
        p1 = PerceptronMapper(source, _perc(8, 16, name="p1"))
        ln = ComputeOpMapper(p1, nn.LayerNorm([8]))
        p2 = PerceptronMapper(Ensure2DMapper(ln), _perc(4, 8, name="p2"))
        # LayerNorm is bounded → walk continues past it → hits p1 → not encoding.
        assert not _is_encoding_segment_start(p2)

    def test_perceptron_after_bare_linear_is_encoding(self):
        source = _flatten_path((1, 4, 4))
        raw_linear = ComputeOpMapper(source, nn.Linear(16, 8))
        p2 = PerceptronMapper(Ensure2DMapper(raw_linear), _perc(4, 8))
        # Raw Linear produces unbounded signed output → p2 must encode.
        assert _is_encoding_segment_start(p2)

    def test_perceptron_after_compute_adapter_transparent(self):
        source = _flatten_path((1, 4, 4))
        p1 = PerceptronMapper(source, _perc(8, 16, name="p1"))
        adapter_op = ComputeOpMapper(
            p1, ComputeAdapter(torch.mean, kwargs={"dim": 0}),
        )
        p2 = PerceptronMapper(Ensure2DMapper(adapter_op), _perc(4, 8, name="p2"))
        # ComputeAdapter is bounded by convention → walk past it → hits p1 → not encoding.
        assert not _is_encoding_segment_start(p2)


class TestMarkEncodingLayers:
    def test_marks_only_segment_starters(self):
        source = _flatten_path((1, 4, 4))
        p1 = PerceptronMapper(source, _perc(8, 16, name="p1"))
        p2 = PerceptronMapper(p1, _perc(4, 8, name="p2"))
        repr_ = ModelRepresentation(p2)
        mark_encoding_layers(repr_)
        assert p1.perceptron.is_encoding_layer is True
        # p2 is not encoding because its upstream is another perceptron.
        assert getattr(p2.perceptron, "is_encoding_layer", False) is False

    def test_idempotent(self):
        source = _flatten_path((1, 4, 4))
        p = PerceptronMapper(source, _perc(8, 16))
        repr_ = ModelRepresentation(p)
        mark_encoding_layers(repr_)
        mark_encoding_layers(repr_)
        assert p.perceptron.is_encoding_layer is True
