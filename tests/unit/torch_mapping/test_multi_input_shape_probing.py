"""Multi-input ComputeOpMapper emission must record per-source input shapes."""

from __future__ import annotations

import operator

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
)
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model


def _convert(model: nn.Module, input_shape):
    gm = normalize_fx_graph(trace_model(model, input_shape, device="cpu"))
    report = RepresentabilityAnalyzer(gm).analyze()
    converter = MapperGraphConverter(gm, input_shape)
    converter.convert(report)
    return converter, gm


class TestBinaryAddRecordsBothInputShapes:
    """Today: `_get_input_shape` only probes `args[0]`, dropping `args[1]`."""

    def test_add_with_same_shape_records_two_input_shapes(self):
        class _ResidualAdd(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.a(x)) + torch.relu(self.b(x))

        converter, gm = _convert(_ResidualAdd(), (8,))
        add_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.add
        )
        mapper = converter._node_to_mapper[add_node]
        assert isinstance(mapper, ComputeOpMapper)
        assert len(mapper.sources) == 2
        assert mapper.input_shapes is not None
        assert len(mapper.input_shapes) == 2, (
            "binary add must record one input shape per source"
        )
        assert mapper.input_shapes[0] == (8,)
        assert mapper.input_shapes[1] == (8,)


class TestPartitionedShapesMatchSourceOrder:
    def test_distinct_upstream_shapes_preserved(self):
        class _AddDifferentRanks(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8)

            def forward(self, x):
                # x: (B, 8)  -> reshape to (B, 1, 8) and add (B, 1, 8)
                a = torch.relu(self.proj(x))  # (B, 8) ... wait, x is (B,4)
                a2 = a.unsqueeze(1)  # (B, 1, 8)
                b2 = a.unsqueeze(1)  # (B, 1, 8) - same source via duplication
                return a2 + b2

        converter, gm = _convert(_AddDifferentRanks(), (4,))
        add_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.add
        )
        mapper = converter._node_to_mapper[add_node]
        assert isinstance(mapper, ComputeOpMapper)
        assert mapper.input_shapes is not None
        assert len(mapper.input_shapes) == 2
        for shape in mapper.input_shapes:
            assert shape == (1, 8)
