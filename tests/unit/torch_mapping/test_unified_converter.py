"""Tests for the unified ``_emit_generic_compute_op`` path in MapperGraphConverter.

Verifies that the converter retired its per-op handlers (``_convert_add``,
``_convert_getitem``) in favour of a single FX-args partition + generic
``ComputeAdapter`` construction.
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.compute_modules import ComputeAdapter, ScaleNormalizingWrapper
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.mappers.perceptron import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model


def _flow_to_ir(flow, input_shape):
    from mimarsinan.mapping.ir_mapping import IRMapping
    from mimarsinan.mapping.per_source_scales import compute_per_source_scales
    mapper_repr = flow.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024,
    )
    return ir_mapping.map(mapper_repr), ir_mapping


class TestAddRoutesThroughGenericPath:
    def test_plus_operator_emits_compute_adapter(self):
        class _ResidualAdd(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.a(x)) + torch.relu(self.b(x))

        flow = convert_torch_model(_ResidualAdd(), (8,), num_classes=8, device="cpu")
        _, ir_mapping = _flow_to_ir(flow, (8,))
        adds = [
            n for n in ir_mapping.nodes
            if isinstance(n, ComputeOp)
            and isinstance(n.params.get("module"), ComputeAdapter)
            and n.params["module"].fn is operator.add
        ]
        assert len(adds) >= 1, "torch.add / + must route through ComputeAdapter(operator.add)"


class TestGetitemRoutesThroughGenericPath:
    def test_select_token_emits_compute_adapter(self):
        class _SelectFirstToken(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                # x: (B, 3, 4)  →  Linear over last dim, then x[:, 0]
                x = torch.relu(self.proj(x))
                return x[:, 0]

        flow = convert_torch_model(_SelectFirstToken(), (3, 4), num_classes=4, device="cpu")
        _, ir_mapping = _flow_to_ir(flow, (3, 4))
        getitems = [
            n for n in ir_mapping.nodes
            if isinstance(n, ComputeOp)
            and isinstance(n.params.get("module"), ComputeAdapter)
            and n.params["module"].fn is operator.getitem
        ]
        assert len(getitems) >= 1


class TestMeanRoutesThroughGenericPath:
    def test_tensor_mean_method_emits_torch_mean_adapter(self):
        class _MeanReduce(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                # x: (B, 6, 4) → linear → reduce over tokens dim → (B, 4)
                x = torch.relu(self.proj(x))
                return x.mean(dim=1)

        flow = convert_torch_model(_MeanReduce(), (6, 4), num_classes=4, device="cpu")
        _, ir_mapping = _flow_to_ir(flow, (6, 4))
        means = [
            n for n in ir_mapping.nodes
            if isinstance(n, ComputeOp)
            and isinstance(n.params.get("module"), ComputeAdapter)
            and n.params["module"].fn is torch.mean
        ]
        assert len(means) >= 1


class TestPartitionFxArgs:
    """Direct test of the FX-args classifier underlying the generic emit."""

    def _converter_with_graph(self, model, input_shape):
        from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
        from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
        from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityAnalyzer
        from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter

        gm = normalize_fx_graph(trace_model(model, input_shape, device="cpu"))
        report = RepresentabilityAnalyzer(gm).analyze()
        converter = MapperGraphConverter(gm, input_shape)
        # Run convert to populate _node_to_mapper / _node_to_attr.
        converter.convert(report)
        return converter, gm

    def test_sources_collected_for_binary_add(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.a(x)) + torch.relu(self.b(x))

        converter, gm = self._converter_with_graph(_Model(), (8,))
        add_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.add
        )
        sources, bound, extra, kwargs = converter._partition_fx_args(add_node)
        assert len(sources) == 2
        assert bound == []
        assert extra == ()
        assert kwargs == {}

    def test_constant_get_attr_goes_to_bound(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.const = nn.Parameter(torch.zeros(8), requires_grad=False)
                self.proj = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.proj(x)) + self.const

        converter, gm = self._converter_with_graph(_Model(), (8,))
        add_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.add
        )
        sources, bound, extra, kwargs = converter._partition_fx_args(add_node)
        assert len(sources) == 1
        assert len(bound) == 1
        assert isinstance(bound[0], (nn.Parameter, torch.Tensor))


class TestStructuralShortcutsStillFire:
    """Cat / flatten / view / permute keep their structural shortcuts."""

    def test_concat_of_pure_sources_emits_no_compute_op(self):
        class _DualBranchCat(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Linear(4, 4)

            def forward(self, x):
                return torch.cat([torch.relu(self.a(x)), torch.relu(self.b(x))], dim=1)

        flow = convert_torch_model(_DualBranchCat(), (4,), num_classes=8, device="cpu")
        ir_graph, _ = _flow_to_ir(flow, (4,))
        # No ComputeOp from cat — pure source-array concat folded at mapping time.
        cat_compute_ops = [
            n for n in ir_graph.nodes
            if isinstance(n, ComputeOp)
            and isinstance(n.params.get("module"), ComputeAdapter)
            and n.params["module"].fn is torch.cat
        ]
        assert len(cat_compute_ops) == 0


class TestPickleSurvivesConversion:
    def test_emitted_ir_pickles_round_trip(self):
        import pickle

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, x):
                return torch.relu(self.lin(x)).mean(dim=0, keepdim=True)

        flow = convert_torch_model(_Model(), (4,), num_classes=4, device="cpu")
        ir_graph, _ = _flow_to_ir(flow, (4,))
        ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp)]
        for op in ops:
            loaded = pickle.loads(pickle.dumps(op))
            assert isinstance(loaded.params["module"], type(op.params["module"]))
