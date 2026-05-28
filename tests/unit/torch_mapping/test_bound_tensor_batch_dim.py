"""Bound tensors stored via ``_partition_fx_args`` must be batch-stripped."""

from __future__ import annotations

import operator

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
)
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model


class _AddLeadingOneParam(nn.Module):
    """Mirrors the ViT positional-embedding shape pattern: param has shape ``(1, N, D)``."""

    def __init__(self, n: int = 6, d: int = 4):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.pos = nn.Parameter(torch.randn(1, n, d), requires_grad=False)

    def forward(self, x):  # x: (B, N, D)
        return torch.relu(self.proj(x)) + self.pos


class TestPartitionFxArgsStripsLeadingSingleton:
    def test_param_with_leading_one_is_squeezed_before_storage(self):
        model = _AddLeadingOneParam(n=6, d=4)
        gm = normalize_fx_graph(trace_model(model, (6, 4), device="cpu"))
        report = RepresentabilityAnalyzer(gm).analyze()
        converter = MapperGraphConverter(gm, (6, 4))
        converter.convert(report)
        add_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.add
        )
        _, bound, _, _ = converter._partition_fx_args(add_node)
        assert len(bound) == 1
        assert bound[0].shape == (6, 4), (
            f"bound pos param must be batch-stripped to (N, D); got {tuple(bound[0].shape)}"
        )

    def test_forward_matches_native_for_leading_one_param(self):
        torch.manual_seed(0)
        model = _AddLeadingOneParam(n=6, d=4).eval()
        flow = convert_torch_model(model, (6, 4), num_classes=4, device="cpu")
        x = torch.randn(3, 6, 4)
        with torch.no_grad():
            native = model(x)
            converted = flow(x)
        assert converted.shape == native.shape, (
            f"shape diverged: native={tuple(native.shape)} converted={tuple(converted.shape)}"
        )
        assert torch.allclose(converted, native, atol=1e-5), (
            f"value diverged: max diff {(converted - native).abs().max().item():.2e}"
        )


class TestPlainParamUnchanged:
    """1-D parameters (no leading singleton) must pass through unchanged."""

    def test_one_d_param_not_modified(self):
        class _OneD(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.bias_extra = nn.Parameter(torch.zeros(4), requires_grad=False)

            def forward(self, x):
                return torch.relu(self.proj(x)) + self.bias_extra

        gm = normalize_fx_graph(trace_model(_OneD(), (4,), device="cpu"))
        report = RepresentabilityAnalyzer(gm).analyze()
        converter = MapperGraphConverter(gm, (4,))
        converter.convert(report)
        add_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.add
        )
        _, bound, _, _ = converter._partition_fx_args(add_node)
        assert len(bound) == 1
        assert bound[0].shape == (4,), (
            f"1-D param must not be modified; got {tuple(bound[0].shape)}"
        )
