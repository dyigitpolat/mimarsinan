"""Generic ops whose traced order is the canonical layout keep the pre-template adapter, byte for byte.

The oracle is the pre-template partition rule (sources, then leading-singleton-
stripped bound tensors, then literal arguments, keyword literals): every witness
here converted exactly before operand-order templates existed, so its adapter must
pickle to the same bytes (tensors by dtype, shape and raw bytes, because
``torch.save`` embeds a random serialization id).
"""

from __future__ import annotations

import io
import operator
import pickle

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityAnalyzer
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

C, N, H, W = 6, 5, 4, 3


def _tensor_marker(*_):
    raise RuntimeError("canonical bytes only")


class _CanonicalPickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().contiguous()
            raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
            return (_tensor_marker, (type(obj).__name__, str(t.dtype), tuple(t.shape),
                                     bool(obj.requires_grad), raw))
        return NotImplemented


def _canonical_bytes(obj) -> bytes:
    buffer = io.BytesIO()
    _CanonicalPickler(buffer, protocol=4).dump(obj)
    return buffer.getvalue()


def _legacy_adapter(converter, node, fn) -> ComputeAdapter:
    """The adapter the pre-template partition built for ``node``."""
    bound, extra = [], []
    for arg in node.args:
        if isinstance(arg, fx.Node):
            const = converter._get_constant_tensor(arg)
            if const is not None:
                bound.append(const.squeeze(0) if const.dim() >= 1 and const.shape[0] == 1 else const)
            continue
        extra.append(arg)
    kwargs = {k: v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)}
    return ComputeAdapter(fn, bound_tensors=bound, extra_args=tuple(extra), kwargs=kwargs)


class _Front(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.pre = nn.Conv2d(C, C, 1) if rank == 4 else nn.Linear(C, C)

    def forward(self, x):
        return self.pre(x)


_SHAPES = {2: (C,), 3: (N, C), 4: (C, H, W)}


class _Op(nn.Module):
    def __init__(self, rank, body, **params):
        super().__init__()
        self.front = _Front(rank)
        self.other = _Front(rank)
        self.body = body
        for name, shape in params.items():
            self.register_parameter(name, nn.Parameter(torch.randn(shape)))

    def forward(self, x):
        return self.body(self, self.front(x), x)


_WITNESSES = {
    "x_add_c": (2, lambda m, y, x: y + m.c, {"c": (C,)}),
    "c_add_x": (2, lambda m, y, x: m.c + y, {"c": (C,)}),
    "x_mul_c": (2, lambda m, y, x: y * m.c, {"c": (C,)}),
    "c_mul_x": (2, lambda m, y, x: m.c * y, {"c": (C,)}),
    "torch_add_c_x": (2, lambda m, y, x: torch.add(m.c, y), {"c": (C,)}),
    "torch_mul_c_x": (2, lambda m, y, x: torch.mul(m.c, y), {"c": (C,)}),
    "literal_mul_left": (3, lambda m, y, x: 2.0 * y, {}),
    "literal_add_left": (3, lambda m, y, x: 2.0 + y, {}),
    "literal_mul_right": (3, lambda m, y, x: y * 2.0, {}),
    "x_sub_c": (2, lambda m, y, x: y - m.c, {"c": (C,)}),
    "x_div_c": (2, lambda m, y, x: y / m.c, {"c": (C,)}),
    "sub_alpha_x_c": (2, lambda m, y, x: torch.sub(y, m.c, alpha=2), {"c": (C,)}),
    "pos_embedding_right": (3, lambda m, y, x: y + m.pos, {"pos": (1, N, C)}),
    "pos_embedding_left": (3, lambda m, y, x: m.pos + y, {"pos": (1, N, C)}),
    "layerscale_c11": (4, lambda m, y, x: m.g * y, {"g": (C, 1, 1)}),
    "layerscale_1c11": (4, lambda m, y, x: m.g * y, {"g": (1, C, 1, 1)}),
    "resmlp_affine_published": (3, lambda m, y, x: m.a * y + m.b, {"a": (1, 1, C), "b": (1, 1, C)}),
    "two_source_add": (3, lambda m, y, x: y + m.other(x), {}),
    "two_source_mul": (3, lambda m, y, x: y * m.other(x), {}),
    "mean_over_space": (4, lambda m, y, x: torch.mean(y, (2, 3)), {}),
    "relu": (3, lambda m, y, x: torch.relu(y), {}),
    "slice_getitem": (3, lambda m, y, x: y[:, 0], {}),
}


def _convert_graph(name):
    rank, body, params = _WITNESSES[name]
    torch.manual_seed(0)
    model = _Op(rank, body, **params).eval()
    gm = normalize_fx_graph(trace_model(model, _SHAPES[rank], device="cpu"))
    report = RepresentabilityAnalyzer(gm).analyze()
    converter = MapperGraphConverter(gm, _SHAPES[rank])
    converter.convert(report)
    return gm, converter


_GENERIC_TARGETS = {
    operator.add, operator.mul, operator.sub, operator.truediv, operator.getitem,
    torch.add, torch.mul, torch.sub, torch.mean, torch.relu,
}


@pytest.mark.parametrize("name", sorted(_WITNESSES))
def test_canonical_generic_op_keeps_the_legacy_adapter_bytes(name):
    gm, converter = _convert_graph(name)
    generic_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target in _GENERIC_TARGETS
    ]
    assert generic_nodes, f"{name}: no generic op traced"
    for node in generic_nodes:
        mapper = converter._node_to_mapper[node]
        adapter = mapper.module
        assert isinstance(adapter, ComputeAdapter)
        assert "arg_template" not in adapter.__dict__, f"{name}/{node.name}: carries a template"
        legacy = _legacy_adapter(converter, node, node.target)
        assert _canonical_bytes(adapter) == _canonical_bytes(legacy), (
            f"{name}/{node.name}: adapter state differs from the pre-template layout"
        )


def test_the_batch_contract_check_leaves_the_global_rng_untouched():
    gm, converter = _convert_graph("layerscale_1c11")
    node = next(n for n in gm.graph.nodes if n.op == "call_function" and n.target is operator.mul)
    partition = converter._partition_fx_args(node, operator.mul)
    state = torch.get_rng_state()
    adapter = converter._batch_contract_adapter(
        node, operator.mul, partition, [arg for _, arg in partition.sources],
    )
    assert torch.equal(torch.get_rng_state(), state)
    assert "arg_template" not in adapter.__dict__
