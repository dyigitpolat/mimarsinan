"""The rules of operand writes (review round 15, ``symbolic/operand_writes``): which operands a call writes as torch
declares them, the run check of the tracer's shape propagation, and dead-code elimination of a write. The witnesses
are in ``test_operand_writes.py``."""

from __future__ import annotations

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import (
    OperandWriteError,
    WriteRecorder,
    check_record,
    declared_writes,
)
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .harness import native
from .test_operand_writes import ResizeAsResize, ResNetStyleInplace, _Fc


class _Unrewritten(_Fc):
    def forward(self, x):
        y = self.fc(x)
        torch.clamp_(y, -1.0, 1.0)
        return y * 2


def test_a_stale_read_is_refused_by_the_run_check():
    """The backstop: a graph whose write was not rewritten (FX's own trace) reads ``y`` after it changed in place."""
    gm = fx.symbolic_trace(native(_Unrewritten))
    recorder = WriteRecorder(gm)
    recorder.propagate(torch.randn(2, 8, dtype=torch.float64))
    assert [(r.name, v.name) for r, v in recorder.record.stale] == [("mul", "fc")]
    with pytest.raises(OperandWriteError, match="mul \\(FX node 'mul'\\): it reads 'fc'"):
        check_record(gm, [], recorder.record)


def test_dead_code_elimination_keeps_an_observable_write():
    """FX's own rule drops ``torch.resize_as_(z, y)`` (no aten schema in its impurity rule): normalization keeps a
    write whose operand a later node reads."""
    gm = fx.symbolic_trace(native(ResizeAsResize))
    normalize_fx_graph(gm)
    assert any(node.target is torch.resize_as_ for node in gm.graph.nodes)


def _call_on(target, *operands, **named):
    """A graph ``y -> z = y * 1 -> target(...)``, the operands named ``"y"`` and ``"z"`` replaced by those nodes."""
    graph = fx.Graph()
    y = graph.placeholder("y")
    z = graph.call_function(torch.mul, (y, 1))
    nodes = {"y": y, "z": z}

    def node(value):
        return nodes.get(value, value) if isinstance(value, str) else value

    call = graph.call_function(target, tuple(map(node, operands)), {k: node(v) for k, v in named.items()})
    return call, z


@pytest.mark.parametrize("target, operands, named, written", [
    (torch.resize_as_, ("z", "y"), {}, True), (torch.clamp_, ("z", 0, 1), {}, True),
    (torch.ops.aten.set_.source_Tensor, ("z", "y"), {}, True), (F.relu, ("z",), {"inplace": True}, True),
    (F.relu, ("z", True), {}, True), (torch.relu, ("z",), {}, False), (torch.sum, ("z",), {}, False),
    (torch.add, ("y", "z"), {"out": "z"}, True), (torch.add, ("y", "z"), {}, False),
], ids=["resize_as_", "clamp_", "aten.set_", "relu-inplace-keyword", "relu-inplace-positional", "relu", "sum",
        "add-out", "add"])
def test_declared_writes_follow_torchs_schemas(target, operands, named, written):
    """A write is torch's schema annotation ``Tensor(a!)`` (an ``out=`` argument where the call passes it), or
    ``inplace=True``; each returns its written operand."""
    call, z = _call_on(target, *operands, **named)
    assert [(w.operand, w.returned) for w in declared_writes(call, {})] == ([(z, True)] if written else [])


def test_an_in_place_module_writes_its_input():
    gm = trace_model(native(ResNetStyleInplace), (8,))
    act = next(node for node in gm.graph.nodes if node.name == "act")
    assert [(w.operand.name, w.returned) for w in declared_writes(act, dict(gm.named_modules()))] == [("fc", True)]


class SortedPlus(_Fc):
    """``y.sort(0)[0] + y`` (the round-4 reviewer's ``layout_fuzz.py`` form): ``aten::sort`` has TorchScript overloads
    that sort a list in place (``sort.Tensor(Tensor[](a!) self)``), which are no tensor's write."""

    def forward(self, x):
        y = self.fc(x)
        return y.sort(0)[0] + y


def test_a_list_overload_of_a_tensor_method_is_no_write():
    gm = trace_model(native(SortedPlus), (8,))
    sort = next(node for node in gm.graph.nodes if node.target == "sort")
    assert declared_writes(sort, dict(gm.named_modules())) == []
