"""Which values are batch-free (``converter_handlers/symbolic/batch_free.py``): no placeholder upstream, every call
upstream a declared operator that neither writes in place nor draws a random number, and no write, ordering edge or
``read_after`` of the program touching the computation; computed on the traced graph."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from batch_free.models import (
    SHAPE,
    CoordinateBias,
    InputDtypeFactory,
    OpaqueCone,
    OpaqueConsumer,
    RandomCone,
    RelativeBias,
    StateCopyToOpaqueCall,
    StateViewToLaterOpaqueCall,
    StateViewToOpaqueCall,
    WrittenByOpaqueCall,
    WrittenState,
)
from operand_completeness.harness import native


def _batch_free(model, shape=SHAPE):
    from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_free import batch_free_nodes

    gm = trace_model(model, shape)
    return gm, batch_free_nodes(gm)


def _ops(nodes, op=None, target=None):
    return [n for n in nodes if (op is None or n.op == op) and (target is None or n.target == target)]


def test_relative_bias_is_batch_free():
    gm, free = _batch_free(native(RelativeBias))
    for method in ("view", "permute", "contiguous", "unsqueeze"):
        assert _ops(free, "call_method", method), (method, sorted(n.name for n in free))
    assert any(n.op == "call_function" and n.name.startswith("getitem") for n in free)
    assert not _ops(free, "get_attr") and not _ops(free, "placeholder")
    add = next(n for n in gm.graph.nodes if n.op == "call_function" and getattr(n.target, "__name__", "") == "add")
    assert add not in free  # it reads the input


def test_module_calls_on_a_buffer_are_batch_free():
    gm, free = _batch_free(native(CoordinateBias))
    called = {str(n.target) for n in free if n.op == "call_module"}
    assert {"cpb.0", "cpb.1", "cpb.2"} <= called, called
    assert any(getattr(n.target, "__name__", "") == "sigmoid" for n in free)


def test_value_read_by_an_opaque_call_is_batch_free():
    gm, free = _batch_free(native(OpaqueConsumer))
    assert _ops(free, "call_method", "permute")
    opaque = next(n for n in gm.graph.nodes if getattr(n.target, "__name__", "") == "_opaque_attention")
    assert opaque not in free


def _upstream_of_permute(gm, free):
    permute = next(n for n in gm.graph.nodes if n.op == "call_method" and n.target == "permute")
    return permute, permute in free


@pytest.mark.parametrize("factory", [RandomCone, WrittenState, OpaqueCone, WrittenByOpaqueCall,
                                     StateViewToOpaqueCall, StateViewToLaterOpaqueCall], ids=lambda f: f.__name__)
def test_excluded_computations(factory):
    """A random draw, a written state, an opaque call, a value an opaque call writes and a later node reads, a view of
    the state an opaque call reads: the bias's permute is not batch-free."""
    gm, free = _batch_free(native(factory))
    permute, is_free = _upstream_of_permute(gm, free)
    assert not is_free, sorted(n.name for n in free)


def test_random_draw_is_not_batch_free():
    gm, free = _batch_free(native(RandomCone))
    assert not any(getattr(n.target, "__name__", "") == "randn" for n in free)


def test_value_read_from_an_input_is_not_batch_free():
    """``x.new_ones(x.size(-1))`` reads the input's extent. [changed in coverage W8: the witness was
    ``x.new_ones(T)``, which reads the input's dtype and device only, a metadata read and batch-free now]"""

    class InputSizeFactory(torch.nn.Module):
        def forward(self, x):
            return x + x.new_ones(x.size(-1))

    gm, free = _batch_free(native(InputSizeFactory))
    assert not any(n.op == "call_method" and n.target == "new_ones" for n in free)
    gm, free = _batch_free(native(InputDtypeFactory))
    assert [n.target for n in free if n.op == "call_method"] == ["new_ones"]


def test_a_write_of_a_view_of_the_state_taints_the_state():
    """A view of a buffer written in place: the write reaches the buffer, so nothing read from it is batch-free."""

    class ViewWritten(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.randn(4, 3))

        def forward(self, x):
            scaled = self.table * 3.0
            self.table.view(-1)[:1].mul_(1.0)
            return x + scaled.sum()

    gm, free = _batch_free(native(ViewWritten), (3,))
    assert not any(getattr(n.target, "__name__", "") == "mul" for n in free), sorted(n.name for n in free)


def test_a_copy_of_the_state_read_by_an_opaque_call_keeps_the_state():
    """A ``clone`` of the table given to an opaque call shares no storage with it (the tracer marks nothing), so the
    bias gathered from the table stays batch-free."""
    gm, free = _batch_free(native(StateCopyToOpaqueCall))
    permute, is_free = _upstream_of_permute(gm, free)
    assert is_free, sorted(n.name for n in free)
