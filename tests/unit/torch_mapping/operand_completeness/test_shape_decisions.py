"""A batch decision is refused where it reaches a per-sample shape the program reads, not where it reaches a layout
(review round 8).

Round 7 refused every layout a batch decision reached through the data flow. ``PickFlattenStart``,
``torch.flatten(z.view(B, 6 // g, g), 1)`` with ``g = half_if_big(B)``, gives the same values for either ``g`` and was
exact on w0-base; it was refused by name. Now a decision's scalar results are fresh symbols in the symbolic pass, so the
pass computes which per-sample shapes a decision reaches (``BatchLayouts.shape_decision``): a value it only scales is
not reached, and a pure reshape that reads only its input's per-sample element count (``view``, ``reshape``, a flatten
of every per-sample axis) or axes (``squeeze``, ``unsqueeze``, a partial flatten) restores a per-sample shape the
decision does not reach. A layout whose per-sample shape is reached is kept where every value it reaches is first
restored; one that reaches any other operator (a softmax, a permute, a host op) or the output is refused by name,
naming the decision (``LayoutMixin._require_restored``).

[restated in review round 9: the helper ``half_if_big`` makes a program impure (``symbolic/purity.py``), so the pass
never runs for it and every layout is the host op of its native method: each helper form is exact at every batch, the
refused ones included, and ``PickFlattenStart``'s structure is host ops where it was w0-base's mappers (identical
values). The rules above apply to pure programs: each form is pinned with the declared gate ``6 - 3 * (B > 7)``, the
helper's values, in place of the helper.]
"""

from __future__ import annotations

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.converter_handlers.symbolic.shape_decisions import reshape_kind
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .harness import CONTRACTS, assert_converted_exact, assert_impure_exact, convert, native

EVERY_BATCH = (1, 2, 3, 6, 7, 9, 13)


@torch.fx.wrap
def half_if_big(n):
    return 3 if n > 7 else 6


class Decided(nn.Module):
    """Layouts a helper of the batch size decides, and what reads them; ``gate``: the declared gate in its place."""

    FORMS = {
        # restored before anything reads them: exact
        "flatten_start": lambda m, x, y, g: torch.flatten(F.softmax(y.view(x.size(0), 2, 3), -1).view(
            x.size(0), 6 // g, g), 1),
        "batch_extent": lambda m, x, y, g: F.softmax(_by_batch_extent(y.view(x.size(0), 6 // g, g)), -1),
        "reshape_chain": lambda m, x, y, g: F.softmax(y.reshape(x.size(0), g, -1).reshape(x.size(0), -1), -1),
        "scaled_view": lambda m, x, y, g: F.softmax((y * g).view(x.size(0), 2, 3), -1).flatten(1),
        # read before a restoring reshape: refused by name
        "softmax_view": lambda m, x, y, g: F.softmax(y.view(x.size(0), g, -1), -1).flatten(1),
        "permute_view": lambda m, x, y, g: y.view(x.size(0), g, -1).permute(0, 2, 1).flatten(1),
        "output_view": lambda m, x, y, g: y.view(x.size(0), g, -1),
        "partial_flatten": lambda m, x, y, g: F.softmax(y.view(x.size(0), 1, g, -1).flatten(2), -1).flatten(1),
        "shape_read": lambda m, x, y, g: y.view(x.size(0), g, -1).flatten(1) * y.view(x.size(0), g, -1).shape[1],
    }

    def __init__(self, how: str, gate: bool = False):
        super().__init__()
        self.how = how
        self.gate = gate
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        g = 6 - 3 * (x.size(0) > 7) if self.gate else half_if_big(x.size(0))
        return self.FORMS[self.how](self, x, self.fc(x), g)


def _by_batch_extent(z):
    return z.view(z.size(0), -1)


# The w0-base mapper structure of PickFlattenStart under both contracts (``fix_r8/out/structure_pins_base.txt``; the
# layer ``fc`` is a perceptron under the spiking contract, a host op under the MVM one).
W0_BASE_STRUCTURE = ["InputMapper", "Ensure2DMapper", "fc", "ReshapeMapper", "softmax", "ReshapeMapper",
                     "ReshapeMapper"]


def _structure(flow) -> list:
    rows = [m.name if isinstance(m, ComputeOpMapper) else type(m).__name__
            for m in flow.get_mapper_repr().execution_order()]
    return ["fc" if row == "PerceptronMapper" else row for row in rows]


# [restated in review round 14: the gate ``6 - 3 * (B > 7)`` is an opaque value; the view it reaches is the host op of
# its native method, and the flatten that restores ``(6,)`` stays static]
GATED_STRUCTURE = ["InputMapper", "Ensure2DMapper", "fc", "ReshapeMapper", "softmax", "size_2", "size", "gt", "mul",
                   "sub", "floordiv", "view_1", "ReshapeMapper"]


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_value_neutral_reshape_chain_a_gate_decides_runs_the_gated_view_on_the_host(contract):
    """``PickFlattenStart``: exact on w0-base, refused by name on the round-7 tree, w0-base's mapper structure in rounds
    8 to 13; exact at every batch now, no literal, the gated view the host op of its native method and the restoring
    flatten a static ``ReshapeMapper`` of ``(6,)`` (identical values)."""
    model = native(lambda: Decided("flatten_start", gate=True))
    flow = assert_converted_exact(model, (8,), contract, batches=EVERY_BATCH, kinds=("normal", "uniform"))
    assert _structure(flow) == GATED_STRUCTURE and flow.shape_literals == {}
    shapes = [m.output_shape for m in flow.get_mapper_repr().execution_order() if type(m).__name__ == "ReshapeMapper"]
    assert shapes == [(2, 3), (6,)]


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_value_neutral_reshape_chain_a_helper_decides_is_its_native_host_ops(contract):
    """Review round 9, declared change: ``PickFlattenStart`` itself is impure; its layouts are the host ops of their
    native methods (round 8: w0-base's ``ReshapeMapper`` structure), identical values at every batch."""
    flow = assert_impure_exact(native(lambda: Decided("flatten_start")), (8,), contract, "half_if_big", "half_if_big")
    assert _structure(flow) == ["InputMapper", "Ensure2DMapper", "fc", "size_1", "view", "softmax", "size_2", "size",
                                "half_if_big", "floordiv", "view_1", "flatten"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("gate", [True, False], ids=["gate", "helper"])
@pytest.mark.parametrize("how", ["batch_extent", "reshape_chain", "scaled_view"])
def test_a_layout_a_decision_reaches_and_a_reshape_restores_is_exact(how, gate, contract):
    """Refused by name on the round-7 tree (the decision reached the layout); exact on w0-base."""
    assert_converted_exact(native(lambda: Decided(how, gate)), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


READ_BEFORE_RESTORED = ["softmax_view", "permute_view", "output_view", "partial_flatten", "shape_read"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", READ_BEFORE_RESTORED)
def test_a_decided_per_sample_shape_read_before_it_is_restored_runs_on_the_host(how, contract):
    """Refused by name in rounds 8 to 13, naming the gate's guard and the operator it reached. [restated in review
    round 14: the gate is an opaque value, so every layout it reaches is the host op of its native method, and every
    reader here is a host op or the output: exact at every batch]"""
    flow = assert_converted_exact(native(lambda: Decided(how, gate=True)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_opaque[:2] == ("gt", "gt")


class StaticReader(nn.Module):
    """A gated per-sample shape read by a Linear (a perceptron fixes its per-sample input at conversion); natively the
    Linear's fan-in no longer matches from batch 8 on."""

    def __init__(self):
        super().__init__()
        self.fc, self.head = nn.Linear(8, 6), nn.Linear(2, 2)

    def forward(self, x):
        g = 2 + (x.size(0) > 7)
        return self.head(self.fc(x).view(x.size(0), 6 // g, g)).flatten(1)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_static_layout_an_opaque_value_reaches_is_refused_by_name(contract):
    """The owner's rule: a static per-sample layout an opaque value reaches is refused by name, naming the value."""
    with pytest.raises(BatchAxisLayoutError) as excinfo:
        convert(native(StaticReader), (8,), contract)
    message = str(excinfo.value)
    assert message.startswith("view (FX node 'view', used by 'head')")
    assert "through gt (FX node 'gt'), a value outside the affine batch domain" in message
    assert "it reaches Linear (FX node 'head'), which fixes its per-sample shape at conversion" in message


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(READ_BEFORE_RESTORED))
def test_a_per_sample_shape_a_helper_decides_is_exact_however_it_is_read(how, contract):
    """Review round 9: refused by name in round 8 (naming ``half_if_big``); impure now, exact at every batch."""
    assert_impure_exact(native(lambda: Decided(how)), (8,), contract, "half_if_big", "half_if_big")


def test_the_pass_restores_the_flatten_and_keeps_the_view_decided():
    """[restated in review round 9: on the declared gate; the helper program has no pass; review round 14: the
    decision is the gate's comparison, an opaque value]"""
    gm = trace_model(native(lambda: Decided("flatten_start", gate=True)), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    decision = proof.shape_decision(nodes["view_1"])
    assert decision is not None and decision.node is nodes["gt"]
    assert proof.shape_decision(nodes["flatten"]) is None and proof.shape_decision(nodes["softmax"]) is None
    assert proof.shapes[nodes["flatten"]].dims[1] == 6  # the restored value is the traced one, decision-free
    assert nodes["flatten"] in proof.unproved and layouts.layout(nodes["flatten"]) is None  # computed from it


class TestReshapeKind:
    @staticmethod
    def _node(op, target, *args, meta_rank=3, **kwargs):
        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta["tensor_meta"] = type("Meta", (), {"shape": (1,) * meta_rank})()
        return graph.create_node(op, target, (x, *args), kwargs)

    @pytest.mark.parametrize("op,target,args,kind", [
        ("call_method", "view", (1, -1), "count"), ("call_method", "reshape", (1, 6), "count"),
        ("call_method", "flatten", (1,), "count"), ("call_method", "flatten", (1, -1), "count"),
        ("call_method", "flatten", (2,), "axes"), ("call_method", "flatten", (), "axes"),
        ("call_method", "squeeze", (1,), "axes"), ("call_method", "unsqueeze", (1,), "axes"),
        ("call_function", torch.flatten, (1,), "count"), ("call_function", torch.reshape, ((1, -1),), "count"),
        ("call_method", "permute", (0, 2, 1), None), ("call_method", "expand", (-1, 2, 3), None),
        ("call_function", torch.transpose, (1, 2), None), ("call_function", F.softmax, (-1,), None),
    ])
    def test_pure_reshapes_and_what_they_read(self, op, target, args, kind):
        assert reshape_kind(self._node(op, target, *args), {}) == kind

    def test_a_flatten_module_of_every_per_sample_axis_reads_the_count(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta["tensor_meta"] = type("Meta", (), {"shape": (1, 2, 3)})()
        node = graph.create_node("call_module", "f", (x,), {})
        assert reshape_kind(node, {"f": nn.Flatten(1)}) == "count"
        assert reshape_kind(node, {"f": nn.Flatten(2)}) == "axes"
