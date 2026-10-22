"""Literal operands of declared calls, and literal containers FX re-types (review round 8).

A declared call runs on symbolic values only when every literal operand has a declared literal type
(``DECLARED_LITERAL_TYPES`` by exact type, in plain containers or FX's immutable ones). Before, ``opaque_reason`` read
the call target and callable arguments only: ``mul(size, Odd2(1))``, with an ``int`` subclass whose ``__rmul__``
answers a ``SymInt``, ran user code on the batch symbol, the pass proved ``zeros(b * k - b + 6)`` as ``(6,)`` with no
guard request, and ``t.shape[0]`` folded to ``6``: certified under both contracts and wrong by rel 0.25 at batches 9
and 13 (refused by name on w0-base, at its forward probe). Such a call is opaque now: a batch decision, never run on
symbolic values and never folded; its users are run-time host ops.

FX records a literal container's subclass (a ``tuple`` with its own ``__getitem__``) as the plain container, so the
traced graph calls ``tuple.__getitem__``: ``TableFold`` was certified and wrong by rel 0.5 at batches 9 and 13 (on
w0-base refused only through an unrelated host-op crash). The tracer refuses such a literal by name (``TracingError``),
and ``is_static_tree`` reads exact container types.

[restated in review round 9: a call with a foreign literal makes the program impure (``symbolic/purity.py``): the pass
never runs for it, and every layout is the host op of its native method, so ``Odd2View``, refused by name in round 8,
is exact at every batch; the declared literal types are the round-9 closed list, which holds no ``complex``.]
"""

from __future__ import annotations

import collections
import typing
from operator import getitem as _getitem

import numpy as np
import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.immutable_collections import immutable_dict, immutable_list

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.shape_value_mixin import is_static_tree
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import opaque_reason
from mimarsinan.torch_mapping.torch_graph_tracer import TracingError, trace_model

from .harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_impure_exact,
    assert_impure_stage_2_refused,
    convert,
    native,
)

EVERY_BATCH = (1, 2, 3, 6, 7, 9, 13)
torch.fx.wrap("_getitem")


class Odd2(int):
    """An ``int`` literal whose reflected product sees the native batch: ``n * Odd2(1)`` is ``n`` up to 8 and
    ``n + 2`` above for a native ``int``, plain ``n`` for anything else (a ``SymInt``)."""

    seen: list = []

    def __rmul__(self, other):
        Odd2.seen.append(type(other).__name__)
        if type(other) is int:
            return int(self) * other + (2 if other > 8 else 0)
        return int(self) * other

    def __repr__(self) -> str:
        return f"Odd2({int(self)})"


class OddProgram(nn.Module):
    """The reviewer's ``Odd2Zeros``, ``Odd2ZerosLit``, ``Odd2Arange`` and ``Odd2View`` (``r7_mods.py``), and the
    control ``y * Odd2(1)``."""

    FORMS = {
        "zeros": lambda m, y, b: y * torch.zeros(b * m.k - b + 6, dtype=y.dtype).shape[0],
        "zeros_literal_dtype": lambda m, y, b: y * torch.zeros(b * m.k - b + 6, dtype=torch.float64).shape[0],
        "arange": lambda m, y, b: y * torch.arange(b * m.k - b + 6).shape[0],
        "view": lambda m, y, b: F.softmax(y.view(b, (b * m.k - b) // 2 + 1, -1), -1).flatten(1),
        "product": lambda m, y, b: y * m.k,
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.k = Odd2(1)
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](self, self.fc(x), x.size(0))


def _host_ops(flow) -> list:
    return [m.name for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and m.name != "fc"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["zeros", "zeros_literal_dtype", "arange"])
def test_a_size_computed_through_a_user_int_literal_is_a_run_time_host_op_and_exact(how, contract):
    """Certified and wrong by rel 0.25 at batches 9 and 13 before (the literal ``getitem = 6``); refused by name on
    w0-base at its forward probe. The factory and its shape query are host ops now, and ``Odd2.__rmul__`` sees ints
    only (the native forward, the traces and the probe)."""
    Odd2.seen.clear()
    flow = assert_converted_exact(native(lambda: OddProgram(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    factory = "arange" if how == "arange" else "zeros"
    assert flow.shape_literals == {}
    assert {"mul", "sub", "add", factory, "getitem", "mul_1"} <= set(_host_ops(flow))
    assert Odd2.seen and set(Odd2.seen) == {"int"}


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_layout_a_user_int_literal_decides_is_exact(contract):
    """``Odd2View``: certified and wrong by rel 0.38 to 1.0 at batches 9 and 13 in round 7 (on w0-base too), refused by
    name in round 8 naming ``mul``. [restated in review round 9: impure, its view the host op of ``Tensor.view`` on the
    run-time size: exact at every batch; the stage-2 refusal names ``mul`` and the literal type]"""
    model = native(lambda: OddProgram("view"))
    flow = assert_impure_exact(model, (8,), contract, "mul", "mul", batches=EVERY_BATCH)
    assert "a literal operand of type Odd2 outside the declared literal types" in flow.shape_impurity.reason
    if contract == "mvm":
        assert "type Odd2" in assert_impure_stage_2_refused(model, (8,), "mul", "mul")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_product_with_a_user_int_literal_stays_exact(contract):
    assert_converted_exact(native(lambda: OddProgram("product")), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


def test_a_call_with_a_user_int_literal_never_runs_on_symbolic_values():
    """[restated in review round 9: the program is impure, so the pass never runs; round 8 asserted the opaque ``mul``
    was a batch decision whose users' shapes it decided]"""
    gm = trace_model(native(lambda: OddProgram("zeros")), (8,))
    Odd2.seen.clear()
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    assert layouts.impurity.node == "mul" and proof.batch is None and proof.shapes == {}
    assert set(Odd2.seen) <= {"int"} and layouts.layout(nodes["zeros"]) is None  # the seeded traces only


def _node(target, *args, **kwargs) -> fx.Node:
    graph = fx.Graph()
    return graph.create_node("call_function", target, (graph.placeholder("x"), *args), kwargs)


Pair = collections.namedtuple("Pair", "a b")


class TestDeclaredLiteralTypes:
    @pytest.mark.parametrize("literal", [
        1, 1.5, True, "s", None, Ellipsis, slice(1, None, 2), torch.float64, torch.device("cpu"), torch.strided,
        torch.contiguous_format, torch.Size([2, 3]), (1, [2, {"a": 3}]), immutable_list([1, 2]),
        immutable_dict({"a": 1}), torch.sigmoid,
    ], ids=lambda value: type(value).__name__)
    def test_a_literal_of_a_declared_type_keeps_the_call_declared(self, literal):
        assert opaque_reason(_node(torch.add, literal), None, {}) is None

    @pytest.mark.parametrize("literal", [
        Odd2(1), np.float64(1.0), (1, Odd2(1)), [Odd2(1)], {"k": Odd2(1)}, slice(None, Odd2(3)), Pair(1, 2), int,
        object(), 1j,  # [review round 9: complex is outside the round-9 closed list]
    ], ids=["int_subclass", "numpy_scalar", "in_tuple", "in_list", "in_dict", "in_slice", "namedtuple", "class",
            "object", "complex"])
    def test_a_literal_of_any_other_type_makes_the_call_opaque(self, literal):
        reason = opaque_reason(_node(torch.add, literal), None, {})
        assert reason is not None and reason.startswith("a declared call given a literal operand of type ")
        assert reason.endswith(" outside the declared literal types")

    def test_the_reason_names_the_literal_type(self):
        assert opaque_reason(_node(torch.add, Odd2(1)), None, {}) == (
            "a declared call given a literal operand of type Odd2 outside the declared literal types")

    def test_a_module_call_with_a_foreign_literal_is_opaque(self):
        graph = fx.Graph()
        node = graph.create_node("call_module", "m", (graph.placeholder("x"), Odd2(1)), {})
        reason = opaque_reason(node, None, {"m": nn.Identity()})
        assert reason == "a declared call given a literal operand of type Odd2 outside the declared literal types"


# ----------------------------------------------------------------------------- containers FX re-types
class Table(tuple):
    """A tuple literal whose ``__getitem__`` reads a scale ``note_batch`` sets: 1 up to batch 8, 2 above."""

    scale = {"v": 1}

    def __getitem__(self, index):
        return tuple.__getitem__(self, index) * Table.scale["v"]


class Rows(list):
    def __getitem__(self, index):
        return list.__getitem__(self, index) * Table.scale["v"]


class Weights(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key) * Table.scale["v"]


class Widths(tuple):
    pass


@torch.fx.wrap
def note_batch(t):
    Table.scale["v"] = 1 if t.shape[0] <= 8 else 2
    return t


class ContainerLiteral(nn.Module):
    """The reviewer's ``TableFold`` and its list, dict and torch-function forms."""

    FORMS = {
        "tuple": lambda m, y: note_batch(y) * _getitem(Table((3, 5)), y.shape[1] - 6),
        "list": lambda m, y: note_batch(y) * _getitem(Rows([3, 5]), y.shape[1] - 6),
        "dict": lambda m, y: note_batch(y) * _getitem(Weights({0: 3}), y.shape[1] - 6),
        "torch_function": lambda m, y: F.pad(y, Widths((0, 2))),
    }
    TYPES = {"tuple": ("Table", "tuple"), "list": ("Rows", "list"), "dict": ("Weights", "dict"),
             "torch_function": ("Widths", "tuple")}

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](self, self.fc(x))


@pytest.mark.parametrize("how", sorted(ContainerLiteral.FORMS))
def test_a_literal_container_subclass_is_refused_by_the_tracer(how):
    """``TableFold`` (``tuple``) was certified and wrong by rel 0.5 at batches 9 and 13: FX recorded ``(3, 5)``."""
    kind, base = ContainerLiteral.TYPES[how]
    operator = "pad" if how == "torch_function" else "getitem"
    with pytest.raises(TracingError) as excinfo:
        trace_model(native(lambda: ContainerLiteral(how)), (8,))
    assert isinstance(excinfo.value, ToolchainRefusal)
    assert str(excinfo.value).startswith(
        f"{operator} (the FX node being traced, call_function): a literal operand of type {kind}, a subclass of "
        f"{base}; FX records it as a plain {base}")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_the_table_fold_is_refused_by_name_under_both_contracts(contract):
    with pytest.raises(TracingError):
        convert(native(lambda: ContainerLiteral("tuple")), (8,), contract)


class KeptLiterals(nn.Module):
    """A namedtuple literal (FX records its construction as a node) and a ``torch.Size`` literal (FX records its
    entries): neither is re-typed, so the tracer keeps both."""

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        return F.pad(y, Pair(0, 2)) if self.how == "namedtuple" else F.pad(y, torch.Size([0, 2]))


@pytest.mark.parametrize("how", ["namedtuple", "size"])
def test_namedtuple_and_size_literals_are_not_refused_by_the_tracer(how):
    gm = trace_model(native(lambda: KeptLiterals(how)), (8,))
    pad = next(node for node in gm.graph.nodes if node.target is F.pad)
    assert isinstance(pad.args[1], fx.Node) if how == "namedtuple" else pad.args[1] == (0, 2)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_size_literal_still_converts_exactly(contract):
    assert_converted_exact(native(lambda: KeptLiterals("size")), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


class TestStaticTreesAreExactTypes:
    @pytest.mark.parametrize("value", [(1, 2), [1, 2], torch.Size([2]), immutable_list([1]),
                                       {"a": 1}, immutable_dict({"a": 1}), slice(1, 2)],
                             ids=lambda value: type(value).__name__)
    def test_plain_and_fx_containers_are_static(self, value):
        assert is_static_tree(value)

    @pytest.mark.parametrize("value", [Table((1, 2)), Rows([1]), Weights({0: 1}), Pair(1, 2), (1, Odd2(1))],
                             ids=["tuple_subclass", "list_subclass", "dict_subclass", "namedtuple", "int_subclass"])
    def test_a_container_subclass_is_not_static(self, value):
        assert not is_static_tree(value)


# ----------------------------------------------------------------------------- review round 13: namedtuple literals
class ShiftedPair(Pair):
    """A namedtuple whose constructor shifts its first field: FX records the literal as a call of its class on its
    iterated fields (``ShiftedPair(7, 0)``), so the converted program shifts it again (the reviewer's
    ``NamedTupleDirectRoll``)."""

    def __new__(cls, a, b):
        return super().__new__(cls, a + 6, b)


class IterLyingPair(Pair):
    """A namedtuple whose iteration differs from its storage: torch's argument parser reads the storage, ``(1, 0)``;
    FX iterates, ``(7, 0)`` (``NamedTupleIterRoll``)."""

    def __iter__(self):
        return iter([v + 6 if i == 0 else v for i, v in enumerate(tuple.__iter__(self))])


class FieldsList(list):
    """A list subclass carrying ``_fields``, duck-typed as a namedtuple, whose iteration differs from its storage: torch
    reads ``[1]``, FX records the plain list ``[7]`` (``FieldsListRoll``)."""

    _fields = ()

    def __iter__(self):
        return iter([v + 6 for v in list.__iter__(self)])


class TypedPair(typing.NamedTuple):
    a: int
    b: int


class RollLiteral(nn.Module):
    """``torch.roll(y, shifts=<literal>, dims=...)``: certified on the round-12 tree for the three user classes and
    wrong at batches 4, 5, 7, 9 and 13 by rel 0.6 to 1.2 (the model object itself converted, no deep copy); w0-base
    refused the namedtuple forms at its probe and was wrong on the list form."""

    SHIFTS = {"shifted_pair": lambda: ShiftedPair(1, 0), "iter_lying_pair": lambda: IterLyingPair(1, 0),
              "fields_list": lambda: FieldsList([1]), "plain_namedtuple": lambda: Pair(1, 0),
              "typed_namedtuple": lambda: TypedPair(1, 0)}

    def __init__(self, how: str):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.shift = self.SHIFTS[how]()
        self.dims = 0 if how == "fields_list" else (0, 1)

    def forward(self, x):
        y = self.fc(x)
        return torch.roll(y, shifts=self.shift, dims=self.dims) + y


ROLL_REFUSALS = {  # form -> the start of its refusal after the operator
    "shifted_pair": "a literal operand of type ShiftedPair, a subclass of tuple other than a namedtuple class as "
                    "collections.namedtuple makes it; FX records it as a call of ShiftedPair on its iterated fields",
    "iter_lying_pair": "a literal operand of type IterLyingPair, a subclass of tuple other than a namedtuple class "
                       "as collections.namedtuple makes it; FX records it as a call of IterLyingPair on its iterated "
                       "fields",
    "fields_list": "a literal operand of type FieldsList, a subclass of list; FX records it as a plain list",
}


@pytest.mark.parametrize("how", sorted(ROLL_REFUSALS))
def test_a_namedtuple_literal_of_a_class_with_its_own_methods_is_refused_by_the_tracer(how):
    """The tracer exempted any container with a ``_fields`` attribute; it exempts an unmodified
    ``collections.namedtuple`` or ``typing.NamedTuple`` class only, and refuses every other container subclass."""
    model = native(lambda: RollLiteral(how))
    with pytest.raises(TracingError) as excinfo:
        trace_model(model, (8,))
    assert str(excinfo.value).startswith(f"roll (the FX node being traced, call_function): {ROLL_REFUSALS[how]}"), (
        str(excinfo.value))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(ROLL_REFUSALS))
def test_the_namedtuple_roll_forms_are_refused_by_name_under_both_contracts(how, contract):
    with pytest.raises(TracingError):
        convert(native(lambda: RollLiteral(how)), (8,), contract)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["plain_namedtuple", "typed_namedtuple"])
def test_a_plain_namedtuple_shift_keeps_its_conversion(how, contract):
    """FX's call of an unmodified namedtuple class on its fields rebuilds the same tuple: the program is impure (the
    call is outside the declared operator set) and exact at every batch, as on the round-12 tree."""
    assert_impure_exact(native(lambda: RollLiteral(how)), (8,), contract, {
        "plain_namedtuple": "pair", "typed_namedtuple": "typed_pair"}[how],
        {"plain_namedtuple": "Pair", "typed_namedtuple": "TypedPair"}[how], batches=(1, 2, 3, 4, 5, 7, 9, 13))


@pytest.mark.parametrize("value, plain", [
    (Pair(1, 0), True), (TypedPair(1, 0), True), (ShiftedPair(1, 0), False), (IterLyingPair(1, 0), False),
    (FieldsList([1]), False), (type("Fake", (tuple,), {"_fields": ("a",)})((1,)), False),
], ids=["namedtuple", "typed", "own_new", "own_iter", "list_with_fields", "tuple_with_fields"])
def test_only_an_unmodified_namedtuple_class_is_exempt(value, plain):
    from mimarsinan.torch_mapping.torch_graph_tracer import _retyped_literal

    assert (_retyped_literal(value) is None) is plain
