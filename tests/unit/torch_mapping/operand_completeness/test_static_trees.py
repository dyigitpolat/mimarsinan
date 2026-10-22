"""A shape value is evaluated at conversion time only when its whole argument tree is static (review round 5).

Static: every leaf is a concrete Python ``int``, ``float``, ``bool``, ``str`` or ``None`` (exact types), through
tuples, lists, dicts, ``torch.Size`` and slices, and every extent in it is proved by the symbolic-batch pass. A tree
that holds an unproven extent never reaches a Python function at conversion time: CPython compares containers by
identity first, so with one stand-in for every run-time extent ``(B, 6) == (B, 6)`` held for two different extents
and ``y * (y[::3].shape == y[:2].shape) + y`` folded to ``2 * y``, wrong at batches 2, 3, 7, 8, 12 and 13. Such a value
keeps its w0-base host ops, which evaluate it at run time. The one operation on such a tree is a projection: an
entry or a slice of a recorded shape at a static index, or its length, which reads no entry. An exception of a static
evaluation propagates as itself, never as a fold.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.immutable_collections import immutable_dict, immutable_list

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.shape_value_mixin import (
    BATCH_SIZE,
    evaluate_shape_value,
    is_static_tree,
)

from .harness import CONTRACTS, PACKAGINGS, assert_converted_exact, convert, native, quietly

B = BATCH_SIZE


# ----------------------------------------------------------------------------- the predicate
class _SneakyInt(int):
    """An ``int`` whose comparisons are user code: not a concrete Python int."""

    def __eq__(self, other):
        return True

    __hash__ = int.__hash__


STATIC_TREES = [
    6, -1, 0.5, float("inf"), True, None, "dim", (), (4, 8), [4, (8, None)], {"dim": 1}, torch.Size([4, 8]),
    slice(1, None), slice(None, None, 2), ((4,), {"k": [slice(0, 2)]}), immutable_list([4, 8]),
    immutable_dict({"dim": 1, "keepdim": True}),
]
UNPROVEN_TREES = [
    B, (B, 6), [6, (B,)], {"dim": B}, slice(B, None), slice(None, 2, B), ((), {"k": (6, B)}), immutable_list([B]),
    immutable_dict({"size": (B, 4)}), ((4, 8), [{"a": [(), (B,)]}]),
]


FOREIGN_LEAVES = {
    "tensor": lambda: torch.tensor(6),
    "dtype": lambda: torch.float64,
    "device": lambda: torch.device("cpu"),
    "numpy_int": lambda: np.int64(6),
    "symint": lambda: ShapeEnv().create_unbacked_symint(),
    "int_subclass": lambda: _SneakyInt(6),
    "complex": lambda: complex(1, 2),
    "ellipsis": lambda: Ellipsis,
    "object": object,
    "tensor_in_tuple": lambda: (4, torch.tensor(8)),
    "dtype_in_dict": lambda: {"dtype": torch.float64},
}


class TestStaticTree:
    @pytest.mark.parametrize("tree", STATIC_TREES, ids=repr)
    def test_concrete_python_leaves_through_containers_are_static(self, tree):
        assert is_static_tree(tree)

    @pytest.mark.parametrize("tree", UNPROVEN_TREES, ids=range(len(UNPROVEN_TREES)))
    def test_an_unproven_extent_anywhere_in_the_tree_is_not_static(self, tree):
        assert not is_static_tree(tree)

    @pytest.mark.parametrize("leaf", sorted(FOREIGN_LEAVES))
    def test_no_other_object_is_a_static_leaf(self, leaf):
        assert not is_static_tree(FOREIGN_LEAVES[leaf]())


# ----------------------------------------------------------------------------- no evaluation of an unproven tree
def _spy():
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    return spy, calls


UNPROVEN_CALLS = [
    (((B, 6), (B, 6)), {}),
    ((B, 2), {}),
    (([6, (B,)],), {}),
    ((6,), {"size": (B, 4)}),
    (((4, 8), slice(B, None)), {}),
]


class TestNoEvaluationOfAnUnprovenTree:
    @pytest.mark.parametrize("args,kwargs", UNPROVEN_CALLS, ids=range(len(UNPROVEN_CALLS)))
    def test_a_function_is_never_called_on_an_unproven_tree(self, args, kwargs):
        spy, calls = _spy()
        ok, value = evaluate_shape_value(spy, args, kwargs)
        assert ok and value is BATCH_SIZE and calls == []

    @pytest.mark.parametrize("compare", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt,
                                         operator.ge, operator.contains])
    def test_a_container_comparison_of_run_time_extents_is_not_folded(self, compare):
        """CPython's identity shortcut made ``(B, 6) == (B, 6)`` True for two different run-time extents."""
        ok, value = evaluate_shape_value(compare, ((B, 6), (B, 6)), {})
        assert ok and value is BATCH_SIZE

    @pytest.mark.parametrize("fn", [torch.sym_max, torch.sym_min, torch.sym_float, max, min])
    def test_a_function_that_would_raise_on_the_stand_in_is_not_called(self, fn):
        """``torch.sym_max(B, 2)`` raised an AssertionError at conversion: an anonymous crash."""
        args = (B,) if fn is torch.sym_float else (B, 2)
        ok, value = evaluate_shape_value(fn, args, {})
        assert ok and value is BATCH_SIZE


class _Observed:
    """Raises on, and records, every observation of itself: a projection must make none."""

    def __init__(self):
        self.seen = []


def _observer(name):
    def observe(self, *_args):
        self.seen.append(name)
        raise AssertionError(f"observed by {name}")
    return observe


for _name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__bool__", "__index__", "__int__",
              "__float__", "__hash__", "__repr__", "__str__", "__format__", "__len__", "__iter__"):
    setattr(_Observed, _name, _observer(_name))


class TestProjections:
    def test_an_entry_or_a_slice_at_a_static_index_reads_no_entry(self):
        entry = _Observed()
        shape = (entry, 4, 8)
        assert evaluate_shape_value(operator.getitem, (shape, 1), {}) == (True, 4)
        assert evaluate_shape_value(operator.getitem, (shape, -1), {}) == (True, 8)
        assert evaluate_shape_value(operator.getitem, (shape, slice(1, None)), {}) == (True, (4, 8))
        ok, picked = evaluate_shape_value(operator.getitem, (shape, 0), {})
        assert ok and picked is entry
        ok, picked = evaluate_shape_value(operator.getitem, (shape, slice(None, 1)), {})
        assert ok and picked[0] is entry and len(picked) == 1
        assert evaluate_shape_value(len, (shape,), {}) == (True, 3)
        assert entry.seen == []

    def test_a_projection_of_an_unproven_entry_stays_unproven(self):
        ok, value = evaluate_shape_value(operator.getitem, ((B, 4), 0), {})
        assert ok and value is BATCH_SIZE
        ok, value = evaluate_shape_value(operator.getitem, ((B, 4), slice(None, 1)), {})
        assert ok and not is_static_tree(value)

    @pytest.mark.parametrize("index", [B, slice(B, None), slice(None, B), (0,), 1.0])
    def test_a_projection_needs_a_static_integer_index(self, index):
        ok, value = evaluate_shape_value(operator.getitem, ((B, 4), index), {})
        assert ok and value is BATCH_SIZE


# ----------------------------------------------------------------------------- a static evaluation is loud
def _raises(exc):
    def fn(*_args):
        raise exc("raised on a static tree")
    return fn


class TestStaticEvaluation:
    def test_a_static_tree_evaluates(self):
        assert evaluate_shape_value(operator.eq, ((4, 8), (4, 8)), {}) == (True, True)
        assert evaluate_shape_value(torch.sym_max, (6, 2), {}) == (True, 6)
        assert evaluate_shape_value(operator.truediv, (6, 4), {}) == (True, 1.5)

    @pytest.mark.parametrize("fn,args,exc", [
        (operator.truediv, (6, 0), ZeroDivisionError),
        (operator.getitem, ((4, 8), 5), IndexError),
        (operator.add, ((4, 8), 1), TypeError),
        (torch._assert, (False, "a static assertion"), AssertionError),
        (math.sqrt, (-1,), ValueError),
    ], ids=["zero_division", "index", "type_error", "assertion", "value_error"])
    def test_an_exception_of_a_static_evaluation_propagates_as_itself(self, fn, args, exc):
        """Never swallowed into a literal or a fallback: the closed-set operator receives the native values and
        types, so the native model raises it too (it should not happen). Review round 6: a callable outside the
        closed set is never called at conversion time (``test_conversion_time_operators.py``), so the user
        functions that raised here before are host ops now."""
        with pytest.raises(exc):
            evaluate_shape_value(fn, args, {})

    @pytest.mark.parametrize("exc", [TypeError, AssertionError, RuntimeError], ids=lambda e: e.__name__)
    def test_a_raising_function_outside_the_closed_set_is_not_called(self, exc):
        """The round-5 user-function cases: not called at all (round 6), so nothing to propagate at conversion."""
        assert evaluate_shape_value(_raises(exc), (6,), {}) == (False, None)


# ----------------------------------------------------------------------------- comparisons of run-time extents
class ShapeComparison(nn.Module):
    """Comparisons of shapes whose batch axis is a run-time extent (round-4 blocker, its forms and controls)."""

    FORMS = {
        "slice_eq": lambda x, y: y[::3].shape == y[:2].shape,
        "slice_ne": lambda x, y: y[::3].shape != y[:2].shape,
        "slice_lt": lambda x, y: y[::3].shape < y[:2].shape,
        "stride_eq": lambda x, y: y[::7].shape == y[::6].shape,
        "stride_ne": lambda x, y: y[::7].shape != y[::6].shape,
        "stride_lt": lambda x, y: y[::7].shape < y[::6].shape,
        "stride_head_eq": lambda x, y: y[::7].shape[:1] == y[::6].shape[:1],
        "tail_vs_full": lambda x, y: y[1:].shape == y.shape,
        "head2_vs_full": lambda x, y: y[:2].shape == y.shape,
        "output_vs_input": lambda x, y: y.shape == x.shape,
        "output_head_vs_input_head": lambda x, y: y.shape[:1] == x.shape[:1],
        "tail_vs_init": lambda x, y: y[1:].shape == y[:-1].shape,
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        return y * self.FORMS[self.how](x, y) + y


_SLICED_PAIR = ["getitem", "getattr_1", "getitem_1", "getattr_2"]
# form -> the w0-base host ops (exact there at every batch); the comparison's own operator last but two.
COMPARISON_HOST_OPS = {
    "slice_eq": _SLICED_PAIR + ["eq", "mul", "add"],
    "slice_ne": _SLICED_PAIR + ["ne", "mul", "add"],
    "slice_lt": _SLICED_PAIR + ["lt", "mul", "add"],
    "stride_eq": _SLICED_PAIR + ["eq", "mul", "add"],
    "stride_ne": _SLICED_PAIR + ["ne", "mul", "add"],
    "stride_lt": _SLICED_PAIR + ["lt", "mul", "add"],
    "stride_head_eq": ["getitem", "getattr_1", "getitem_1", "getitem_2", "getattr_2", "getitem_3", "eq", "mul", "add"],
    "tail_vs_full": ["getitem", "getattr_1", "getattr_2", "eq", "mul", "add"],
    "head2_vs_full": ["getitem", "getattr_1", "getattr_2", "eq", "mul", "add"],
    "output_vs_input": ["getattr_1", "getattr_2", "eq", "mul", "add"],
    "output_head_vs_input_head": ["getattr_1", "getitem", "getattr_2", "getitem_1", "eq", "mul", "add"],
    "tail_vs_init": _SLICED_PAIR + ["eq", "mul", "add"],
}
COMPARISON_BATCHES = (1, 2, 3, 4, 6, 7, 8, 12, 13)


def host_ops(flow) -> list:
    return [m.name for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and m.name != "fc"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(COMPARISON_HOST_OPS))
def test_a_comparison_of_run_time_extents_is_exact_at_every_batch(how, contract):
    """slice and stride forms: certified and wrong by rel 0.5-1.0 at batches 2, 3, 7, 8, 12, 13 before; the full-shape
    forms refused by the probe; the controls folded (exact). All keep the w0-base host ops now."""
    flow = assert_converted_exact(native(lambda: ShapeComparison(how)), (8,), contract,
                                  batches=COMPARISON_BATCHES, kinds=("normal", "uniform"))
    assert host_ops(flow) == COMPARISON_HOST_OPS[how] and flow.shape_literals == {}


_OFF_BATCH = ("HostValueBatchLayoutError", "host op getitem", "getitem")  # y[1:]: axis 0 is not the batch
_RUN_TIME_SHAPE = ("NonTensorHostValueError", "host op getattr .shape", "getattr_1")
# [restated in review round 14: a slice of the batch axis whose length is no affine form of B (``y[::3]``, ``y[:2]``)
# is an opaque value, and stage 2 refuses a program with one by name, naming the first (``OpaqueBatchValueError``);
# ``y[1:]`` is ``B - 1``, in the domain, and keeps its round-4 refusal]
_OPAQUE_SLICE = ("OpaqueBatchValueError", "getitem", "getitem")
# form -> the stage-2 refusal: class, operator and FX node of the first host value without an IR form
COMPARISON_STAGE_2 = {how: _OPAQUE_SLICE for how in COMPARISON_HOST_OPS}
COMPARISON_STAGE_2.update(output_vs_input=_RUN_TIME_SHAPE, output_head_vs_input_head=_RUN_TIME_SHAPE,
                          tail_vs_full=_OFF_BATCH, tail_vs_init=_OFF_BATCH)


@pytest.mark.parametrize("how", sorted(COMPARISON_STAGE_2))
def test_a_comparison_of_run_time_extents_is_refused_by_name_at_stage_2(how):
    """The slice and stride forms passed stage 2 wrong before; the controls passed it as a literal (declared change)."""
    refusal, operator_name, node = COMPARISON_STAGE_2[how]
    model = native(lambda: ShapeComparison(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    message = str(excinfo.value)
    assert refusal in message and f"{operator_name} (FX node {node!r})" in message


# ----------------------------------------------------------------------------- an extent no Python function takes
class SymMaxOfExtent(nn.Module):
    """``torch.sym_max`` of a run-time extent: an anonymous AssertionError at conversion in review round 4."""

    FORMS = {
        "batch": lambda x, y: torch.sym_max(x.size(0), 2),
        "stride": lambda x, y: torch.sym_max(y[::6].size(0), 2),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        return y * self.FORMS[self.how](x, y) + y


# form -> (host ops, the stage-2 refusal: class, operator and FX node of the first host value without an IR form)
SYM_MAX = {  # [restated in review round 14: stage 2 names the first opaque value, ``OpaqueBatchValueError``]
    "batch": (["size", "sym_max", "mul", "add"], ("OpaqueBatchValueError", "sym_max", "sym_max")),
    "stride": (["getitem", "size", "sym_max", "mul", "add"], ("OpaqueBatchValueError", "getitem", "getitem")),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(SYM_MAX))
def test_a_function_of_a_run_time_extent_keeps_its_host_op_and_is_exact(how, contract):
    """The host op runs ``torch.sym_max`` at run time (w0-base refused it by name at its forward probe)."""
    flow = assert_converted_exact(native(lambda: SymMaxOfExtent(how)), (8,), contract, batches=COMPARISON_BATCHES,
                                  kinds=("normal", "uniform"))
    assert host_ops(flow) == SYM_MAX[how][0] and flow.shape_literals == {}


@pytest.mark.parametrize("how", sorted(SYM_MAX))
def test_a_function_of_a_run_time_extent_is_refused_by_name_at_stage_2(how):
    refusal, operator_name, node = SYM_MAX[how][1]
    model = native(lambda: SymMaxOfExtent(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    message = str(excinfo.value)
    assert refusal in message and f"{operator_name} (FX node {node!r})" in message


# ----------------------------------------------------------------------------- the static forms still fold
class StaticForms(nn.Module):
    FORMS = {
        "entry": lambda x, y: y * y.shape[1],
        "slice_comparison": lambda x, y: y * (x.shape[1:] == (8,)) + y,
        "slice_entry": lambda x, y: y * y.shape[-1:][0],
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


# A slice of a shape is a ``torch.Size``, as natively (review round 6: "(8,)" and "(6,)" before).
STATIC_LITERALS = {
    "entry": {"getitem": ("getitem", "6")},
    "slice_comparison": {"getitem": ("getitem", "torch.Size([8])"), "eq": ("eq", "True")},
    "slice_entry": {"getitem": ("getitem", "torch.Size([6])"), "getitem_1": ("getitem", "6")},
}


@pytest.mark.parametrize("how", sorted(STATIC_LITERALS))
def test_a_projection_of_a_proved_axis_is_still_a_literal(how):
    """The static-folding benefit kept: a proved axis picked out of a shape with a run-time batch axis."""
    flow = assert_converted_exact(native(lambda: StaticForms(how)), (8,), "spiking", batches=COMPARISON_BATCHES)
    assert flow.shape_literals == STATIC_LITERALS[how]


# ----------------------------------------------------------------------------- the stage-2 refusal names the operator
class RunTimeShapeValue(nn.Module):
    FORMS = {
        "size_eq": lambda x, y: y * (x.size(0) == 1) + y,
        "shape_entry_eq": lambda x, y: y * (x.shape[0] == 1) + y,
        "shape_eq": lambda x, y: y * (x.shape == (1, 8)) + y,
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


# form -> (operator, FX node) the stage-2 refusal names: [restated in review round 14: the first host op whose value
# is not a tensor (``NonTensorHostValueError``); the comparison of the batch with a constant is the first opaque value
# now (``B == 1`` changes with the batch), named by ``OpaqueBatchValueError``]
RUN_TIME_SHAPE_HOST_VALUE = {
    "size_eq": ("eq", "eq"),
    "shape_entry_eq": ("eq", "eq"),
    "shape_eq": ("eq", "eq"),
}


@pytest.mark.parametrize("how", sorted(RUN_TIME_SHAPE_HOST_VALUE))
def test_a_run_time_shape_value_is_refused_at_stage_2_by_operator_and_node(how):
    """``NonTensorHostValueError`` named the FX node and the adapter class only ("host op 'getattr_1' (ComputeAdapter)")."""
    operator_name, node = RUN_TIME_SHAPE_HOST_VALUE[how]
    model = native(lambda: RunTimeShapeValue(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    message = str(excinfo.value)
    assert "OpaqueBatchValueError" in message and f"{operator_name} (FX node {node!r})" in message


@pytest.mark.parametrize("how", sorted(RUN_TIME_SHAPE_HOST_VALUE))
def test_a_run_time_shape_value_is_exact_in_the_converted_model(how):
    assert_converted_exact(native(lambda: RunTimeShapeValue(how)), (8,), "mvm", batches=COMPARISON_BATCHES,
                           kinds=("normal", "uniform"))
