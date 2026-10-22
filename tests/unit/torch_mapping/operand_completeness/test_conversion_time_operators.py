"""Conversion-time evaluation is a closed set of pure operators on native values (review round 6).

A static shape value is evaluated at conversion time only when its operator is in ``CONVERSION_TIME_OPERATORS`` (the
``operator`` module, ``math``, ``torch.sym_*``, ``torch._assert``); any other call target FX records, a user's
``torch.fx.wrap`` helper or a tensor factory among them, is never called at conversion and keeps its w0-base host op.
Before, every call on a static tree was evaluated, and static shapes were plain tuples: ``is_size(y.shape[1:])``
(``isinstance(s, torch.Size)``) folded to 0, ``len(str(s))`` to the tuple's width, and ``s.numel()`` raised an
anonymous ``AttributeError``, while the native model hands the helper a ``torch.Size``. A static whole shape and a
slice of a shape are now ``torch.Size`` literals, as natively. A call whose every argument is static (a factory fed by
a folded shape, ``torch.zeros(y.shape[-1])``) runs as a host op anchored on the value its shapes were read from, as on
w0-base, instead of losing its only source (refused before, ``GenericOpArgumentError``).
"""

from __future__ import annotations

import copy
import math
import operator

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model, trace_and_convert
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.converter_handlers.shape_value_mixin import (
    BATCH_SIZE,
    CONVERSION_TIME_OPERATORS,
    evaluate_shape_value,
)

from .harness import (
    CONTRACTS,
    PACKAGINGS,
    assert_converted_exact,
    assert_impure_exact,
    assert_impure_stage_2_refused,
    convert,
    native,
    quietly,
)

B = BATCH_SIZE
EVERY_BATCH = (1, 2, 3, 6, 7, 13)


@torch.fx.wrap
def is_size(s):
    return int(isinstance(s, torch.Size))


@torch.fx.wrap
def width_code(s):
    """A helper whose value depends on the shape's string form."""
    return len(str(s))


@torch.fx.wrap
def numel_scale(s):
    return s.numel() ** -0.5


# ----------------------------------------------------------------------------- the closed set
class TestTheClosedSet:
    @pytest.mark.parametrize("fn", [
        operator.add, operator.mul, operator.sub, operator.floordiv, operator.truediv, operator.pow, operator.neg,
        operator.getitem, operator.eq, operator.lt, operator.contains, operator.not_, math.ceil, math.prod,
        torch.sym_max, torch.sym_min, torch.sym_float, torch.sym_int, torch._assert,
    ], ids=lambda fn: getattr(fn, "__name__", repr(fn)))
    def test_pure_operators_are_in_it(self, fn):
        assert fn in CONVERSION_TIME_OPERATORS

    @pytest.mark.parametrize("fn", [
        is_size, width_code, numel_scale, torch.zeros, torch.randn, torch.arange, len, str, repr, isinstance,
        operator.setitem, operator.delitem, operator.iadd, operator.attrgetter("numel"), lambda s: s,
    ], ids=lambda fn: getattr(fn, "__name__", type(fn).__name__))
    def test_no_other_callable_is_in_it(self, fn):
        assert fn not in CONVERSION_TIME_OPERATORS


def _spy():
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("called at conversion time")

    return spy, calls


class TestNoOtherCallIsEvaluated:
    @pytest.mark.parametrize("args,kwargs", [
        ((torch.Size([6]),), {}), ((6,), {}), (((4, 8), 1), {}), ((6,), {"dim": 1}),
    ], ids=["shape", "entry", "tuple", "keyword"])
    def test_a_callable_outside_the_set_is_never_called_on_a_static_tree(self, args, kwargs):
        spy, calls = _spy()
        assert evaluate_shape_value(spy, args, kwargs) == (False, None) and calls == []

    def test_membership_is_by_identity(self):
        """A callable equal to every operator (its own ``__eq__``) and unhashable is neither in the set nor called."""

        class EqualToEverything:
            __hash__ = None  # type: ignore[assignment]

            def __eq__(self, other):
                return True

            def __call__(self, *args):
                raise AssertionError("called at conversion time")

        assert evaluate_shape_value(EqualToEverything(), (6,), {}) == (False, None)

    def test_a_random_factory_is_not_drawn_at_conversion_time(self):
        """``evaluate_shape_value`` called ``torch.randn(6)`` and advanced the global RNG (w0-base does not)."""
        before = torch.get_rng_state()
        assert evaluate_shape_value(torch.randn, (6,), {}) == (False, None)
        assert torch.equal(before, torch.get_rng_state())

    def test_a_closed_set_operator_still_evaluates(self):
        assert evaluate_shape_value(operator.mul, (6, 2), {}) == (True, 12)
        assert evaluate_shape_value(math.prod, (torch.Size([4, 8]),), {}) == (True, 32)


class TestNativeShapeTypes:
    def test_a_slice_of_a_shape_is_a_size(self):
        """``y.shape[1:]`` is a ``torch.Size`` natively; it was the tuple ``(4, 8)``."""
        ok, value = evaluate_shape_value(operator.getitem, ((B, 4, 8), slice(1, None)), {})
        assert ok and type(value) is torch.Size and value == torch.Size([4, 8])
        ok, value = evaluate_shape_value(operator.getitem, (torch.Size([4, 8]), slice(1, None)), {})
        assert ok and type(value) is torch.Size and value == torch.Size([8])

    def test_an_entry_of_a_shape_is_an_int(self):
        assert evaluate_shape_value(operator.getitem, (torch.Size([4, 8]), -1), {}) == (True, 8)

    def test_a_slice_holding_an_unproven_extent_is_not_a_size(self):
        ok, value = evaluate_shape_value(operator.getitem, ((B, 4, 8), slice(None, 2)), {})
        assert ok and type(value) is tuple and value[0] is B

    def test_a_closed_set_result_keeps_its_native_type(self):
        """The ``torch.Size`` to tuple conversion of a result is gone: ``Size + tuple`` is what the native model gets."""
        ok, value = evaluate_shape_value(operator.add, (torch.Size([4]), (8,)), {})
        assert ok and value == torch.Size([4]) + (8,) and type(value) is type(torch.Size([4]) + (8,))


# ----------------------------------------------------------------------------- fx.wrap helpers over static shapes
class WrapHelper(nn.Module):
    """``torch.fx.wrap`` helpers on static shapes (review round 6 blocker): each is a call FX records faithfully."""

    FORMS = {
        "is_size_gate": lambda x, y: y * (y.shape[0] > 6 + 7 * is_size(y.shape[1:])) + y,
        "width_code_gate": lambda x, y: y * (y.shape[0] > 2 * width_code(y.shape[1:])) + y,
        "is_size_slice": lambda x, y: y[: 6 + 7 * is_size(y.shape[1:])],
        "width_code_slice": lambda x, y: y[: 2 * width_code(y.shape[1:])],
        "is_size_scale": lambda x, y: y * (1 + is_size(y.shape[1:])),
        "numel_scale": lambda x, y: y * numel_scale(y.shape[1:]),
        "whole_shape": lambda x, y: y * (1 + is_size(y[0].shape)),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


# form -> the host ops: w0-base's run-time shape queries. [restated in review round 9: a helper makes the program
# impure, so nothing of it is folded: the static entry ``y.shape[1:]`` (``getitem`` or ``getitem_1``; ``getattr_1`` in
# ``whole_shape``) is a host op again, as on w0-base; rounds 7 and 8 folded it into the helper's literal
# ``torch.Size([6])``. Round 8 had found the slice forms without a static proof (their slice's guard on B).]
WRAP_HOST_OPS = {
    "is_size_gate": ["getattr_1", "getitem", "getattr_2", "getitem_1", "is_size", "mul", "add", "gt", "mul_1", "add_1"],
    "width_code_gate": ["getattr_1", "getitem", "getattr_2", "getitem_1", "width_code", "mul", "gt", "mul_1", "add"],
    "is_size_slice": ["getattr_1", "getitem", "is_size", "mul", "add", "getitem_1"],
    "width_code_slice": ["getattr_1", "getitem", "width_code", "mul", "getitem_1"],
    "is_size_scale": ["getattr_1", "getitem", "is_size", "add", "mul"],
    "numel_scale": ["getattr_1", "getitem", "numel_scale", "mul"],
    "whole_shape": ["getitem", "getattr_1", "is_size", "add", "mul"],
}
HELPER = {how: "width_code" if how.startswith("width") else "numel_scale" if how == "numel_scale" else "is_size"
          for how in WRAP_HOST_OPS}


def _host_ops(flow) -> list:
    return [m.name for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and m.name != "fc"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(WRAP_HOST_OPS))
def test_a_wrapped_helper_on_a_static_shape_is_a_host_op_and_exact_at_every_batch(how, contract):
    """The gate forms were certified and wrong by rel 1.0 at batches 7 and 13, the slice forms truncated there, and
    ``numel_scale`` an anonymous ``AttributeError``; all exact on w0-base (the slices refused by name there)."""
    flow = assert_impure_exact(native(lambda: WrapHelper(how)), (8,), contract, HELPER[how], HELPER[how],
                               batches=EVERY_BATCH)
    assert _host_ops(flow) == WRAP_HOST_OPS[how]


@pytest.mark.parametrize("how", ["is_size_slice", "width_code_slice"])
def test_a_slice_of_the_batch_axis_by_a_helpers_value_has_no_static_proof(how):
    """Declared change (round 8): the slice's guard on B was unseen while the pass skipped the helper's users.
    [restated in review round 9: the statement names the helper that makes the program impure; round 8's was the
    slice's guard, ``B < 13`` or ``B < 30``]"""
    flow = convert(native(lambda: WrapHelper(how)), (8,), "mvm")
    statement = (f"no static shape proof: impure program at {HELPER[how]} (FX node {HELPER[how]!r}) (a call outside "
                 "the declared operator set; its operand 's' is not a strided tensor (a value of type Size))")  # [the
    # reason leaf-call inlining appends, restated in review round 1 of the coverage fixes (the operand is a shape)
    # and in fix round 2 (a traced operand must be a tensor at every trace batch)]
    assert flow.shape_proof == statement and flow.conversion_certificate.shape_proof == statement


@pytest.mark.parametrize("how", sorted(WRAP_HOST_OPS))
def test_a_wrapped_helper_on_a_static_shape_is_refused_by_name_at_stage_2(how):
    """The slice forms passed stage 2 wrong before; a helper's value is a run-time host value with no IR form.
    [restated in review round 9: refused as an impure program, naming the helper; round 8 named a run-time shape value
    (``NonTensorHostValueError``) or a host op off the batch layout (``HostValueBatchLayoutError``)]"""
    assert_impure_stage_2_refused(native(lambda: WrapHelper(how)), (8,), HELPER[how], HELPER[how])


# ----------------------------------------------------------------------------- factories fed by folded shapes
class ShapeFactory(nn.Module):
    """Tensor factories whose every argument is a folded static shape (review round 6, major 2)."""

    FORMS = {
        "zeros_entry": lambda x, y: y + torch.zeros(y.shape[-1], dtype=torch.float64),
        "zeros_shape": lambda x, y: y + torch.zeros(y.shape[1:], dtype=torch.float64),
        "arange": lambda x, y: y + torch.arange(y.shape[1], dtype=torch.float64),
        "ones_scaled": lambda x, y: y + torch.ones(y.shape[-1], dtype=torch.float64) * 2,
        "zeros_like_and_arange": lambda x, y: (
            y + torch.zeros_like(y[0]) + torch.arange(y.shape[1], dtype=torch.float64)),
        "arange_size": lambda x, y: y + torch.arange(y.size(1), dtype=torch.float64),
        "eye_matmul": lambda x, y: y @ torch.eye(y.shape[1], dtype=torch.float64),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


# form -> the host ops (the factory anchored on the value its shape was read from)
FACTORY_HOST_OPS = {
    "zeros_entry": ["getattr_1", "zeros", "add"],
    "zeros_shape": ["getattr_1", "zeros", "add"],
    "arange": ["getattr_1", "arange", "add"],
    "ones_scaled": ["getattr_1", "ones", "mul", "add"],
    "zeros_like_and_arange": ["getitem", "zeros_like", "add", "getattr_1", "arange", "add_1"],
    "arange_size": ["arange", "add"],
    "eye_matmul": ["getattr_1", "eye", "matmul"],
}
# form -> (operator, FX node) of the stage-2 refusal: the factory's value carries no batch axis
FACTORY_STAGE_2 = {
    "zeros_entry": ("zeros", "zeros"),
    "zeros_shape": ("zeros", "zeros"),
    "arange": ("arange", "arange"),
    "ones_scaled": ("ones", "ones"),
    "zeros_like_and_arange": ("getitem", "getitem"),
    "arange_size": ("arange", "arange"),
    "eye_matmul": ("eye", "eye"),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(FACTORY_HOST_OPS))
def test_a_factory_of_folded_shapes_is_an_anchored_host_op_and_exact(how, contract):
    """Refused before (``GenericOpArgumentError``: the fold had removed the factory's only source); exact on w0-base
    (``eye_matmul`` certified there and crashing at batch 2, the elementwise operand check)."""
    flow = assert_converted_exact(native(lambda: ShapeFactory(how)), (8,), contract,
                                  batches=tuple(range(1, 14)), kinds=("normal", "uniform"))
    assert _host_ops(flow) == FACTORY_HOST_OPS[how]


def test_a_host_op_whose_operands_do_not_broadcast_is_not_checked_as_elementwise():
    """``y @ eye(6)``: (B, 6) and (6, 6) broadcast only at batches 1 and 6, the probe's, so the elementwise operand
    check made the certified conversion crash at batch 2 (on w0-base too); an op is elementwise when its traced
    operands broadcast to its traced output."""
    flow = convert(native(lambda: ShapeFactory("eye_matmul")), (8,), "spiking")
    flags = {m.name: m.elementwise_inputs for m in flow.get_mapper_repr().execution_order()
             if isinstance(m, ComputeOpMapper) and m.name in ("matmul",)}
    assert flags == {"matmul": False}
    flow = convert(native(lambda: ShapeFactory("zeros_like_and_arange")), (8,), "spiking")
    flags = {m.name: m.elementwise_inputs for m in flow.get_mapper_repr().execution_order()
             if isinstance(m, ComputeOpMapper) and m.name in ("add", "add_1")}
    assert flags == {"add": True, "add_1": True}


@pytest.mark.parametrize("how", sorted(FACTORY_STAGE_2))
def test_a_factory_of_folded_shapes_is_refused_by_name_at_stage_2(how):
    operator_name, node = FACTORY_STAGE_2[how]
    model = native(lambda: ShapeFactory(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    message = str(excinfo.value)
    assert "HostValueBatchLayoutError" in message and f"host op {operator_name} (FX node {node!r})" in message


class RandomFactory(nn.Module):
    """``torch.randn`` of a folded shape: the static tree the round-5 rule evaluated at conversion."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        return y + torch.randn(y.shape[1])


def test_a_random_factory_is_not_drawn_at_conversion():
    """The conversion drew ``torch.randn(6)`` at conversion time, advancing the global RNG, and then refused its
    consumer. Now the factory is a host op: the spiking conversion draws exactly what tracing draws (the traced
    forward's one ``randn(6)``), as on w0-base. (An MVM conversion also initialises its affine layers, on w0-base
    too, so its RNG is compared through ``evaluate_shape_value`` in ``TestNoOtherCallIsEvaluated``.)"""
    model = native(RandomFactory)
    torch.manual_seed(5)
    quietly(trace_model, copy.deepcopy(model), (8,))
    traced = torch.get_rng_state()
    torch.manual_seed(5)
    flow = quietly(trace_and_convert, copy.deepcopy(model), (8,), "cpu", packaging=PACKAGINGS["spiking"]).flow
    assert torch.equal(traced, torch.get_rng_state()) and "randn" in _host_ops(flow)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_random_factory_converts_as_a_host_op(contract):
    """Refused before (``GenericOpArgumentError``, its only source folded away); both contracts convert it now."""
    flow = quietly(trace_and_convert, copy.deepcopy(native(RandomFactory)), (8,), "cpu",
                   packaging=PACKAGINGS[contract]).flow
    assert _host_ops(flow) == ["getattr_1", "randn", "add"]


# ----------------------------------------------------------------------------- operands that need not broadcast
class TwoActivationProduct(nn.Module):
    """A product of two activations (review_reverify/w_shape.py's AttnScale, and ``torch.bmm``)."""

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.q = nn.Linear(6, 6)
        self.k = nn.Linear(6, 6)

    def forward(self, x):
        q, k = self.q(x), self.k(x)
        if self.how == "attention_scores":
            return torch.softmax(q @ k.transpose(-2, -1) * q.size(-1) ** -0.5, dim=-1)
        return torch.bmm(q, k.transpose(1, 2))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["attention_scores", "bmm"])
def test_a_product_of_two_activations_is_exact_at_every_batch(how, contract):
    """(B, 5, 6) and (B, 6, 5) do not broadcast: the elementwise operand check refused both at the forward probe
    (w0-base by name, ``ConversionProbeError``; the round-5 tree anonymously, ``RuntimeError``)."""
    assert_converted_exact(native(lambda: TwoActivationProduct(how)), (5, 6), contract,
                           batches=tuple(range(1, 14)), kinds=("normal", "uniform"))
