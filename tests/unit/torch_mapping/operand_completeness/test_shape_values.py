"""Shape queries (``size``, ``dim``, ``shape``, ``ndim``) are values, never the tensor they query.

A non-batch shape value is known at conversion time and becomes a literal of the op
it feeds (exact in the converted model and in the IR); a batch-dependent one is
evaluated at run time by a host op, as the native model evaluates it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.shape_value_mixin import (
    BATCH_SIZE,
    depends_on_batch,
    evaluate_shape_value,
)

from .harness import (
    CONTRACTS,
    PACKAGINGS,
    assert_converted_exact,
    assert_value_program_exact,
    convert,
    native,
    quietly,
)


class ClipToWidth(nn.Module):
    """Clip to +-feature width with ``size(1)`` (the review's witness: converted as clamp(y, -y, y) before)."""

    def __init__(self, query="size"):
        super().__init__()
        self.query = query
        self.fc = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        y = self.fc(x)
        w = y.size(1) if self.query == "size" else y.shape[1]
        return self.fc2(torch.clamp(y, min=-w, max=w))


class ScaleBySize(nn.Module):
    def __init__(self, how):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        if self.how == "size1":
            return y * y.size(1)
        if self.how == "size_neg":
            return y * y.size(-1)
        if self.how == "size_keyword":
            return y * y.size(dim=1)
        if self.how == "dim":
            return y * y.dim()
        if self.how == "ndim":
            return y * y.ndim
        if self.how == "shape_slice":
            return y * y.shape[1:][-1]
        raise ValueError(self.how)


class AttentionScale(nn.Module):
    """``q * q.size(-1) ** -0.5`` on (B, N, D) tokens (attention scaling; NaN before)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 16)
        self.h = nn.Linear(16, 4)

    def forward(self, x):
        q = self.q(x)
        return self.h((q * q.size(-1) ** -0.5).mean(dim=1))


class BatchMean(nn.Module):
    """Divides by the batch size: a batch-dependent shape value, evaluated at run time."""

    def __init__(self, query="size"):
        super().__init__()
        self.query = query
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        b = y.size(0) if self.query == "size" else y.shape[0]
        return y / b


class FlattenBySize(nn.Module):
    """``x.view(x.size(0), -1)``: the shape value only shapes a view and never runs."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x):
        y = torch.relu(self.c(x))
        return self.fc(y.view(y.size(0), -1))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("query", ["size", "shape"])
def test_clip_to_width_is_exact_beyond_the_width(contract, query):
    model = native(lambda: ClipToWidth(query))
    assert_converted_exact(model, (8,), contract, kinds=("wide", "normal", "zero"))


@pytest.mark.parametrize("query", ["size", "shape"])
def test_clip_to_width_value_program_is_exact(query):
    assert_value_program_exact(native(lambda: ClipToWidth(query)), (8,), kinds=("wide", "normal"))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["size1", "size_neg", "size_keyword", "dim", "ndim", "shape_slice"])
def test_static_shape_value_scales_exactly(contract, how):
    assert_converted_exact(native(lambda: ScaleBySize(how)), (8,), contract)


@pytest.mark.parametrize("how", ["size1", "dim"])
def test_static_shape_value_program_is_exact(how):
    assert_value_program_exact(native(lambda: ScaleBySize(how)), (8,))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_attention_scaling_by_head_width_is_exact(contract):
    assert_converted_exact(native(AttentionScale), (5, 8), contract)


def test_static_shape_value_is_a_literal_of_the_op_it_feeds():
    flow = convert(native(lambda: ScaleBySize("size1")), (8,), "spiking")
    ops = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper)]
    muls = [m for m in ops if m.name == "mul"]
    assert len(muls) == 1 and len(muls[0].sources) == 1
    assert muls[0].module.extra_args == (6,)
    assert not [m for m in ops if m.name == "size"]


@pytest.mark.parametrize("query", ["size", "shape"])
def test_batch_dependent_shape_value_is_evaluated_at_run_time(query):
    """Spiking keeps the converted model exact at every batch (the IR cannot express it; MVM's stage 2 refuses)."""
    assert_converted_exact(native(lambda: BatchMean(query)), (8,), "spiking", batches=(1, 2, 3, 5))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_view_by_batch_size_keeps_its_mappers(contract):
    flow = assert_converted_exact(native(FlattenBySize), (3, 4, 4), contract)
    names = [getattr(m, "name", None) for m in flow.get_mapper_repr().execution_order()]
    assert "size" not in names


class TestShapeValueEvaluation:
    def test_batch_axis_is_symbolic(self):
        assert depends_on_batch(BATCH_SIZE)
        assert depends_on_batch((BATCH_SIZE, 4))
        assert not depends_on_batch((4, 8))
        assert not depends_on_batch(6)

    def test_static_parts_of_a_shape_evaluate(self):
        import operator

        shape = (BATCH_SIZE, 4, 8)
        assert evaluate_shape_value(operator.getitem, (shape, 1), {}) == (True, 4)
        assert evaluate_shape_value(operator.getitem, (shape, slice(1, None)), {}) == (True, (4, 8))
        assert evaluate_shape_value(len, (shape,), {}) == (True, 3)
        assert evaluate_shape_value(operator.neg, (6,), {}) == (True, -6)

    def test_arithmetic_on_the_batch_axis_is_batch_dependent(self):
        import operator

        ok, value = evaluate_shape_value(operator.mul, (BATCH_SIZE, 2), {})
        assert ok and depends_on_batch(value)
        ok, value = evaluate_shape_value(operator.getitem, ((BATCH_SIZE, 4), 0), {})
        assert ok and depends_on_batch(value)

    def test_tensor_results_are_not_shape_values(self):
        assert evaluate_shape_value(torch.zeros, (3,), {}) == (False, None)


# ----------------------------------------------------------------------------- comparisons on the batch axis
class BatchComparison(nn.Module):
    """A comparison or coercion on the batch axis: known only at run time (folded to a constant before)."""

    def __init__(self, how):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        if self.how == "size_eq":
            return y * (x.size(0) == 1) + y
        if self.how == "shape_index_eq":
            return y * (x.shape[0] == 1) + y
        if self.how == "size_ne":
            return y * (x.size(0) != 2) + y
        if self.how == "shape_tuple_eq":
            return y * (x.shape == (1, 8)) + y
        raise ValueError(self.how)


BATCH_COMPARISONS = ["size_eq", "shape_index_eq", "size_ne", "shape_tuple_eq"]
COMPARISON_BATCHES = (1, 2, 3, 6)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", BATCH_COMPARISONS)
def test_a_comparison_on_the_batch_axis_is_evaluated_at_run_time(how, contract):
    assert_converted_exact(native(lambda: BatchComparison(how)), (8,), contract, batches=COMPARISON_BATCHES)


@pytest.mark.parametrize("how", BATCH_COMPARISONS)
def test_a_comparison_on_the_batch_axis_is_refused_by_name_in_the_value_program(how):
    """The IR carries tensors only: the pipeline's MVM conversion refuses the run-time shape value by name.
    [restated in review round 14: ``NonTensorHostValueError``; ``B == 1`` (``B != 2``) changes with the batch, an
    opaque value, so stage 2 refuses the program naming it (``OpaqueBatchValueError``)]"""
    model = native(lambda: BatchComparison(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    name = "ne" if how == "size_ne" else "eq"
    assert "OpaqueBatchValueError" in str(excinfo.value) and f"{name} (FX node {name!r})" in str(excinfo.value)


# The w0-base mapper structure of the two shape forms that converted exactly there (runtime host ops), kept.
BASE_HOST_OP_STRUCTURE = {
    "shape_index_eq": ["fc", "getattr_1", "getitem", "eq", "mul", "add"],
    "shape_tuple_eq": ["fc", "getattr_1", "eq", "mul", "add"],
}


@pytest.mark.parametrize("how", sorted(BASE_HOST_OP_STRUCTURE))
def test_shape_comparisons_keep_the_base_host_op_structure(how):
    flow = convert(native(lambda: BatchComparison(how)), (8,), "spiking")
    ops = [m.name for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper)]
    assert ops == BASE_HOST_OP_STRUCTURE[how]


class TestBatchSymbolIsOpaque:
    @pytest.mark.parametrize("observe", [
        lambda b: b == 1, lambda b: b != 2, lambda b: b < 3, lambda b: b <= 3, lambda b: b > 3, lambda b: b >= 3,
        lambda b: 1 == b, lambda b: bool(b), lambda b: int(b), lambda b: float(b), lambda b: [0, 1][b],
        lambda b: hash(b), lambda b: (b, 8) == (1, 8), lambda b: 8 in (b, 8),
    ])
    def test_every_value_observation_raises(self, observe):
        with pytest.raises(TypeError):
            observe(BATCH_SIZE)

    @pytest.mark.parametrize("compare", [
        lambda s: s == (1, 8), lambda s: s != (2, 8), lambda s: s[0] == 1, lambda s: s < (2, 8),
    ])
    def test_a_comparison_involving_the_batch_axis_is_batch_dependent(self, compare):
        ok, value = evaluate_shape_value(compare, ((BATCH_SIZE, 8),), {})
        assert ok and depends_on_batch(value)

    def test_identity_survives_copy(self):
        import copy
        import pickle

        assert copy.deepcopy(BATCH_SIZE) is BATCH_SIZE
        assert copy.copy(BATCH_SIZE) is BATCH_SIZE
        assert pickle.loads(pickle.dumps(BATCH_SIZE)) is BATCH_SIZE


# ----------------------------------------------------------------------------- shapes of parameters and buffers
class ParameterShape(nn.Module):
    """A parameter's leading axis is not the batch axis: its size is static."""

    def __init__(self, how):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)
        self.p = nn.Parameter(torch.randn(4, 6))
        self.register_buffer("table", torch.randn(3, 6))

    def forward(self, x):
        y = self.fc(x)
        if self.how == "param_size0":
            return y * self.p.size(0)
        if self.how == "param_shape0":
            return y * self.p.shape[0]
        if self.how == "buffer_size0":
            return y + self.table.size(0)
        if self.how == "derived_size0":
            return y * (2 * self.p).size(0)
        raise ValueError(self.how)


PARAMETER_SHAPES = ["param_size0", "param_shape0", "buffer_size0", "derived_size0"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", PARAMETER_SHAPES)
def test_a_parameter_size_is_static_and_exact(how, contract):
    flow = assert_converted_exact(native(lambda: ParameterShape(how)), (8,), contract)
    names = [getattr(m, "name", None) for m in flow.get_mapper_repr().execution_order()]
    assert "size" not in names and "getattr_1" not in names


@pytest.mark.parametrize("how", ["param_size0", "param_shape0"])
def test_a_parameter_size_value_program_is_exact(how):
    assert_value_program_exact(native(lambda: ParameterShape(how)), (8,))


# ----------------------------------------------------------------------------- declared IR change (review round 2)
class ShapeScale(nn.Module):
    """``y * y.shape[-1] ** -0.5``: exact on w0-base through getattr/getitem/pow host ops, now a literal."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.h = nn.Linear(6, 3)

    def forward(self, x):
        y = self.fc(x)
        return self.h(y * y.shape[-1] ** -0.5)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_static_shape_chain_is_one_literal_and_bit_identical_to_native(contract):
    """Declared change: three host ops (getattr, getitem, pow) become the literal 6 ** -0.5 of the mul."""
    model = native(ShapeScale)
    flow = convert(model, (5, 6), contract)
    ops = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper)]
    (mul,) = [m for m in ops if m.name == "mul"]
    assert mul.module.extra_args == (6 ** -0.5,) and len(mul.sources) == 1
    assert not {"getattr_1", "getitem", "pow_1"} & {m.name for m in ops}
    for batch in (1, 2, 3, 5):
        x = torch.randn(batch, 5, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            assert torch.equal(flow(x), model(x)), f"batch {batch}: values differ from the native model"


def test_static_shape_chain_value_program_is_exact():
    """On w0-base the chain's getattr returned a Size, which no IR lowering accepts."""
    assert_value_program_exact(native(ShapeScale), (5, 6))


# ----------------------------------------------------------------------------- declared IR change, every form (review round 3)
class StaticShapeForm(nn.Module):
    """The forms of a static shape chain that converted exactly on w0-base through ``getattr``/``getitem`` host ops."""

    def __init__(self, form):
        super().__init__()
        self.form = form
        self.fc = nn.Linear(8, 6)
        self.c = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4 * 5 * 5, 3)

    def forward(self, x):
        if self.form == "getitem":
            y = self.fc(x)
            return y * y.shape[1]
        if self.form == "slice_comparison":
            y = self.fc(x)
            return y * (x.shape[1:] == (8,)) + y
        if self.form == "tuple_unpacking":
            y = self.c(x)
            n, c, h, w = y.shape
            return self.head(y.flatten(1) * (c * h * w) ** -0.5)
        raise ValueError(self.form)


# form -> (input shape, host ops after the change, their w0-base host-op chain, the literal folded into the op)
STATIC_SHAPE_FORMS = {
    "getitem": ((8,), ["mul"], ["getattr_1", "getitem", "mul"], ("mul", (6,))),
    "slice_comparison": ((8,), ["mul", "add"], ["getattr_1", "getitem", "eq", "mul", "add"], ("mul", (True,))),
    "tuple_unpacking": ((3, 5, 5), ["mul_2"],
                        ["getattr_1", "getitem_1", "getitem_2", "mul", "getitem_3", "mul_1", "pow_1", "mul_2"],
                        ("mul_2", (100 ** -0.5,))),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("form", sorted(STATIC_SHAPE_FORMS))
def test_every_static_shape_form_is_one_literal_and_bit_identical_to_native(form, contract):
    """Declared change of the class: the chain's host ops (``base``) become one literal of the op they feed."""
    shape, after, base, (op, literal) = STATIC_SHAPE_FORMS[form]
    model = native(lambda: StaticShapeForm(form))
    flow = convert(model, shape, contract)
    ops = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper)]
    names = [m.name for m in ops if m.name not in ("fc", "c", "head")]
    assert names == after and set(base) - set(after) and not set(names) - set(base)
    (fed,) = [m for m in ops if m.name == op]
    assert fed.module.extra_args == literal and len(fed.sources) == 1
    for batch in COMPARISON_BATCHES:
        x = torch.randn(batch, *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            assert torch.equal(flow(x), model(x)), f"{form}, batch {batch}: values differ from the native model"


@pytest.mark.parametrize("form", sorted(STATIC_SHAPE_FORMS))
def test_every_static_shape_form_value_program_is_exact(form):
    """On w0-base every form's ``getattr`` host op returned a ``Size``, which no IR lowering accepts."""
    assert_value_program_exact(native(lambda: StaticShapeForm(form)), STATIC_SHAPE_FORMS[form][0])
