"""Python's semantics of the scalar operators in a pure program (review round 10; restated in review round 14).

The round-9 tree computed the declared scalar operators (``operator``, ``math``, ``torch.sym_*``) on symbolic sizes with
torch's symbolic semantics, and differed from Python where a Boolean meets a number (``(B > 8) == 1`` was the constant
``False`` with no guard, ``True`` natively from batch 9: EqBoolTileShape folded the width 6, 12 natively), where a
decision's assumed type let sympy simplify (``2 + g - g`` of ``g = math.gcd(B, 6)``), where a power's result type
changes with the batch (``(8.5 - B) ** 0.5`` is complex from batch 9) and where IEEE double differs from real algebra
(``B * 1e307`` is ``inf`` from batch 18). Rounds 10 to 13 added a rule per difference.

[restated in review round 14: the owner's shape-proof domain replaces those rules. A batch-dependent scalar is folded
only through the toolchain's affine forms ``a*B + c`` (``symbolic/batch_domain``): a comparison of the batch with a
constant that changes with B, ``math.gcd``, a power, a float operation, every value outside the domain is opaque and
computed on the host at run time, and so is every node depending on it in a shape position (a layout it reaches is the
host op of its native method). So each form below is exact at every batch (or refused by name where the seeded traces
show its layout moves the batch), and its stage 2 is refused by name. The domain's own semantics are fuzzed against
Python in ``test_batch_domain.py``.]
"""

from __future__ import annotations

import math
import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mapping_utils import ReshapeMapper
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH, Opaque, evaluate, meaning
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .harness import CONTRACTS, assert_converted_exact, convert, native

EVERY_BATCH = tuple(range(1, 14))


def _gate(x):
    """``(B > 8) == 1``: ``False`` up to batch 8, ``True`` above (Python's ``bool`` is an ``int``)."""
    return (x.size(0) > 8) == 1


class Booleans(nn.Module):
    """The round-9 reviewer's witnesses (``r9_mods.py``), pure programs."""

    FORMS = {
        # the width of a host op from a Boolean meeting a number: (B, 6) up to batch 8, (B, 12) or (B, 5) above
        "eq_bool_tile_shape": lambda x, y: y * torch.tile(y, (1, 1 + _gate(x))).shape[1],
        "eq_bool_slice_shape": lambda x, y: y * y[:, :7 - (1 + _gate(x))].shape[1],
        "eq_bool_narrow_shape": lambda x, y: y * torch.narrow(y, 1, 0, 7 - (1 + _gate(x))).size(1),
        "eq_bool_repeat_interleave_shape": lambda x, y: y * torch.repeat_interleave(y, 1 + _gate(x), 1).shape[-1],
        "ne_bool_tile_shape": lambda x, y: y * torch.tile(y, (1, 3 - ((x.size(0) > 8) != 1))).shape[1],
        "eq_bool_tile_dim": lambda x, y: (lambda z: y * z.shape[-1] + z.size(1))(
            torch.tile(y, (1, 1 + (1 == (x.size(0) > 8))))),
        # no symbolic Boolean: a bool literal meeting a size (2 at batch 9 only)
        "int_eq_true_tile_shape": lambda x, y: y * torch.tile(
            y, (1, 1 + ((x.size(0) - 8) == True))).shape[1],  # noqa: E712
        # a per-sample view the Boolean decides: softmax over 12, then over 6
        "eq_bool_view": lambda x, y: F.softmax(y.view(x.size(0), 1 + _gate(x), -1), -1).flatten(1),
        "eq_bool_view_shape": lambda x, y: y * y.view(x.size(0), 1 + _gate(x), -1).shape[2],
        # the control: (B > 8) == True compares two Booleans and stays symbolic
        "eq_bool_control": lambda x, y: F.softmax(
            y.view(x.size(0), 1 + ((x.size(0) > 8) == True), -1), -1).flatten(1),  # noqa: E712
    }
    WIDTH = {"eq_bool_view": 12, "eq_bool_control": 12}

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, self.WIDTH.get(how, 6))

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


HOST_WIDTHS = [
    "eq_bool_narrow_shape", "eq_bool_repeat_interleave_shape", "eq_bool_slice_shape", "eq_bool_tile_dim",
    "eq_bool_tile_shape", "int_eq_true_tile_shape", "ne_bool_tile_shape",
]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", HOST_WIDTHS)
def test_a_width_from_a_boolean_meeting_a_number_is_a_run_time_host_op(how, contract):
    """Certified and wrong at batches 9 and 13 on the round-9 tree (rel 0.2 to 0.5; the int-eq-True form at batch 9
    only), refused by name on w0-base: exact at every batch with no literal, its width the host op's at run time."""
    flow = assert_converted_exact(native(lambda: Booleans(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_impurity is None
    first = "eq" if how == "int_eq_true_tile_shape" else "gt"
    assert flow.shape_opaque[:2] == (first, first)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["eq_bool_view", "eq_bool_view_shape", "eq_bool_control"])
def test_a_per_sample_view_a_boolean_meeting_a_number_decides_runs_on_the_host(how, contract):
    """Certified and wrong at batches 9 and 13 on the round-9 tree, on the round-8 tree and on w0-base; refused by name
    in rounds 10 to 13. [restated in review round 14: the gate is an opaque value; the view is the host op of its
    native method, exact at every batch]"""
    flow = assert_converted_exact(native(lambda: Booleans(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_opaque[:2] == ("gt", "gt")


def test_a_boolean_meeting_a_number_is_an_opaque_value_and_reaches_the_width():
    """[restated in review round 14: round 10 read ``(B > 8) == 1`` as a decision with its Python cases]"""
    gm = trace_model(native(lambda: Booleans("eq_bool_tile_shape")), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    assert proof.batch_decision(nodes["eq"]).node is nodes["gt"]
    assert proof.shape_decision(nodes["getitem"]).node is nodes["gt"]
    assert proof.static_proof and layouts.layout(nodes["tile"]) is None


# ----------------------------------------------------------------------------- the operands, unit by unit
_OPERATORS = [operator.eq, operator.ne, operator.add, operator.sub, operator.mul, operator.and_, operator.or_,
              operator.xor, operator.lt, operator.le, operator.gt, operator.ge, operator.floordiv, operator.mod,
              operator.truediv, math.gcd, torch.sym_max, torch.sym_min]
_OPERANDS = {  # name -> (operand of the batch, the other operand); "rev": the other operand first
    "gate_and_1": (lambda b: b > 8, 1), "gate_and_0": (lambda b: b > 8, 0), "gate_and_2.0": (lambda b: b > 8, 2.0),
    "size_and_true": (lambda b: b - 8, True), "size_and_false": (lambda b: 9 - b, False),
}


def _domain_operand(make):
    """``make`` (a comparison or a difference of the batch) computed in the domain, operator by operator."""
    class _B:
        def __gt__(self, k):
            return evaluate(operator.gt, [BATCH, k])

        def __sub__(self, k):
            return evaluate(operator.sub, [BATCH, k])

        def __rsub__(self, k):
            return evaluate(operator.sub, [k, BATCH])
    return make(_B())


@pytest.mark.parametrize("reverse", [False, True], ids=["first", "rev"])
@pytest.mark.parametrize("pattern", sorted(_OPERANDS))
@pytest.mark.parametrize("fn", _OPERATORS, ids=lambda fn: fn.__name__)
def test_every_value_the_domain_gives_a_boolean_meeting_a_number_is_pythons(fn, pattern, reverse):
    """The symfuzz property of the round-9 review: a scalar operator on a Boolean and a number. [restated in review
    round 14: round 10 ran it on torch's symbolic values through ``python_operands`` (deleted) at batches 2 to 20; the
    domain computes it now, and its value is ``Opaque`` (computed on the host at run time, as natively) or Python's
    value and exact type at every batch 1 to 64: ``B > 8`` is opaque, and so is every operation on it; ``(B - 8) ==
    True`` is opaque (true at batch 9 only); ``(B - 8) + True`` is ``B - 7``]"""
    make, other = _OPERANDS[pattern]
    mine = _domain_operand(make)
    operands = [other, mine] if reverse else [mine, other]
    try:
        value = evaluate(fn, operands)
    except (ArithmeticError, TypeError, ValueError):  # a batch-free call raising: Python's own exception
        return
    if isinstance(value, Opaque):
        return
    for b in range(1, 65):
        try:
            want = fn(*((other, make(b)) if reverse else (make(b), other)))
        except (ArithmeticError, TypeError, ValueError):
            continue
        got = meaning(value, b)
        assert type(got) is type(want) and got == want, (fn.__name__, pattern, reverse, b, got, want)


def test_the_gate_and_the_offset_of_the_table_are_the_domains_values():
    assert isinstance(_domain_operand(lambda b: b > 8), Opaque)
    assert evaluate(operator.add, [_domain_operand(lambda b: b - 8), True]) == evaluate(operator.sub, [BATCH, 7])


# ----------------------------------------------------------------------------- type-stable decisions
class Gcds(nn.Module):
    """The round-9 reviewer's controls: ``g = math.gcd(B, 6)`` is a decision (its guard ``Eq(B, 7)``) without cases,
    an ``int`` at every batch."""

    FORMS = {
        "gcd_cancel_view": lambda x, y, g: F.softmax(y.view(x.size(0), 2 + g - g, -1), -1).flatten(1),
        "gcd_scale_view": lambda x, y, g: F.softmax(y.view(x.size(0), (2 * g) // g, -1), -1).flatten(1),
        "gcd_rows_view": lambda x, y, g: F.softmax(y.view(x.size(0) * g // g, -1), -1),
    }
    WIDTH = {"gcd_cancel_view": 12, "gcd_scale_view": 12}

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, self.WIDTH.get(how, 6))

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x), math.gcd(x.size(0), 6))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(Gcds.FORMS))
def test_a_view_a_gcd_of_the_batch_reaches_runs_on_the_host(how, contract):
    """Refused by name on the round-9 tree, exact on w0-base and on the round-8 tree with static views. [restated in
    review round 14: ``math.gcd`` of the batch is an opaque value, so the view is the host op of its native method:
    exact at every batch, no static layout of it, no literal]"""
    flow = assert_converted_exact(native(lambda: Gcds(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())
    assert flow.shape_literals == {} and flow.shape_opaque[:2] == ("gcd", "gcd")


# ----------------------------------------------------------------------------- review round 11: pow's result type
def _complex_gap(g):
    """``|g*g - |g|^2|``: 0 for a real ``g``, ``2|g|^2`` for an imaginary one."""
    return math.floor(abs(g * g - abs(g) * abs(g)))


class Pows(nn.Module):
    """The round-10 reviewer's ``r10_mods.py`` (``S1``, ``S5``) and two controls of a dropped operand."""

    FORMS = {
        # g = (8.5 - B) ** 0.5 is complex from batch 9: the view's width is 1, then 2
        "complex_pow_view": lambda x, y: F.softmax(
            y.view(x.size(0), 1 + _complex_gap((8.5 - x.size(0)) ** 0.5), -1), -1).flatten(1),
        "int_pow_complex_view": lambda x, y: F.softmax(
            y.view(x.size(0), 1 + _complex_gap((8 - x.size(0)) ** 0.5), -1), -1).flatten(1),
        # the width 2 at every batch: the decision multiplied by 0, or an operand cancelled by itself
        "complex_pow_rows_view": lambda x, y: F.softmax(
            y.view(x.size(0), 2 + 0 * _complex_gap((8.5 - x.size(0)) ** 0.5), -1), -1).flatten(1),
        "floor_self_cancel": lambda x, y: (lambda f: F.softmax(y.view(x.size(0), 2 + (f - f), -1), -1).flatten(1))(
            math.floor(2 ** (9 - x.size(0)))),
        # ceil(g) - floor(g) is 0 for an int g, 1 at batch 10 (g = 0.5): w0-base was wrong by rel 0.39
        "ceil_floor_view": lambda x, y: (lambda g: F.softmax(
            y.view(x.size(0), 2 + (math.ceil(g) - math.floor(g)), -1), -1).flatten(1))(2 ** (9 - x.size(0))),
        # 0.0 * h with a float h: h * 1e308 is inf from batch 19 and 0.0 * inf is nan, so the width is 3, then 2
        "float_operand_view": lambda x, y: F.softmax(y.view(x.size(0), 2 + (0.0 * (math.sqrt(abs(
            (8.5 - x.size(0)) ** 0.5)) * 1e308) == 0.0), -1), -1).flatten(1),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 12)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


POW_FIRST_OPAQUE = {  # form -> the first opaque value: a float operation on the batch, or a power of it
    "complex_pow_view": ("sub", "sub"), "float_operand_view": ("sub", "sub"), "complex_pow_rows_view": ("sub", "sub"),
    "int_pow_complex_view": ("pow_1", "pow"), "ceil_floor_view": ("pow_1", "pow"),
    "floor_self_cancel": ("pow_1", "pow"),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(POW_FIRST_OPAQUE))
def test_a_view_a_power_of_the_batch_reaches_runs_on_the_host(how, contract):
    """ComplexPowView and IntPowComplexView were certified and wrong at batch 9 on the round-10 tree (rel 0.25 to 1.0,
    stage 2 passing), FloatOperandView kept a width of 3 there (2 natively from batch 19), w0-base was wrong on all
    three; ComplexPowRowsView and FloorSelfCancel were exact on w0-base with static views. [restated in review round
    14: each was refused by name or kept its static view by a rule of rounds 10 to 13; a power and a float of the
    batch are opaque now, so every view here is the host op of its native method: exact at every batch at which the
    native model runs, and raising as natively at the others]"""
    model = native(lambda: Pows(how))
    defined, undefined = _runs_natively(model, (*EVERY_BATCH, 19, 20))
    flow = assert_converted_exact(model, (8,), contract, batches=defined, kinds=("normal", "uniform"))
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())
    assert flow.shape_opaque[:2] == POW_FIRST_OPAQUE[how] and flow.shape_impurity is None
    for batch in undefined:  # the native view raises (its width does not divide 12): so does the host op
        with pytest.raises(RuntimeError) as excinfo:
            flow(torch.randn(batch, 8, dtype=torch.float64))
        assert "is invalid for input of size" in str(excinfo.value.__cause__ or excinfo.value)


def _runs_natively(model, batches):
    """The batches at which ``model`` runs, and those at which it raises."""
    def runs(batch):
        try:
            with torch.no_grad():
                model(torch.randn(batch, 8, dtype=torch.float64))
            return True
        except RuntimeError:
            return False
    defined = tuple(b for b in batches if runs(b))
    return defined, tuple(b for b in batches if b not in defined)


# ----------------------------------------------------------------------------- review round 12: float arithmetic
def _overflow(b):
    """``B * 1e307 - B * 1e307 == 0``: sympy's real algebra gives ``True``; in IEEE ``B * 1e307`` is ``inf`` from batch
    18 and ``inf - inf`` is ``nan``, so it is ``False`` there."""
    return (b * 1e307 - b * 1e307) == 0


def _absorb(b):
    """``(B + 1e16) - 1e16 == B``: sympy gives ``True``; IEEE rounds ``B + 1e16`` to an even integer, so it is
    ``False`` at every odd batch."""
    return ((b + 1e16) - 1e16) == b


def _nan_product(b):
    """``0.0 * (B * 1e308) == 0.0``: sympy's product is 0; ``B * 1e308`` is ``inf`` from batch 2, ``0.0 * inf`` is
    ``nan``."""
    return 0.0 * (b * 1e308) == 0.0


def _rounding(b):
    """``B / 3 * 3 == B``: sympy keeps an expression of B; IEEE is ``False`` first at ``2 ** 53 + 1``."""
    return b / 3 * 3 == b


FLOAT_CONDITIONS = {"overflow": _overflow, "absorb": _absorb, "nan_product": _nan_product, "rounding": _rounding}


class Floats(nn.Module):
    """Pure programs that compute a width with float arithmetic on the batch size (the round-11 open issue,
    ``OverflowLateTile``: certified and wrong at batches 18 and 20 on the round-10 and round-11 trees)."""

    def __init__(self, how: str, condition: str):
        super().__init__()
        self.how, self.condition = how, FLOAT_CONDITIONS[condition]
        self.fc = nn.Linear(8, 12 if how == "view" else 6)

    def forward(self, x):
        y, b = self.fc(x), x.size(0)
        w = 1 + self.condition(b)
        if self.how == "tile":
            return y * torch.tile(y, (1, w)).shape[1]
        if self.how == "view":
            return F.softmax(y.view(b, w, -1), -1).flatten(1)
        return y * w  # "scale": the decision reaches a value, never a shape


FLOAT_BATCHES = (*EVERY_BATCH, 17, 18, 20)


FLOAT_FIRST = {"overflow": ("mul", "mul"), "absorb": ("add", "add"), "nan_product": ("mul", "mul"),
               "rounding": ("truediv", "truediv")}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("condition", sorted(FLOAT_CONDITIONS))
@pytest.mark.parametrize("how", ["scale", "tile"])
def test_a_value_from_float_arithmetic_on_the_batch_is_a_run_time_host_value(how, condition, contract):
    """``OverflowLateTile`` (``tile`` of ``overflow``) folded the width 12 on the round-11 tree, 6 natively from batch
    18. A float operation on the batch is opaque: its width is the host op's at run time, the converted model is exact
    at every batch (17, 18 and 20 included), and nothing is folded."""
    flow = assert_converted_exact(native(lambda: Floats(how, condition)), (8,), contract, batches=FLOAT_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_impurity is None
    assert flow.shape_opaque[:2] == FLOAT_FIRST[condition]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("condition", ["overflow", "rounding"])
def test_a_per_sample_view_float_arithmetic_on_the_batch_reaches_runs_on_the_host(condition, contract):
    """``OverflowLateView``: certified and wrong at batches 18 and 20 on the round-11 tree, wrong on w0-base, refused by
    name in rounds 12 and 13. [restated in review round 14: the view is the host op of its native method, exact at
    every batch]"""
    flow = assert_converted_exact(native(lambda: Floats("view", condition)), (8,), contract, batches=FLOAT_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_opaque[:2] == FLOAT_FIRST[condition]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("condition", ["absorb", "nan_product"])
def test_a_view_whose_width_the_seeded_traces_show_changing_is_refused_by_name(condition, contract):
    """The width changes within the seeded batches 1 to 3: the traces show the layout moving the batch, a fact."""
    with pytest.raises(BatchAxisLayoutError) as excinfo:
        convert(native(lambda: Floats("view", condition)), (8,), contract)
    assert str(excinfo.value).startswith(
        "view (FX node 'view', used by 'softmax'): in its output, an axis other than axis 0 carries the batch")


def test_a_float_operation_on_the_batch_is_the_opaque_value_of_what_it_reaches():
    """In the pass: ``B * 1e307`` (``mul``) is opaque; every value computed from it is reached by a decision, and so
    is the width; the Linear before it keeps its proof."""
    gm = trace_model(native(lambda: Floats("tile", "overflow")), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    first = proof.batch_decision(nodes["mul"])
    assert first is not None and first.node is nodes["mul"] and first.kind == "opaque"
    for name in ("sub", "eq", "add", "tile", "getitem"):
        assert proof.batch_decision(nodes[name]) is not None, name
    assert proof.shape_decision(nodes["getitem"]) is not None and layouts.layout(nodes["tile"]) is None
    assert layouts.layout(nodes["fc"]) is not None


class FloatControls(nn.Module):
    """Float arithmetic the round-11 tree computes exactly: on static sizes (Python's own floats, folded as natively),
    and on the batch where no shape reads it (``batch_ratio``)."""

    FORMS = {
        "static_floor": lambda x, y: y * torch.tile(y, (1, math.floor(y.size(1) * 0.5))).shape[1],
        "static_scale": lambda x, y: y * (y.size(1) ** -0.5),
        "batch_ratio": lambda x, y: y * (x.size(0) / 4),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


FLOAT_CONTROL_LITERALS = {  # form -> the literals folded (as on the round-11 tree)
    "static_floor": {"size": ("size", "6"), "mul": ("mul", "3.0"), "floor": ("floor", "3"),
                     "getitem": ("getitem", "18")},
    "static_scale": {"size": ("size", "6"), "pow_1": ("pow", "0.408248290463863")},
    "batch_ratio": {},
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(FLOAT_CONTROL_LITERALS))
def test_float_arithmetic_on_static_sizes_keeps_its_fold_and_a_batch_ratio_its_program(how, contract):
    """Exact on the round-11 tree with these literals; a float of static sizes is Python's own value, and an opaque
    value that reaches no shape changes nothing a conversion computes. [restated in review round 14: the batch ratio
    is named on the flow as the first opaque value]"""
    flow = assert_converted_exact(native(lambda: FloatControls(how)), (8,), contract, batches=FLOAT_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == FLOAT_CONTROL_LITERALS[how] and flow.shape_impurity is None
    assert (flow.shape_opaque is None) == (how != "batch_ratio")


# ----------------------------------------------------------------------------- review round 12: a kernel's float shape
_K = 2 ** 50  # B * 2**50 reaches 2**53 at batch 8: a double no longer holds B * 2**50 + 1


class Kernels(nn.Module):
    """A declared kernel that computes its result's shape with floats from a symbolic size (found in review round 12):
    ``torch.arange(B * 2**50, B * 2**50 + 1, 0.5)`` has 2 elements in the pass for every B (torch's meta function
    subtracts the operands exactly), and natively 2 up to batch 7, 0 from batch 8 (the eager kernel converts each one
    to a double first). Certified and wrong from batch 8 on the round-11 tree (``ArangeLateTile``, ``ArangeLateView``);
    w0-base refuses the tile form at its probe and is wrong on the view form."""

    FORMS = {
        "arange_step_tile": lambda x, y, b: y * torch.tile(
            y, (1, 1 + torch.arange(b * _K, b * _K + 1, 0.5).size(0))).shape[1],
        "arange_dtype_tile": lambda x, y, b: y * torch.tile(
            y, (1, 2 + torch.arange(b * _K, b * _K + 1, dtype=y.dtype).size(0))).shape[1],
        "arange_step_view": lambda x, y, b: F.softmax(
            y.view(b, 1 + torch.arange(b * _K, b * _K + 1, 0.5).size(0), -1), -1).flatten(1),
        # controls: the eager int64 kernel is exact (1 element at every batch); a float that only fills keeps its
        # program; a shape that keeps B is never folded
        "arange_int_tile": lambda x, y, b: y * torch.tile(
            y, (1, 1 + torch.arange(b * _K, b * _K + 1).size(0))).shape[1],
        "full_fill": lambda x, y, b: y + torch.full((6,), b * 0.5, dtype=y.dtype),
        "arange_batch_scale": lambda x, y, b: y * torch.arange(0, b, 0.5).size(0),
    }
    WIDTH = {"arange_step_view": 18}

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, self.WIDTH.get(how, 6))

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x), x.size(0))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["arange_dtype_tile", "arange_step_tile", "arange_step_view"])
def test_a_width_from_a_floating_arange_of_the_batch_is_a_run_time_host_value(how, contract):
    """Certified and wrong from batch 8 on the round-11 tree (rel 2.0; the literals ``size_1 = 2`` and ``getitem``);
    ``ArangeLateView`` refused by name in rounds 12 and 13. [restated in review round 14: a floating ``arange`` has no
    declared batch rule (its length is computed in doubles), so its size is opaque: every width and view it reaches
    runs on the host, exact at every batch]"""
    flow = assert_converted_exact(native(lambda: Kernels(how)), (8,), contract, batches=FLOAT_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_impurity is None
    assert flow.shape_opaque[:2] == ("arange", "arange")


KERNEL_CONTROL_LITERALS = {  # form -> the literals folded, as on the round-11 tree
    "arange_batch_scale": {},
    "arange_int_tile": {"size_1": ("size", "1"), "add_1": ("add", "2"), "getitem": ("getitem", "12")},
    "full_fill": {},
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(KERNEL_CONTROL_LITERALS))
def test_an_exact_kernel_or_a_float_that_reaches_no_shape_keeps_its_program(how, contract):
    """The integer ``arange``'s declared rule states its length ``end - start`` in the domain (1 at every batch); a
    float fill value and a shape that keeps B are no fold: their round-11 programs are kept."""
    flow = assert_converted_exact(native(lambda: Kernels(how)), (8,), contract, batches=FLOAT_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == KERNEL_CONTROL_LITERALS[how] and flow.shape_impurity is None


def test_a_floating_aranges_size_is_the_opaque_value_of_the_kernels_node():
    gm = trace_model(native(lambda: Kernels("arange_step_tile")), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    decision = proof.batch_decision(nodes["arange"])
    assert decision is not None and decision.node is nodes["arange"] and decision.kind == "opaque"
    assert proof.batch_decision(nodes["tile"]) is decision and layouts.layout(nodes["arange"]) is None
