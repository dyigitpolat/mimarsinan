"""Integer floor division and modulo on the batch size (review round 13; restated in review round 14).

Torch's symbolic integers are sympy expressions, and torch 2.11 rewrites two integer forms under assumptions Python's
integers need not satisfy (``torch/utils/_sympy/functions.py``):

- ``FloorDiv(FloorDiv(a, b), c)`` becomes ``FloorDiv(a, b * c)``, which holds for a positive ``c`` only:
  ``(B // -2) // -3`` becomes ``B // 6`` (Python floors each step: 1 against 0 at B = 5, 11, 17, ...), and
  ``(B // 2) // (13 - 2 * B)`` becomes ``B // (26 - 4 * B)``;
- ``Mod`` and ``PythonMod`` return 0 when sympy calls ``p / q`` an integer, and sympy 1.14 calls
  ``(3 + B // 7) * 2**B / 6`` one: ``((3 + B // 7) << B) % 6`` is 0 for every B, 2 natively at batch 7.

The round-12 tree folded the widths these forms compute and kept the per-sample views they decide; each such program
was pure, held a static shape proof and was certified, wrong at the batches above (the reviewer's ``r12_floor.py``).
w0-base refused the tile forms at its probe and was wrong on the views. Round 13 read such a division or modulo as an
integer decision (``symbolic/arithmetic/integer_rewrites.py``).

[restated in review round 14: the owner's shape-proof domain (``symbolic/batch_domain``) computes every
batch-dependent integer itself, as ``a*B + c``; no value of torch's symbolic engine is read back for a fold. A floor
division or modulo of ``a*B + c`` folds only by a positive literal ``k`` that divides ``a`` (``(a*B + c) // k = (a//k)*B
+ c//k``, ``(a*B + c) % k = c % k``); every other one (``B // -2``, ``B // 2``, ``B % 4``, a batch-dependent divisor) is
opaque, computed on the host at run time, with every node depending on it in a shape position: a width it reaches is
the host op's, a per-sample view it reaches is the host op of its native method, so each witness below is exact at
batches 1 to 64 and its stage 2 is refused by name. The controls lose their folds and static views (named below) and
stay exact. The fuzz is restated against the domain, from batch 1.]
"""

from __future__ import annotations

import operator
import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ReshapeMapper
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH, Affine, Opaque, evaluate, meaning
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .harness import CONTRACTS, PACKAGINGS, assert_converted_exact, native, quietly
from .test_batch_domain import _domain, _Raised, _same

ALL_BATCHES = tuple(range(1, 65))


class Integers(nn.Module):
    """The round-12 reviewer's witnesses (``r12_floor.py``) and the controls, pure programs."""

    WIDTH = {"floor_neg_const_view": 12, "floor_nest_view": 12, "shift_mod_view": 12, "floor_half_view": 12,
             "floor_chain_view": 12, "mod_four_view": 12}

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, self.WIDTH.get(how, 6))

    def forward(self, x):
        y, b = self.fc(x), x.size(0)
        if self.how == "floor_neg_const_tile":  # 12 folded on the round-12 tree, 18 natively at B = 5, 11, 17, ...
            return y * torch.tile(y, (1, 2 + (b // -2) // -3 - b // 6)).shape[1]
        if self.how in ("floor_nest_tile", "floor_nest_wide"):  # natively 6 at batch 7 (33, 37, 41, 61 for wide)
            c = 13 - 2 * b if self.how == "floor_nest_tile" else 31 - b
            e = (b // 2) // c == b // (2 * c)
            return y * torch.tile(y, (1, 1 + e)).shape[1]
        if self.how == "shift_mod_tile":  # 12 folded, 24 natively at batch 7 (2 + 2)
            return y * torch.tile(y, (1, 2 + ((3 + b // 7) << b) % 6)).shape[1]
        if self.how == "floor_neg_const_view":
            return F.softmax(y.view(b, 2 + 2 * ((b // -2) // -3 - b // 6), -1), -1).flatten(1)
        if self.how == "floor_nest_view":
            c = 13 - 2 * b
            return F.softmax(y.view(b, 2 + (b // 2) // c - b // (2 * c), -1), -1).flatten(1)
        if self.how == "shift_mod_view":
            return F.softmax(y.view(b, 2 + ((3 + b // 7) << b) % 6, -1), -1).flatten(1)
        # controls: B // 2, (B - 10) // 2 // 3 and B % 4 keep torch's value; each pair cancels, so the width is 2
        if self.how == "floor_half_view":
            return F.softmax(y.view(b, 2 + b // 2 - b // 2, -1), -1).flatten(1)
        if self.how == "floor_chain_view":
            return F.softmax(y.view(b, 2 + (b - 10) // 2 // 3 - (b - 10) // 2 // 3, -1), -1).flatten(1)
        if self.how == "mod_four_view":
            return F.softmax(y.view(b, 2 + b % 4 - b % 4, -1), -1).flatten(1)
        if self.how == "floor_half_tile":
            return y * torch.tile(y, (1, 1 + (b // 2 == b // 2))).shape[1]
        if self.how == "floor_chain_tile":
            return y * torch.tile(y, (1, 1 + ((b - 10) // 2 // 3 == (b - 10) // 2 // 3))).shape[1]
        if self.how == "mod_four_tile":
            return y * torch.tile(y, (1, 2 + b % 4 - b % 4)).shape[1]
        raise ValueError(self.how)


WITNESS_TILES = ["floor_neg_const_tile", "floor_nest_tile", "floor_nest_wide", "shift_mod_tile"]
WITNESS_VIEWS = ["floor_neg_const_view", "floor_nest_view", "shift_mod_view"]
FIRST_OPAQUE = {how: "floordiv" for how in WITNESS_TILES + WITNESS_VIEWS}  # B // -2, B // 2, B // 7


def _batches(how):
    return [b for b in ALL_BATCHES if how != "floor_nest_wide" or b != 31]  # natively a division by zero at 31


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", WITNESS_TILES)
def test_a_width_an_integer_rewrite_on_the_batch_computes_is_a_run_time_host_value(how, contract):
    """Certified and wrong on the round-12 tree (FloorNegConstTile at 40 cells, B = 5 mod 6, rel 0.33; FloorNestTile
    at batch 7; FloorNestWide at 33, 37, 41 and 61; ShiftModTile at 42 batches from 7, rel 0.5 to 0.67): the first
    division is opaque, so the width is the host op's at run time, exact at batches 1 to 64, and nothing is folded."""
    flow = assert_converted_exact(native(lambda: Integers(how)), (8,), contract, batches=_batches(how),
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_impurity is None
    assert flow.shape_opaque[:2] == (FIRST_OPAQUE[how], FIRST_OPAQUE[how])


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", WITNESS_VIEWS)
def test_a_per_sample_view_an_integer_rewrite_decides_runs_on_the_host(how, contract):
    """Certified and wrong on the round-12 tree with stage 2 passing (FloorNegConstView at B = 5 mod 6, rel 0.25 to 1.0;
    FloorNestView at batch 7; ShiftModView at 42 batches from 7), wrong on w0-base too; refused by name on the round-13
    tree. [restated in review round 14: the view is the host op of its native method, exact at batches 1 to 64]"""
    flow = assert_converted_exact(native(lambda: Integers(how)), (8,), contract, batches=ALL_BATCHES,
                                  kinds=("normal", "uniform"))
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())
    assert flow.shape_opaque[:2] == (FIRST_OPAQUE[how], FIRST_OPAQUE[how])


@pytest.mark.parametrize("how", ["floor_neg_const_tile", "floor_nest_view", "shift_mod_view", "mod_four_tile"])
def test_stage_2_of_a_program_with_an_opaque_division_is_refused_by_name(how):
    name = "mod" if how == "mod_four_tile" else "floordiv"
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, native(lambda: Integers(how)), (8,), 10, packaging=PACKAGINGS["mvm"],
                value_program_stage=True)
    message = str(excinfo.value)
    assert f"OpaqueBatchValueError: {name} (FX node {name!r}): a value outside the affine batch domain ({name} by " \
           in message, message


CONTROL_OPAQUE = {  # form -> the first opaque value; the round-12 tree folded the tile widths 12 (``getitem``)
    "floor_half_tile": "floordiv", "floor_chain_tile": "floordiv", "mod_four_tile": "mod",
    "floor_half_view": "floordiv", "floor_chain_view": "floordiv", "mod_four_view": "mod",
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(CONTROL_OPAQUE))
def test_a_division_outside_the_domain_cancelled_by_itself_is_a_run_time_host_value(how, contract):
    """``B // 2``, ``(B - 10) // 2 // 3`` and ``B % 4``, each cancelled by itself. [restated in review round 14: on the
    round-12 and round-13 trees torch's value was kept, the tile widths folded to 12 (``getitem``) and the views kept
    their static layouts; the divisor does not divide the batch coefficient 1, so each is opaque now: no fold, no
    static view, exact at batches 1 to 64]"""
    flow = assert_converted_exact(native(lambda: Integers(how)), (8,), contract, batches=ALL_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_impurity is None
    assert flow.shape_opaque[:2] == (CONTROL_OPAQUE[how], CONTROL_OPAQUE[how])
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())


class DividingForms(nn.Module):
    """A division and a modulo inside the domain: ``(6B + 4) // 2 - 3B`` is 2 and ``(4B + 6) % 4`` is 2 at every B."""

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 12)

    def forward(self, x):
        y, b = self.fc(x), x.size(0)
        w = (6 * b + 4) // 2 - 3 * b if self.how.startswith("floordiv") else (4 * b + 6) % 4
        if self.how.endswith("tile"):
            return y * torch.tile(y, (1, w)).shape[1]
        return F.softmax(y.view(b, w, -1), -1).flatten(1)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["floordiv_tile", "floordiv_view", "mod_tile", "mod_view"])
def test_a_division_by_a_divisor_of_the_batch_coefficient_folds(how, contract):
    """The width 2 folds (the tile's width 24 is a literal); the view keeps its static layout ``(2, 6)``; nothing is
    opaque, and stage 2 converts."""
    flow = assert_converted_exact(native(lambda: DividingForms(how)), (8,), contract, batches=ALL_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_opaque is None and flow.shape_impurity is None
    layouts = [tuple(m.output_shape) for m in flow.get_mapper_repr().execution_order() if isinstance(m, ReshapeMapper)]
    if how.endswith("view"):
        assert layouts[0] == (2, 6)
    else:
        assert flow.shape_literals["getitem"] == ("getitem", "24")
    if contract == "mvm":  # stage 2: the value program's static shapes hold at every batch
        quietly(convert_torch_model, native(lambda: DividingForms(how)), (8,), 10, packaging=PACKAGINGS["mvm"],
                value_program_stage=True)


def test_an_opaque_division_is_a_decision_of_its_node_and_reaches_the_width():
    """In the pass: ``B // -2`` (``floordiv``) is opaque; ``(B // -2) // -3`` (``floordiv_1``) is computed from it;
    ``B // 6`` (``floordiv_2``) is opaque of its own; the width computed from them is decided."""
    gm = trace_model(native(lambda: Integers("floor_neg_const_tile")), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    decision = proof.batch_decision(nodes["floordiv"])
    assert decision is not None and decision.node is nodes["floordiv"] and decision.kind == "opaque"
    assert proof.batch_decision(nodes["floordiv_1"]) is decision
    assert proof.batch_decision(nodes["floordiv_2"]).node is nodes["floordiv_2"]
    assert proof.shape_decision(nodes["getitem"]) is not None
    assert proof.static_proof and layouts.layout(nodes["tile"]) is None and proof.opaque is decision


@pytest.mark.parametrize("args, expected", [
    ((Affine(2, 3), 2), Affine(1, 1)),  # (2B + 3) // 2 = B + 1
    ((Affine(2, -3), 2), Affine(1, -2)),  # (2B - 3) // 2 = B - 2: floor(-1.5)
    ((Affine(-4, 6), 2), Affine(-2, 3)),
    ((Affine(3, 7), 1), Affine(3, 7)),
], ids=["positive", "negative_constant", "negative_coefficient", "unit"])
def test_a_floor_division_by_a_divisor_of_the_coefficient_is_affine(args, expected):
    assert evaluate(operator.floordiv, list(args)) == expected
    assert evaluate(operator.mod, list(args)) == args[0].c % args[1]


@pytest.mark.parametrize("args", [
    (BATCH, -2), (evaluate(operator.floordiv, [BATCH, -2]), -3), (BATCH, 2), (BATCH, Affine(-2, 13)), (6, BATCH),
    (BATCH, 0), (Affine(4, 0), 8), (BATCH, 2.0),
], ids=["negative", "nested_negative", "not_dividing", "runtime_divisor", "literal_dividend", "zero", "half_dividing",
        "float_divisor"])
def test_every_other_floor_division_and_modulo_of_a_form_is_opaque(args):
    for target in (operator.floordiv, operator.mod):
        assert isinstance(evaluate(target, list(args)), Opaque), (target, args)


# ----------------------------------------------------------------------------- the domain, fuzzed against Python
_BINARY = [operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod, operator.rshift, operator.lshift,
           torch.sym_max, torch.sym_min, operator.and_, operator.or_, operator.xor, operator.pow, operator.eq,
           operator.lt, operator.le]
_UNARY = [operator.neg, operator.abs, operator.invert]
_CONSTANTS = [-7, -3, -2, -1, 1, 2, 3, 5, 6, 13]
FUZZ_EXPRESSIONS = 1500
FUZZ_BATCHES = ALL_BATCHES  # [restated in review round 14: from batch 1; the round-13 pass's symbolic batch was B >= 2]


def _expression(rng: random.Random, depth: int):
    pick = rng.random()
    if depth == 0 or pick < 0.25:
        return ("B",) if rng.random() < 0.5 else ("c", rng.choice(_CONSTANTS))
    if pick < 0.35:
        return ("u", rng.choice(_UNARY), _expression(rng, depth - 1))
    return ("b", rng.choice(_BINARY), _expression(rng, depth - 1), _expression(rng, depth - 1))


class _Bounded(Exception):
    """Python's value is a float, a huge integer or a huge shift there: the batch is skipped."""


def _python(tree, batch):
    if tree[0] == "B":
        return batch
    if tree[0] == "c":
        return tree[1]
    args = tuple(_python(part, batch) for part in tree[2:])
    fn = tree[1]
    if fn is operator.pow and not (type(args[1]) is int and 0 <= args[1] <= 3):
        raise _Bounded()
    if fn is operator.lshift and not (type(args[1]) is int and 0 <= args[1] <= 8):
        raise _Bounded()
    if any(type(a) is int and abs(a) > 10 ** 30 for a in args):
        raise _Bounded()
    return fn(*args)


def test_every_integer_value_the_domain_folds_is_pythons_at_every_batch():
    """The round-13 fuzz's expressions (random trees of depth 3 over B and small constants, with floor division, modulo,
    shifts, bitwise operators, powers, comparisons, ``sym_max`` and ``sym_min``), restated against the domain: every
    expression the domain folds is Python's value at batches 1 to 64; an opaque one folds nothing. The round-12 roots
    ``(B // -2) // -3`` and ``((3 + B // 7) << B) % 6`` are opaque (``test_every_other_floor_division...``)."""
    rng, seen, folded, divisions, divergent = random.Random(13), set(), 0, 0, []
    while len(seen) < FUZZ_EXPRESSIONS:
        tree = _expression(rng, 3)
        if repr(tree) in seen or "'B'" not in repr(tree):
            continue
        seen.add(repr(tree))
        value = _domain(tree)
        if isinstance(value, Opaque):
            continue
        folded += 1
        divisions += isinstance(value, Affine) and ("floordiv" in repr(tree) or "mod" in repr(tree))
        for b in FUZZ_BATCHES:
            try:
                want = _python(tree, b)
            except _Bounded:
                continue
            except (ArithmeticError, ValueError, TypeError) as exc:
                want = _Raised(exc)
            found = value if isinstance(value, _Raised) else meaning(value, b)
            if not _same(found, want):
                divergent.append((repr(tree), b, want, found))
                break
    assert not divergent, divergent[:5]
    # [restated in review round 14: the round-13 pass kept torch's value of 500 and more of these expressions (100
    # divisions); the domain folds 175 (2 divisions) and makes the rest opaque. The division rule is fuzzed on its own
    # below, where most divisions divide the coefficient]
    assert folded > 150 and divisions >= 2, (folded, divisions)


def _form_tree(rng: random.Random, depth: int):
    """A tree of ``+``, ``-`` and ``*`` by a constant over B: an affine form with a coefficient of many divisors."""
    if depth == 0:
        return ("B",) if rng.random() < 0.6 else ("c", rng.randint(-12, 12))
    fn = rng.choice([operator.add, operator.sub, operator.mul])
    if fn is operator.mul:
        return ("b", fn, _form_tree(rng, depth - 1), ("c", rng.choice([2, 3, 4, 6, -2, -6, 12])))
    return ("b", fn, _form_tree(rng, depth - 1), _form_tree(rng, depth - 1))


DIVISION_BATCHES = ALL_BATCHES + (2 ** 31 - 1, 2 ** 53 + 1, 10 ** 12 + 7)


def test_every_division_and_modulo_of_a_form_the_domain_folds_is_pythons():
    """Floor division and modulo of random affine forms (and of a division of one) by small constants of either sign,
    zero included: every folded value is Python's at batches 1 to 64 and three large ones."""
    rng, folded, opaque, divergent = random.Random(1314), 0, 0, []
    for _ in range(FUZZ_EXPRESSIONS):
        inner = _form_tree(rng, 3)
        if rng.random() < 0.3:
            inner = ("b", rng.choice([operator.floordiv, operator.mod]), inner, ("c", rng.choice([1, 2, 3, 4])))
        tree = ("b", rng.choice([operator.floordiv, operator.mod]), inner, ("c", rng.choice([1, 2, 3, 4, 6, 12, -2,
                                                                                            -3, 0])))
        value = _domain(tree)
        if isinstance(value, Opaque):
            opaque += 1
            continue
        folded += 1
        for b in DIVISION_BATCHES:
            try:
                want = _python(tree, b)
            except (ArithmeticError, ValueError, TypeError) as exc:
                want = _Raised(exc)
            found = value if isinstance(value, _Raised) else meaning(value, b)
            if not _same(found, want):
                divergent.append((repr(tree), b, want, found))
                break
    assert not divergent, divergent[:5]
    assert folded > 300 and opaque > 300, (folded, opaque)


# ----------------------------------------------------------------------------- Booleans in arithmetic (found by the fuzz)
class BooleanArithmetic(nn.Module):
    """Python's arithmetic operators read two Booleans as integers (``True + True`` is 2, ``~True`` is -2); torch's
    symbolic layer builds a malformed sympy ``Add`` of a Boolean it decided (``B % 2 < 6`` is ``True``) and a ``bool``,
    or raises for ``-``, ``~`` and ``abs`` of a symbolic Boolean. The round-13 fuzz (``fuzz_r13.py``, seed 3) found the
    first form; each view below was refused by name on the round-12 tree (its symbolic run raised) and is exact on
    w0-base. [restated in review round 14: ``B % 2`` is opaque (2 does not divide the coefficient 1), so each width is
    the host's at run time and each view the host op of its native method; round 13's ``python_operands`` is deleted]"""

    FORMS = {
        "const_add": lambda b: ((b % 2) < 6) + True,  # 2 at every batch
        "const_sub": lambda b: True - ((b % 2) < 6) + 1,  # 1
        "const_mul": lambda b: ((b % 2) < 6) * True + 1,  # 2
        "const_neg": lambda b: 3 + -((b % 2) < 6),  # 2
        "const_invert": lambda b: 4 + ~((b % 2) < 6),  # 2
        "gate_add": lambda b: 1 + ((b > 8) + (b > 9)),  # 1, 2 at batch 9, 3 from batch 10
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 12)

    def forward(self, x):
        y, b = self.fc(x), x.size(0)
        w = self.FORMS[self.how](b)
        if self.how.startswith("gate"):
            return y * torch.tile(y, (1, w)).shape[1]
        return F.softmax(y.view(b, w, -1), -1).flatten(1)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["const_add", "const_invert", "const_mul", "const_neg", "const_sub"])
def test_a_per_sample_view_from_booleans_in_arithmetic_is_exact(how, contract):
    """[restated in review round 14: the round-13 tree kept the static view; it is the host op of its native method
    now, exact at batches 1 to 64]"""
    flow = assert_converted_exact(native(lambda: BooleanArithmetic(how)), (8,), contract, batches=ALL_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_impurity is None and flow.shape_opaque[:2] == ("mod", "mod")
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_width_from_symbolic_booleans_in_arithmetic_is_a_run_time_host_value(contract):
    flow = assert_converted_exact(native(lambda: BooleanArithmetic("gate_add")), (8,), contract, batches=ALL_BATCHES,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_impurity is None and flow.shape_opaque[:2] == ("gt", "gt")


@pytest.mark.parametrize("target, args, expected", [
    (operator.add, [True, True], 2), (operator.sub, [True, True], 0), (operator.mul, [True, BATCH], BATCH),
    (operator.add, [evaluate(operator.lt, [BATCH, 0]), BATCH], BATCH),
    (operator.neg, [evaluate(operator.ge, [BATCH, 1])], -1),
], ids=["add", "sub", "mul_form", "decided_add", "decided_neg"])
def test_the_domain_reads_a_boolean_as_its_integer(target, args, expected):
    """A Boolean the domain decides (``B < 0`` is False at every B >= 1) is a Python ``bool``: Python's arithmetic on it
    is integer arithmetic, so its sum with a form is a form."""
    found = evaluate(target, args)
    assert found == expected and type(found) is type(expected)
