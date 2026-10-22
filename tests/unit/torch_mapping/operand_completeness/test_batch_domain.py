"""The affine batch domain against Python's semantics (review round 14, the owner's shape-proof domain).

``symbolic/batch_domain.py`` folds a batch-dependent scalar only through its own affine forms ``a*B + c`` (B >= 1);
everything else on a batch-dependent operand is ``Opaque``. The fuzz below builds random scalar expressions of B with
every operator the reviewers used (``//``, ``%``, ``<<``, ``>>``, ``&``, ``|``, ``^``, ``**``, ``/``, float operations,
comparisons, Booleans, ``sym_max``, ``sym_min``, ``abs``, ``math.floor``, ``math.gcd``, ``sym_ite``) and holds every
expression the domain folds to Python's value, type included, at every batch from 1 to 256 and at a few large ones.
An opaque expression folds nothing, so it cannot diverge.
"""

from __future__ import annotations

import math
import operator
import random

import pytest
import torch

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import (
    BATCH,
    Affine,
    Opaque,
    evaluate,
    meaning,
)

BATCHES = tuple(range(1, 257)) + (2 ** 31 - 1, 2 ** 53 + 1, 10 ** 12 + 7, 2 ** 62 + 3)

BINARY = [operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod, operator.lshift,
          operator.rshift, operator.and_, operator.or_, operator.xor, operator.pow, operator.truediv,
          operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge, torch.sym_max,
          torch.sym_min, math.gcd]
UNARY = [operator.neg, operator.pos, operator.invert, operator.abs, operator.not_, operator.truth, operator.index,
         math.floor, torch.sym_int, torch.sym_float, torch.sym_not]
CONSTANTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, -1, -2, -3, -6, -7, True, False, 0.5, 1.0, 2.5, -1.5, 1e16]


def _node(rng: random.Random, depth: int):
    if depth == 0 or rng.random() < 0.2:
        return ("B",) if rng.random() < 0.55 else ("c", rng.choice(CONSTANTS))
    roll = rng.random()
    if roll < 0.12:
        return ("u", rng.choice(UNARY), _node(rng, depth - 1))
    if roll < 0.18:
        return ("ite", _node(rng, depth - 1), _node(rng, depth - 1), _node(rng, depth - 1))
    op = rng.choice(BINARY)
    if op in (operator.lshift, operator.rshift, operator.pow):  # a small exponent or shift, or B: never a huge one
        right = ("B",) if rng.random() < 0.3 else ("c", rng.choice([1, 2, 3, 4, 6, -2, -3, 7, 12, True, 0.5]))
    else:
        right = _node(rng, depth - 1)
    return ("b", op, _node(rng, depth - 1), right)


class _Raised:
    def __init__(self, exc: BaseException):
        self.kind = type(exc)

    def __eq__(self, other):
        return isinstance(other, _Raised) and other.kind is self.kind

    def __repr__(self):
        return f"raises {self.kind.__name__}"


def _call(fn, *args):
    try:
        return fn(*args)
    except (ArithmeticError, ValueError, TypeError, AssertionError, IndexError) as exc:
        return _Raised(exc)


def _sequence(kind, values):
    """A sequence literal: in the domain a ``torch.Size`` holding a form is the tuple of its extents."""
    if kind is torch.Size and any(isinstance(v, Affine) for v in values):
        return tuple(values)
    return kind(values)


def _domain(node):
    """The node in the domain (``Opaque`` absorbs; a batch-free call that raises is ``_Raised``)."""
    if node[0] == "B":
        return BATCH
    if node[0] == "c":
        return node[1]
    if node[0] == "s":
        return _sequence(node[1], [_domain(item) for item in node[2]])
    parts = []
    for child in (node[2:] if node[0] != "ite" else node[1:]):  # left to right, as Python evaluates operands
        part = _domain(child)
        if isinstance(part, _Raised):  # raises here, unless an earlier opaque operand raises or never ends first
            return Opaque("an earlier operand is opaque") if any(isinstance(p, Opaque) for p in parts) else part
        parts.append(part)
    target = node[1] if node[0] != "ite" else torch.sym_ite
    return _call(evaluate, target, parts)


def _python(node, batch: int):
    if node[0] == "B":
        return batch
    if node[0] == "c":
        return node[1]
    if node[0] == "s":
        return node[1]([_python(item, batch) for item in node[2]])
    parts = []
    for child in (node[2:] if node[0] != "ite" else node[1:]):
        part = _python(child, batch)
        if isinstance(part, _Raised):
            return part
        parts.append(part)
    target = node[1] if node[0] != "ite" else torch.sym_ite
    return _call(target, *parts)


def _same(found, want) -> bool:
    if isinstance(found, _Raised) or isinstance(want, _Raised):
        return found == want
    if type(want) is torch.Size and type(found) in (tuple, torch.Size) or type(want) in (tuple, list) and \
            type(found) is type(want):  # the one difference in type the lemma allows: a torch.Size as a tuple
        return len(found) == len(want) and all(map(_same, found, want))
    if type(found) is not type(want):
        return False
    if isinstance(found, float) and math.isnan(found):
        return math.isnan(want)
    return found == want


FORM_BINARY = [operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod, operator.eq, operator.ne,
               operator.lt, operator.le, operator.gt, operator.ge, torch.sym_max, torch.sym_min]
FORM_UNARY = [operator.neg, operator.pos, operator.abs, operator.not_, operator.truth, operator.index, torch.sym_int]
FORM_CONSTANTS = [0, 1, 2, 3, 4, 6, 12, -1, -2, -3, -12, True, False]


def _form_node(rng: random.Random, depth: int):
    """An expression over B, small integers and Booleans with the operations the domain closes, a divisor and most
    factors a literal: most such expressions fold, so each rule of the domain meets Python often."""
    if depth == 0 or rng.random() < 0.25:
        return ("B",) if rng.random() < 0.5 else ("c", rng.choice(FORM_CONSTANTS))
    roll = rng.random()
    if roll < 0.15:
        return ("u", rng.choice(FORM_UNARY), _form_node(rng, depth - 1))
    if roll < 0.2:
        return ("ite", _form_node(rng, depth - 1), _form_node(rng, depth - 1), _form_node(rng, depth - 1))
    op = rng.choice(FORM_BINARY)
    if op in (operator.floordiv, operator.mod) or (op is operator.mul and rng.random() < 0.6):
        right = ("c", rng.choice([1, 2, 3, 4, 6, 12, -2, -3, 0, True]))
    else:
        right = _form_node(rng, depth - 1)
    return ("b", op, _form_node(rng, depth - 1), right)


SEQUENCE_KINDS = (list, tuple, torch.Size)


def _sequence_node(rng: random.Random, depth: int):
    """Review round 15: sequences of forms of the three kinds (a list, a tuple, a ``torch.Size``), concatenated,
    compared, repeated and indexed with each other, mixed kinds included."""
    if depth == 0 or rng.random() < 0.3:
        items = [("B",) if rng.random() < 0.4 else ("c", rng.choice([1, 2, 6, 7])) for _ in range(rng.choice([1, 2]))]
        return ("s", rng.choice(SEQUENCE_KINDS), items)
    op = rng.choice([operator.add, operator.add, operator.eq, operator.ne, operator.mul, operator.getitem])
    if op is operator.mul:
        return ("b", op, _sequence_node(rng, depth - 1), ("c", rng.choice([0, 1, 2])))
    if op is operator.getitem:
        return ("b", op, _sequence_node(rng, depth - 1), ("c", rng.choice([0, -1, 1])))
    return ("b", op, _sequence_node(rng, depth - 1), _sequence_node(rng, depth - 1))


def fuzz(seed: int, count: int, depth: int = 4, node=_node):
    """``(divergences, folded affine, folded batch-dependent constants, opaque)`` over ``count`` expressions of the
    generator ``node``."""
    rng, divergences, affine, constant, opaque = random.Random(seed), [], 0, 0, 0
    for _ in range(count):
        tree = node(rng, depth)
        value = _domain(tree)
        if isinstance(value, Opaque):
            opaque += 1
            continue
        affine += isinstance(value, Affine)
        constant += not isinstance(value, Affine) and "B" in repr(tree)
        for batch in BATCHES:
            want = _python(tree, batch)
            found = value if isinstance(value, _Raised) else _call(meaning, value, batch)
            if not _same(found, want):
                divergences.append((tree, batch, found, want))
                break
    return divergences, affine, constant, opaque


def test_every_expression_the_domain_folds_is_pythons_at_every_batch():
    divergences, affine, constant, opaque = fuzz(seed=14, count=4000)
    assert not divergences, divergences[:5]
    assert affine > 200 and constant > 100 and opaque > 200, (affine, constant, opaque)


def test_every_affine_expression_the_domain_folds_is_pythons_at_every_batch():
    """The domain's own operations, most expressions folded: the rules meet Python where they decide."""
    divergences, affine, constant, opaque = fuzz(seed=141, count=4000, node=_form_node)
    assert not divergences, divergences[:5]
    assert affine > 600 and constant > 600 and opaque > 200, (affine, constant, opaque)


def test_every_sequence_expression_the_domain_folds_is_pythons_at_every_batch():
    """Review round 15: lists, tuples and ``torch.Size`` values meet each other (the round-14 domain compared a list
    and a tuple element by element, where Python compares them unequal, and concatenated them, where Python raises)."""
    divergences, affine, constant, opaque = fuzz(seed=15, count=4000, depth=3, node=_sequence_node)
    assert not divergences, divergences[:5]
    assert constant > 300 and opaque > 100, (affine, constant, opaque)


@pytest.mark.parametrize("left, right", [([BATCH, 6], (BATCH, 6)), ((BATCH, 6), [BATCH, 6]), ([BATCH], (BATCH,)),
                                         ([6], torch.Size([6])), ([BATCH, 6], [BATCH, 7])])
def test_a_list_and_a_tuple_are_never_equal(left, right):
    """``[B, 6] == (B, 6)`` is False at every batch in Python (and ``!=`` True); a list and a list of other values
    likewise."""
    assert evaluate(operator.eq, [left, right]) is False and evaluate(operator.ne, [left, right]) is True


@pytest.mark.parametrize("left, right", [([BATCH], (6,)), ((BATCH,), [6]), (torch.Size([6]), [BATCH]),
                                         ([BATCH], torch.Size([6]))])
def test_a_list_and_a_tuple_do_not_concatenate(left, right):
    """Python raises ``TypeError`` on ``list + tuple``, ``tuple + list`` and ``torch.Size + list``: opaque."""
    found = evaluate(operator.add, [left, right])
    assert isinstance(found, Opaque) and "a list and a tuple" in found.reason, found


@pytest.mark.parametrize("expression, expected", [
    (lambda: evaluate(operator.add, [BATCH, 3]), Affine(1, 3)),
    (lambda: evaluate(operator.mul, [Affine(1, 3), 4]), Affine(4, 12)),
    (lambda: evaluate(operator.sub, [Affine(4, 12), Affine(4, 2)]), 10),
    (lambda: evaluate(operator.floordiv, [Affine(6, 5), 3]), Affine(2, 1)),
    (lambda: evaluate(operator.mod, [Affine(6, 5), 3]), 2),
    (lambda: evaluate(operator.floordiv, [Affine(6, -5), 3]), Affine(2, -2)),
    (lambda: evaluate(operator.lt, [BATCH, 0]), False),
    (lambda: evaluate(operator.ge, [BATCH, 1]), True),
    (lambda: evaluate(operator.eq, [Affine(2, 0), 5]), False),
    (lambda: evaluate(torch.sym_max, [BATCH, 1]), BATCH),
    (lambda: evaluate(torch.sym_min, [Affine(2, 0), BATCH]), BATCH),
    (lambda: evaluate(operator.abs, [Affine(-1, 0)]), Affine(1, 0)),
    (lambda: evaluate(torch.sym_max, [0, Affine(1, -1)]), Affine(1, -1)),  # tied at batch 1 only: the same integer
    (lambda: evaluate(torch.sym_min, [1, BATCH]), 1),
    (lambda: evaluate(operator.getitem, [(BATCH, 6), 0]), BATCH),
    (lambda: evaluate(operator.add, [(BATCH,), (6,)]), (BATCH, 6)),
    (lambda: evaluate(operator.eq, [(BATCH, 6), (BATCH, 6)]), True),
    (lambda: evaluate(operator.add, [[BATCH], [6]]), [BATCH, 6]),
])
def test_the_domain_closes_affine_forms(expression, expected):
    found = expression()
    assert found == expected and type(found) is type(expected)


@pytest.mark.parametrize("target, args, named", [
    (operator.floordiv, [BATCH, 2], "floordiv by 2"),
    (operator.mod, [BATCH, 4], "mod by 4"),
    (operator.floordiv, [Affine(2, 0), -2], "floordiv by -2"),
    (operator.floordiv, [12, BATCH], "batch-dependent divisor"),
    (operator.truediv, [BATCH, BATCH], "truediv"),
    (operator.pow, [2, BATCH], "pow"),
    (operator.lshift, [BATCH, 1], "lshift"),
    (operator.and_, [BATCH, 3], "and_"),
    (operator.mul, [BATCH, 0.0], "float"),
    (operator.mul, [BATCH, BATCH], "product of two batch-dependent values"),
    (operator.gt, [BATCH, 8], "outcome changes with the batch"),
    (operator.ge, [BATCH, 2], "outcome changes with the batch"),
    (torch.sym_max, [6, Affine(2, -12)], "choice changes with the batch"),
    (torch.sym_max, [True, BATCH], "choice changes with the batch"),  # max(True, 1) is True, max(True, 2) is 2
    (operator.abs, [Affine(1, -20)], "sign changes"),
    (math.floor, [BATCH], "floor"),
    (math.gcd, [BATCH, 6], "gcd"),
    (torch.sym_float, [BATCH], "sym_float"),
])
def test_everything_else_on_a_batch_dependent_operand_is_opaque(target, args, named):
    found = evaluate(target, args)
    assert isinstance(found, Opaque) and named in found.reason, found


def test_an_opaque_operand_makes_the_result_opaque_and_a_batch_free_call_is_pythons():
    first = Opaque("a reason")
    assert evaluate(operator.sub, [first, first]) is first
    assert evaluate(operator.mul, [0, first]) is first
    assert evaluate(math.floor, [2.5]) == 2 and evaluate(operator.truediv, [6, 4]) == 1.5
