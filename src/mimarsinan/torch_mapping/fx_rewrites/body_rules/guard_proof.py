"""A capture's statements on the batch, proved in the converter's own affine batch domain (coverage W8, sign-off L8).

Dynamo's capture of a leaf call's body holds what the body assumed of its operands' extents: guards, deferred runtime
asserts, replacements and value ranges, each a sympy expression of the capture's symbols (``body_capture``). Each
symbol stands for its affine form ``a*B + c`` in the batch B (fitted to the seeded samples, then proved for every batch
by the converter's symbolic pass, ``leaf_operands``). A statement is PROVED here when its translation into the domain
(``batch_domain.evaluate``) is the Python value ``True``: every operation of the statement is one the domain closes and
decides for every B >= 1, by the domain's own integer arithmetic, never by torch's symbolic engine (the owner's
shape-proof domain; review rounds 12 to 14 found sympy through ShapeEnv folding batch expressions wrongly). Any other
statement is not proved, and the call stays opaque.

Translation (``prove``), node by node, each the domain operation of the same integer meaning at every B >= 1:
integers; a symbol as its form; ``Add``, ``Mul``; torch's ``FloorDiv`` (Python's ``//``) and its ``CleanDiv``;
``PythonMod`` (Python's ``%``); torch's ``Mod`` (a modulus of non-negative operands) only where the domain proves its
dividend non-negative for every B >= 1, so it is Python's ``%``; ``Max``, ``Min`` (torch's or sympy's) as ``sym_max``,
``sym_min`` folded pairwise; ``Eq``, ``Ne``, ``Lt``, ``Le``, ``Gt``, ``Ge``; ``And``, ``Or``, ``Not`` and the two truth
values. Anything else (a rational, a power, a float, a bitwise or opaque function) is not translated.

Lemma. If every symbol's meaning is its form at every B >= 1, a statement ``prove`` returns ``True`` for holds at every
batch B >= 1: by induction over the expression, each translated node's domain value has the node's integer (or
Boolean) meaning at every B >= 1 (the domain's lemma, ``batch_domain/operations.py``; ``FloorDiv``, ``PythonMod`` and
the comparisons are Python's operations, and torch's ``Mod`` is translated only where it equals Python's ``%``); a
conjunction is ``True`` when each conjunct is, a disjunction when one is, a negation when its operand is ``False``.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Optional, cast

import sympy
import torch
from torch.utils._sympy import functions as torch_sympy

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import Affine, Opaque, evaluate

__all__ = ["affine_of", "prove", "symbol_values", "translate"]

_RELATIONS = {sympy.Eq: operator.eq, sympy.Ne: operator.ne, sympy.StrictLessThan: operator.lt,
              sympy.LessThan: operator.le, sympy.StrictGreaterThan: operator.gt, sympy.GreaterThan: operator.ge}

_UNPROVED = Opaque("a statement the batch domain does not translate")


def affine_of(expression: sympy.Expr, batch: sympy.Symbol) -> Any:
    """The domain value of an integer polynomial of degree at most 1 in ``batch`` (a symbol's fitted form)."""
    try:
        poly = sympy.Poly(expression, batch)
    except sympy.PolynomialError:
        return _UNPROVED
    if poly.degree() > 1 or not all(coefficient.is_Integer for coefficient in poly.all_coeffs()):
        return _UNPROVED
    a, c = (int(poly.coeff_monomial(batch)), int(poly.coeff_monomial(1)))
    return c if a == 0 else Affine(a, c)


def _fold(function: Callable[..., Any], values: list) -> Any:
    found = values[0]
    for value in values[1:]:
        found = evaluate(function, [found, value])
    return found


def translate(expression: Any, symbols: Dict[sympy.Symbol, Any]) -> Any:
    """``expression``'s domain value: an ``int``, an ``Affine`` form, a ``bool``, or ``Opaque`` when the domain does
    not compute it."""
    if expression is sympy.true or expression is True:
        return True
    if expression is sympy.false or expression is False:
        return False
    if isinstance(expression, sympy.Integer):
        return int(expression)
    if isinstance(expression, sympy.Symbol):
        return symbols.get(expression, _UNPROVED)
    args = [translate(arg, symbols) for arg in expression.args]
    if any(isinstance(arg, Opaque) for arg in args):
        return _UNPROVED
    kind = type(expression)
    if kind in _RELATIONS:
        return evaluate(_RELATIONS[kind], args)
    if isinstance(expression, sympy.Add):
        return _fold(operator.add, args)
    if isinstance(expression, sympy.Mul):
        return _fold(operator.mul, args)
    if isinstance(expression, torch_sympy.FloorDiv):
        return evaluate(operator.floordiv, args)
    if isinstance(expression, torch_sympy.PythonMod):
        return evaluate(operator.mod, args)
    if isinstance(expression, torch_sympy.Mod):
        return evaluate(operator.mod, args) if evaluate(operator.ge, [args[0], 0]) is True else _UNPROVED
    if isinstance(expression, (torch_sympy.Max, sympy.Max)):
        return _fold(torch.sym_max, args)
    if isinstance(expression, (torch_sympy.Min, sympy.Min)):
        return _fold(torch.sym_min, args)
    if isinstance(expression, sympy.And):
        return True if all(arg is True for arg in args) else _UNPROVED
    if isinstance(expression, sympy.Or):
        return True if any(arg is True for arg in args) else _UNPROVED
    if isinstance(expression, sympy.Not):
        return True if args[0] is False else _UNPROVED
    return _UNPROVED


def symbol_values(forms: Dict[sympy.Symbol, sympy.Expr], batch: sympy.Symbol,
                  start: int = 1) -> Dict[sympy.Symbol, Any]:
    """Each capture symbol's domain value, and ``batch``'s, for the batches from ``start`` on: with B standing for
    ``B + start - 1``, a form ``a*B + c`` is ``a*B + (a*(start - 1) + c)`` for every B >= 1."""
    shift = {batch: batch + (start - 1)}
    symbols = {symbol: affine_of(sympy.expand(value.xreplace(shift)), batch) for symbol, value in forms.items()}
    symbols[batch] = affine_of(batch + (start - 1), batch)
    return symbols


def prove(statement: Any, forms: Dict[sympy.Symbol, sympy.Expr], batch: sympy.Symbol, start: int = 1) -> bool:
    """``statement`` holds for every batch B >= ``start``, with each capture symbol at its form in ``batch``
    (``forms``), as the batch domain computes it (the module's lemma); ``False`` when the domain does not show it."""
    found: Optional[Any] = translate(cast(Any, statement), symbol_values(forms, batch, start))
    return found is True
