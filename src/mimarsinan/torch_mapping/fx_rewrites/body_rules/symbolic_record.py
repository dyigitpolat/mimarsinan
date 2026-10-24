"""Every symbolic integer operation and static evaluation torch makes while a body is captured, re-proved in the
affine batch domain (coverage W8, sign-off L8).

Dynamo interprets a body on stand-ins whose batch-carrying extents are symbols, so the body's Python arithmetic on an
extent (``x.size(0) * 16``, ``max(6, 2 * x.size(0) - 12) % 6``) runs in torch's symbolic engine: each operation of a
``SymInt`` or ``SymBool`` builds a sympy expression (``SymNode``), simplified by sympy; and a decision on one
(``bool``, ``int``, ``guard_or_false``, a meta function's size check) is a guard, or a static evaluation that leaves
none (``ShapeEnv._maybe_evaluate_static``, ``_maybe_fast_eval_comparison``), or, for ``guard_or_false`` and
``guard_or_true`` where the static evaluation states nothing, their default, taken with no guard either. A guard is
judged by ``body_capture`` (rule (b)); a static evaluation, a default and a simplification are torch's folds, the
owner's shape-proof domain trusts none of them (sympy's simplification of ``Max(6, 2*B - 12) % 6`` to 0 inlined a
branch wrong from batch 10: the coverage W8 reproducers). So ``recorded`` records every one of them while the capture
runs, and ``unproved_record`` re-proves each in the domain:

- an operation (``add``, ``mul``, ``floordiv``, ``mod``, the comparisons, ``sym_max`` ...): the domain's value of the
  operation on its operands' domain values must equal the domain value of the expression torch built, or both must be
  outside the domain (an expression nothing is folded from: torch did not fold it to a value the domain does not
  give);
- a static evaluation to a value, and a decision of ``guard_or_false`` or ``guard_or_true`` (``torch``'s private
  ``_guard_or``, recorded whether the static evaluation or the default took it): the statement ``expr == value`` is
  proved by ``guard_proof.prove``; a static evaluation to an expression is judged as an operation is.

Each is proved for every batch B >= B0, the first batch at which every symbol's form is at least 2, when rule (c)
compares the capture at every batch below B0 (``body_comparison``: at batch 1 and wherever a form is 0 or 1, the
capture equals a static capture, Python's own integers), and for every B >= 1 otherwise: torch assumes a size symbol
is positive (``Eq(s, 0)`` is False as it is built) and a size-oblivious evaluation that it is at least 2, which hold
from B0 on.

Lemma. If every record passes, every Python value the capture computed from a batch symbol and every decision it took
on one is the value or the decision the native helper takes at every batch B >= B0: by induction over the records in
the order torch made them, an operation's result is the domain's (its lemma) or is never folded, and a decision is a
statement the domain proves; below B0, rule (c) shows the capture is the static capture of that batch.

The values Dynamo folds into the body's Python from a symbolic-shape query or a tracing-state function are recorded
with them (``folds``) and re-proved against the helper's native answer.

The recording replaces the ``SymNode`` methods, the two ``ShapeEnv`` methods, ``symbolic_shapes._guard_or`` and
Dynamo's handlers of the folded functions for the duration of one capture, under a lock, and restores them; it calls
each original and changes no result.
"""

from __future__ import annotations

import contextlib
import functools
import operator
import threading
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, cast

import sympy
import torch
from torch.fx.experimental import sym_node, symbolic_shapes
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import Affine, Opaque, evaluate
from mimarsinan.torch_mapping.fx_rewrites.body_rules.folds import Fold, folds_recorded, unproved_fold
from mimarsinan.torch_mapping.fx_rewrites.body_rules.guard_proof import prove, symbol_values, translate

__all__ = ["Record", "recorded", "unproved_record"]

_LOCK = threading.Lock()


class Record(NamedTuple):
    """``kind``: ``"operation"`` (a ``SymNode`` method ``name``) or ``"static"`` (a ``ShapeEnv`` evaluation);
    ``operands`` and ``result`` as sympy expressions or Python values; ``options``: a static evaluation's keywords."""

    kind: str
    name: str
    operands: Tuple[Any, ...]
    result: Any
    options: Tuple[Tuple[str, Any], ...] = ()


def _expr(value: Any) -> Any:
    if isinstance(value, sym_node.SymNode):
        return value.expr
    if type(value) in (list, tuple):
        return tuple(_expr(item) for item in value)
    return value


def _methods() -> List[Tuple[str, str]]:
    """``(attribute, method)`` of every ``SymNode`` operation torch defines (binary, unary, sizes and strides)."""
    names = [*sym_node.magic_methods, *sym_node.unary_methods, *sym_node.sizes_strides_methods]
    trailing = sym_node.magic_methods_on_operator_with_trailing_underscore
    found = {(f"_{name}_" if name in trailing else f"_{name}", name) for name in names}
    found.add(("sym_sum", "sym_sum"))
    return sorted(pair for pair in found if hasattr(sym_node.SymNode, pair[0]))


def _operation(records: List[Record], name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def operation(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        records.append(Record("operation", name, (_expr(self), *map(_expr, args)), _expr(result)))
        return result
    return functools.update_wrapper(operation, original)  # its attributes (a cache's ``cache_clear``) kept


def _static(records: List[Record], name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def static(self, expr, *args, **kwargs):
        result = original(self, expr, *args, **kwargs)
        if result is not None:
            records.append(Record("static", name, (expr, *args), result, tuple(sorted(kwargs.items()))))
        return result
    return functools.update_wrapper(static, original)


def _decision(records: List[Record], original: Callable[..., Any]) -> Callable[..., Any]:
    def decide(value, default):
        result = original(value, default)
        node = getattr(value, "node", None)
        if isinstance(node, sym_node.SymNode):  # the decision, the static evaluation's or the default
            records.append(Record("static", "guard_or", (node.expr,), bool(result)))
        return result
    return functools.update_wrapper(decide, original)


@contextlib.contextmanager
def recorded() -> Iterator[List[Any]]:
    """The records of every ``SymNode`` operation, static evaluation, ``guard_or`` decision and Dynamo fold torch makes
    in the block (the module docstring); the methods are torch's own again when it ends."""
    with _LOCK:
        records: List[Any] = []
        replaced: List[Tuple[Any, str, Any]] = []
        try:
            for attribute, name in _methods():
                original = sym_node.SymNode.__dict__[attribute]
                replaced.append((sym_node.SymNode, attribute, original))
                setattr(sym_node.SymNode, attribute, _operation(records, name, original))
            for attribute in ("_maybe_evaluate_static", "_maybe_fast_eval_comparison"):
                original = ShapeEnv.__dict__[attribute]
                replaced.append((ShapeEnv, attribute, original))
                setattr(ShapeEnv, attribute, _static(records, attribute, original))
            original = symbolic_shapes.__dict__["_guard_or"]
            replaced.append((symbolic_shapes, "_guard_or", original))
            setattr(symbolic_shapes, "_guard_or", _decision(records, original))
            with folds_recorded(records):
                yield records
        finally:
            for owner, attribute, original in reversed(replaced):
                setattr(owner, attribute, original)


def _connective(absorbing: bool) -> Callable[[Any, Any], Any]:
    """``or`` (``absorbing`` True) and ``and`` (False) of decided operands, three-valued: an operand equal to the
    absorbing value decides it, whatever the other operand is."""
    def rule(x: Any, y: Any) -> Any:
        if x is absorbing or y is absorbing:
            return absorbing
        if type(x) is bool and type(y) is bool:
            return not absorbing
        return Opaque("a Boolean connective of a value that changes with the batch")
    return rule


def _integral(x: Any) -> Any:
    """``int``, ``floor``, ``ceil`` and ``trunc`` of an integer are the integer; of anything else, outside the
    domain."""
    return x if type(x) is int or isinstance(x, Affine) else Opaque("rounding of a value that is not an integer")


_DOMAIN: Dict[str, Callable[..., Any]] = {
    **{name: (lambda fn: lambda *a: evaluate(fn, list(a)))(fn) for name, fn in {
        "add": operator.add, "sub": operator.sub, "mul": operator.mul, "floordiv": operator.floordiv,
        "int_floordiv": operator.floordiv, "mod": operator.mod, "eq": operator.eq, "ne": operator.ne,
        "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "neg": operator.neg,
        "pos": operator.pos, "abs": operator.abs, "sym_max": torch.sym_max, "sym_min": torch.sym_min,
        "sym_ite": torch.sym_ite, "sym_not": operator.not_}.items()},
    "and": _connective(False), "or": _connective(True),
    "sym_int": _integral, "floor": _integral, "ceil": _integral, "trunc": _integral,
    "sym_sum": lambda _receiver, terms: evaluate(operator.add, [0, 0]) if not terms else _sum(terms),
}
"""The domain's value of each ``SymNode`` operation on its operands' domain values; any other is outside it."""

_REFLEXIVE = {"eq": True, "le": True, "ge": True, "ne": False, "lt": False, "gt": False}
"""A comparison of an integer expression with itself: its value, whatever the expression's."""


def _sum(terms: Tuple[Any, ...]) -> Any:
    """``sym_sum``: the sum of its terms (its receiver is the first term's node, not a term of its own)."""
    total: Any = 0
    for term in terms:
        total = evaluate(operator.add, [total, term])
    return total


def _value(value: Any, symbols: Dict[sympy.Symbol, Any]) -> Any:
    if type(value) in (int, bool):
        return value
    if type(value) is tuple:
        found = tuple(_value(item, symbols) for item in value)
        return next((item for item in found if isinstance(item, Opaque)), found)
    if isinstance(value, sympy.Basic):
        return translate(value, symbols)
    return Opaque(f"a value of type {type(value).__name__}")


def _free(value: Any) -> bool:
    if type(value) is tuple:
        return any(map(_free, value))
    return bool(getattr(value, "free_symbols", None))


def _agree(mine: Any, theirs: Any) -> bool:
    if isinstance(mine, Opaque) or isinstance(theirs, Opaque):
        return isinstance(mine, Opaque) and isinstance(theirs, Opaque)
    return type(mine) is type(theirs) and mine == theirs or (
        type(mine) in (int, bool) and type(theirs) in (int, bool) and int(mine) == int(theirs))


def _first_batch(forms: Dict[sympy.Symbol, sympy.Expr], batch: sympy.Symbol) -> Optional[int]:
    """The first batch B >= 1 from which every symbol's form ``a*B + c`` is at least 2 (``None``: one never is)."""
    first = 1
    for value in forms.values():
        a, c = int(cast(Any, value.coeff(batch, 1))), int(cast(Any, value.subs(batch, 0)))
        if a < 0 or (a == 0 and c < 2):
            return None
        if a > 0:
            first = max(first, -(-(2 - c) // a))
    return first


def unproved_record(records: List[Any], forms: Dict[sympy.Symbol, sympy.Expr], batch: sympy.Symbol,
                    compared: List[int]) -> Optional[str]:
    """Why a record of the capture is not re-proved (the module docstring); ``None`` when every one is. ``compared``:
    the batches rule (c) compares the capture at."""
    start = _first_batch(forms, batch)
    start = start if start is not None and all(b in compared for b in range(1, start)) else 1
    symbols = symbol_values(forms, batch, start)
    for record in records:
        if isinstance(record, Fold):
            reason = unproved_fold(record, forms, batch, start)
            if reason is not None:
                return reason
            continue
        if not (_free(record.operands) or _free(record.result)):
            continue
        shown = f"{record.name}({', '.join(map(str, record.operands))}) = {record.result}"
        options = dict(record.options)
        if record.kind == "operation" or not isinstance(record.result, (sympy.logic.boolalg.BooleanAtom, bool)) and \
                _free(record.result):
            if record.name in _REFLEXIVE and len(record.operands) == 2 and record.operands[0] == record.operands[1] \
                    and getattr(record.operands[0], "is_integer", False) is True \
                    and _value(record.result, symbols) is _REFLEXIVE[record.name]:
                continue  # x == x, x <= x, x >= x hold and x != x, x < x, x > x fail, for every value of x
            rule = _DOMAIN.get(record.name) if record.kind == "operation" else (lambda x, *_: x)
            operands = [_value(item, symbols) for item in record.operands]
            short = record.name in ("and", "or")  # three-valued: an absorbing operand decides an undecided one
            mine = Opaque("outside the domain") if rule is None or (not short and any(
                isinstance(x, Opaque) for x in operands)) else rule(*operands)
            if not _agree(mine, _value(record.result, symbols)):
                return f"its capture computes {shown} from the batch, which the affine batch domain does not give"
            continue
        if options.get("compute_hint") or options.get("axioms") or options.get("var_to_range"):
            return f"its capture evaluates {shown} on hints or assumptions, which the batch domain does not prove"
        statement = sympy.Eq(record.operands[0], record.result) if not isinstance(
            record.result, (sympy.logic.boolalg.BooleanAtom, bool)) else (
            record.operands[0] if bool(record.result) else sympy.Not(record.operands[0]))
        if not prove(statement, forms, batch, start):
            return f"its capture decides {shown} without a guard, which the affine batch domain does not prove"
    return None
