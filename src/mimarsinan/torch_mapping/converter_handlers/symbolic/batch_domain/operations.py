"""The operations of the affine batch domain (``forms.py``: its values), and its soundness lemma (review round 14).

Closed operations (``evaluate``; operands are domain values, a ``bool`` counts as its integer, as in Python):

- ``+``, ``-``, unary ``-`` and ``+``, and ``*`` by a batch-free integer: (a, c) combined linearly;
- ``x // k`` and ``x % k`` for a positive batch-free integer ``k`` that divides ``a``: ``(a // k)*B + c // k`` and
  ``c % k``;
- ``operator.index`` and ``torch.sym_int``: the form itself; a sequence of forms (a list, or a tuple, which also
  stands for a ``torch.Size`` holding a form): ``+`` of two of the same kind (concatenation), ``*`` by a batch-free
  integer, indexing by a batch-free index, and ``==``, ``!=`` (a list and a tuple are never equal);
- ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``not``, truth, ``abs``, ``torch.sym_max``, ``torch.sym_min`` and
  ``torch.sym_ite`` fold only when their outcome is the same for every B >= 1 (``_holds``, by the sign of the
  difference over [1, inf)); ``sym_sum`` is a sum.

Everything else with a batch-dependent operand is ``Opaque``: a true division, a power, a shift, a bitwise operator,
float arithmetic or a float meeting a form, a ``math`` function, a divisor that is batch-dependent, non-positive or does
not divide ``a``, a product of two forms, a decision that varies with B, and any operation on an ``Opaque`` value. A
call with no batch-dependent operand is Python's own call on Python values, computed as natively.

Soundness lemma (domain). If the meaning of every operand of an operation f is the operand's Python value at every
batch B >= 1, the result r of ``evaluate(f, ...)`` is ``Opaque`` or its meaning is f's Python value at every B >= 1,
of the same type (a ``torch.Size`` holding a form as a tuple, last case). Proof by case:
- batch-free operands: r is f's own Python value on them;
- ``(a1*B + c1) +- (a2*B + c2) = (a1 +- a2)*B + (c1 +- c2)``, ``k*(a*B + c) = (k*a)*B + k*c``: integer identities
  (``Affine`` collapses to the integer c when the coefficient is 0);
- for ``k > 0`` with ``k | a``, ``a*B/k`` is an integer for every integer B, so ``floor((a*B + c)/k) = (a/k)*B +
  floor(c/k)`` and ``(a*B + c) mod k = c mod k``, Python's ``//`` and ``%`` for a positive divisor;
- a comparison of x and y is a comparison of ``d = x - y = a*B + c`` with 0; for ``a != 0``, d is strictly monotone
  in B and unbounded with the sign of a, so ``d < 0`` (``<=``, ``>``, ``>=``) has the same truth at every B >= 1 when it
  has the same truth at B = 1 and in the limit (sign of a); ``d == 0`` holds at B = -c/a only, so it is False at every
  B >= 1 exactly when -c/a is not an integer >= 1; ``abs``, ``sym_max``, ``sym_min``, ``sym_ite``, ``not`` and truth
  pick an operand or negate a decided Boolean, the same one at every B >= 1 (``sym_max`` of two integers whose second
  is never smaller picks the second: where they tie, Python's first operand is the same integer);
- a sequence operation is Python's: ``==`` and ``!=`` of two of the same kind (two lists, or two tuples, a
  ``torch.Size`` counting as a tuple) compare lengths, then elements by the case above; a list and a tuple are unequal
  at every B (whatever their elements); ``+`` concatenates two of the same kind (a list and a tuple: ``Opaque``,
  Python raises), ``*`` repeats one a batch-free number of times, and a batch-free index picks an element. A
  ``torch.Size`` with a batch-dependent extent is held as the tuple of its extents: a closed operation gives Python's
  value for the ``torch.Size`` as a tuple of the same elements (the one difference in type the lemma allows).
  [corrected in review round 15: "a tuple operation applies Python's on the tuple, element by element" did not hold
  for a list and a tuple, which the domain compared element by element and concatenated]
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import (
    SEQUENCES as _SEQUENCES,
    Affine,
    Opaque,
    affine,
    batch_dependent as _batch_dependent,
    form,
    opaque_in,
)

__all__ = ["evaluate"]


def _holds(compare: Callable[[int, int], bool], a: int, c: int) -> Optional[bool]:
    """The truth of ``compare(a*B + c, 0)`` when it is the same at every B >= 1, else ``None``."""
    if a == 0:
        return compare(c, 0)
    if compare in (operator.eq, operator.ne):
        root = c % a == 0 and -c // a >= 1  # a*B + c == 0 at the integer B = -c/a >= 1
        return None if root else compare is operator.ne
    first, limit = compare(a + c, 0), compare(a, 0)  # at B = 1, and as B grows (the sign of a)
    return first if first == limit else None


def _difference(x: Any, y: Any) -> Optional[Tuple[int, int]]:
    fx_, fy = form(x), form(y)
    return None if fx_ is None or fy is None else (fx_[0] - fy[0], fx_[1] - fy[1])


def _float_meets(*values: Any) -> bool:
    return any(type(value) is float for value in values)


def _arithmetic(name: str, combine: Callable[[int, int, int, int], Tuple[int, int]]):
    def rule(x: Any, y: Any) -> Any:
        if _float_meets(x, y):
            return Opaque(f"{name} of a batch-dependent value and a float")
        fx_, fy = form(x), form(y)
        if fx_ is None or fy is None:
            return Opaque(f"{name} of a batch-dependent value and a {type(y if fx_ else x).__name__}")
        a, c = combine(*fx_, *fy)
        return affine(a, c)
    return rule


def _list_meets_tuple(x: Any, y: Any) -> bool:
    """A list and a tuple (or ``torch.Size``): Python never compares them equal and does not concatenate them."""
    return (type(x) is list) != (type(y) is list)


def _concatenation(rule: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def add(x: Any, y: Any) -> Any:
        if type(x) in _SEQUENCES and type(y) in _SEQUENCES:
            if _list_meets_tuple(x, y):
                return Opaque("a concatenation of a list and a tuple, which Python refuses")
            return type(x)([*x, *y]) if type(x) is not torch.Size else tuple([*x, *y])
        return rule(x, y)
    return add


def _multiply(x: Any, y: Any) -> Any:
    for seq, count in ((x, y), (y, x)):
        if type(seq) in _SEQUENCES:
            if type(count) not in (int, bool):
                return Opaque("a sequence repeated a batch-dependent number of times")
            return (list(seq) if type(seq) is list else tuple(seq)) * int(count)
    if _float_meets(x, y):
        return Opaque("mul of a batch-dependent value and a float")
    fx_, fy = form(x), form(y)
    if fx_ is None or fy is None:
        return Opaque(f"mul of a batch-dependent value and a {type(y if fx_ else x).__name__}")
    if fx_[0] and fy[0]:
        return Opaque("a product of two batch-dependent values, not affine in B")
    scale, (a, c) = (fx_[1], fy) if fx_[0] == 0 else (fy[1], fx_)
    return affine(scale * a, scale * c)


def _division(name: str, result: Callable[[int, int, int], Any]):
    def rule(x: Any, k: Any) -> Any:
        fx_ = form(x)
        if type(k) not in (int, bool) or fx_ is None:
            what = "a batch-dependent divisor" if isinstance(k, Affine) else f"a divisor or dividend of type " \
                                                                             f"{type(k if fx_ else x).__name__}"
            return Opaque(f"{name} with {what}")
        a, c = fx_
        if int(k) <= 0 or a % int(k):
            return Opaque(f"{name} by {int(k)}, which is not a positive divisor of the batch coefficient {a}")
        return result(a, c, int(k))
    return rule


def _comparison(name: str, compare: Callable[[int, int], bool]):
    def rule(x: Any, y: Any) -> Any:
        if type(x) in _SEQUENCES and type(y) in _SEQUENCES and compare in (operator.eq, operator.ne):
            if len(x) != len(y) or _list_meets_tuple(x, y):
                return compare is operator.ne
            equal = [rule_eq(p, q) for p, q in zip(x, y)]
            if any(e is False for e in equal):
                return compare is operator.ne
            if all(e is True for e in equal):
                return compare is operator.eq
            return Opaque(f"{name} of shapes whose equality changes with the batch")
        if _float_meets(x, y):
            return Opaque(f"{name} of a batch-dependent value and a float")
        found = _difference(x, y)
        if found is None:
            return Opaque(f"{name} of a batch-dependent value and a {type(y if form(x) else x).__name__}")
        decided = _holds(compare, *found)
        return Opaque(f"{name} whose outcome changes with the batch") if decided is None else decided
    return rule


def rule_eq(x: Any, y: Any) -> Any:
    return _COMPARISONS[operator.eq](x, y)


def _truth(x: Any) -> Any:
    found = form(x)
    if found is None:
        return Opaque(f"the truth of a {type(x).__name__}")
    decided = _holds(operator.ne, *found)
    return Opaque("a truth value that changes with the batch") if decided is None else decided


def _negated(x: Any) -> Any:
    truth = _truth(x)
    return truth if isinstance(truth, Opaque) else not truth


def _absolute(x: Any) -> Any:
    found = form(x)
    if found is None:
        return Opaque(f"abs of a {type(x).__name__}")
    a, c = found
    if _holds(operator.ge, a, c) is True:
        return x
    if _holds(operator.le, a, c) is True:
        return affine(-a, -c)
    return Opaque("abs of a batch-dependent value whose sign changes with the batch")


def _extreme(name: str, first: Callable[[int, int], bool], second: Callable[[int, int], bool],
             tied: Callable[[int, int], bool]):
    """``sym_max`` (``sym_min``): Python's ``max`` (``min``) returns its first operand unless the second is strictly
    larger (smaller). The choice folds when it is the same at every B >= 1, or when the second operand is never
    smaller (larger) and both are integers: where they tie, either is the same integer."""
    def rule(x: Any, y: Any) -> Any:
        if _float_meets(x, y):
            return Opaque(f"{name} of a batch-dependent value and a float")
        found = _difference(x, y)
        if found is not None and _holds(first, *found) is True:
            return x
        if found is not None and (_holds(second, *found) is True or (
                type(x) is not bool and type(y) is not bool and _holds(tied, *found) is True)):
            return y
        return Opaque(f"{name} whose choice changes with the batch")
    return rule


def _identity(name: str):
    def rule(x: Any) -> Any:
        return x if isinstance(x, Affine) else Opaque(f"{name} of a {type(x).__name__}")
    return rule


def _getitem(sequence: Any, index: Any) -> Any:
    if type(sequence) not in _SEQUENCES or _batch_dependent(index):
        return Opaque("an index into or by a batch-dependent value")
    found = sequence[index]
    return tuple(found) if type(found) is torch.Size else found


def _kind(value: Any) -> type:
    return int if isinstance(value, Affine) else type(value)


def _ite(condition: Any, then: Any, otherwise: Any) -> Any:
    """``torch.sym_ite`` of a decided ``bool`` and branches of one type (torch asserts both)."""
    if type(condition) is not bool or _kind(then) is not _kind(otherwise):
        return Opaque("sym_ite of a condition that changes with the batch or of branches of two types")
    return then if condition else otherwise


def _sum(items: Any) -> Any:
    if type(items) not in _SEQUENCES:
        return Opaque("sym_sum of a non-sequence")
    total: Any = 0
    for item in items:
        total = _RULES[operator.add](total, item)
        if isinstance(total, Opaque):
            return total
    return total


_COMPARISONS: Dict[Any, Callable[[Any, Any], Any]] = {
    fn: _comparison(fn.__name__, fn)
    for fn in (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge)
}

_RULES: Dict[Any, Callable[..., Any]] = {
    operator.add: _concatenation(_arithmetic("add", lambda a1, c1, a2, c2: (a1 + a2, c1 + c2))),
    operator.sub: _arithmetic("sub", lambda a1, c1, a2, c2: (a1 - a2, c1 - c2)),
    operator.mul: _multiply,
    operator.neg: lambda x: _arithmetic("neg", lambda a1, c1, a2, c2: (-a1, -c1))(x, 0),
    operator.pos: lambda x: _arithmetic("pos", lambda a1, c1, a2, c2: (a1, c1))(x, 0),
    operator.floordiv: _division("floordiv", lambda a, c, k: affine(a // k, c // k)),
    operator.mod: _division("mod", lambda a, c, k: c % k),
    operator.abs: _absolute,
    operator.not_: _negated,
    operator.truth: _truth,
    operator.index: _identity("index"),
    operator.getitem: _getitem,
    torch.sym_int: _identity("sym_int"),
    torch.sym_not: _negated,
    torch.sym_max: _extreme("sym_max", operator.ge, operator.lt, operator.le),
    torch.sym_min: _extreme("sym_min", operator.le, operator.gt, operator.ge),
    torch.sym_ite: _ite,
    **_COMPARISONS,
}
if hasattr(torch, "sym_sum"):
    _RULES[torch.sym_sum] = _sum

_ARITIES = {operator.neg: 1, operator.pos: 1, operator.abs: 1, operator.not_: 1, operator.truth: 1,
            operator.index: 1, torch.sym_int: 1, torch.sym_not: 1, torch.sym_ite: 3}
if hasattr(torch, "sym_sum"):
    _ARITIES[torch.sym_sum] = 1


def _name(target: Any) -> str:
    return getattr(target, "__name__", None) or type(target).__name__


def evaluate(target: Any, args: Sequence[Any], kwargs: Optional[Mapping[str, Any]] = None) -> Any:
    """``target(*args, **kwargs)`` in the domain (``args``: domain values). A call with no batch-dependent operand is
    Python's own (its exception propagates); one with an ``Opaque`` operand is that ``Opaque`` value; one the domain
    closes is computed by its rule; any other is ``Opaque``, naming the operation."""
    kwargs = dict(kwargs or {})
    if target is operator.getitem and len(args) == 2 and type(args[0]) in _SEQUENCES and not kwargs \
            and not _batch_dependent(args[1]):  # indexing a sequence reads none of its other elements
        found = args[0][args[1]]
        return tuple(found) if type(found) is torch.Size and _batch_dependent(found) else found
    found = opaque_in((tuple(args), kwargs))
    if found is not None:
        return found
    if not _batch_dependent((tuple(args), kwargs)):
        return target(*args, **kwargs)
    rule = _RULES.get(target)
    if rule is None or kwargs or len(args) != _ARITIES.get(target, 2):
        return Opaque(f"{_name(target)} of a batch-dependent value, outside the affine batch domain")
    return rule(*args)
