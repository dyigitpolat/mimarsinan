"""The values Dynamo folds into a captured body's Python from its own state, against the helper's native run (coverage
W8, sign-off L8).

Dynamo answers some calls of the body's Python itself, at capture time, by a handler that returns a constant
(``torch._dynamo.variables.torch.TorchInGraphFunctionVariable._get_handlers``):

- the tracing-state functions (``tracing_state_functions``): ``torch.compiler.is_compiling()``,
  ``torch.compiler.is_exporting()``, ``torch.nn.modules.activation._is_make_fx_tracing()`` and the others answer what
  Dynamo declares for a compiled frame, True for most, where the helper's own run answers False;
- the symbolic-shape queries (``QUERIES``): ``guard_or_false``, ``guard_or_true``, ``guard_size_oblivious``,
  ``statically_known_true``, ``statically_known_false``, ``guard_scalar``, ``has_static_value`` and ``size_hint`` of a
  batch extent answer from torch's symbolic engine (a default where it cannot decide, "not static", the stand-in's
  hint), where the helper's own run answers from the integer: ``x * 2 if statically_known_true(B % 3 == 2) else x``
  was inlined and certified, both stages agreeing, and wrong at batches 2, 5, 8, ... (the coverage W8 reproducers).

``folds_recorded`` records every such fold during one capture (each handler wrapped, under ``symbolic_record``'s lock,
and restored); ``unproved_fold`` re-proves each against the native answer: a tracing-state function's own value
outside Dynamo; a query of a Python value, the query's own answer on it (Dynamo's handler of ``statically_known_false``
returns a constant operand as it is); a query of an expression of the batch symbols, the statement that the native
answer is the folded one, proved by ``guard_proof.prove`` from the first batch ``start`` the records are proved from
(``guard_or_*``, ``guard_size_oblivious``, ``statically_known_true`` and ``guard_scalar`` answer the expression's own
value natively, ``statically_known_false`` its negation, ``size_hint`` its integer, ``has_static_value`` True). Any
other fold, and a handler table this torch does not have, keeps the call opaque.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import sympy
import torch
from torch._dynamo.variables import torch as dynamo_torch
from torch.fx.experimental import symbolic_shapes

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.fx_rewrites.body_rules.guard_proof import prove

__all__ = ["Fold", "QUERIES", "folds_recorded", "unproved_fold"]

QUERIES = ("guard_or_false", "guard_or_true", "guard_size_oblivious", "statically_known_true", "statically_known_false",
           "guard_scalar", "has_static_value", "size_hint")
"""The symbolic-shape queries Dynamo answers at capture time (``torch.fx.experimental.symbolic_shapes``)."""

_VALUE = ("guard_or_false", "guard_or_true", "guard_size_oblivious", "statically_known_true", "guard_scalar")


class Fold(NamedTuple):
    """Dynamo answered ``function`` (``name``) of ``operands`` (sympy expressions or Python values) with ``value``."""

    function: Callable[..., Any]
    name: str
    operands: Tuple[Any, ...]
    value: Any


class _Unknown:
    """An operand that is neither a symbolic value nor a Python constant."""


def _operand(variable: Any) -> Any:
    """A symbolic value as its expression (a number as the Python number), a constant as itself."""
    number = getattr(variable, "sym_num", None)
    if isinstance(number, (torch.SymInt, torch.SymBool, torch.SymFloat)):
        expr = number.node.expr
        if expr.free_symbols:
            return expr
        kind = {torch.SymInt: int, torch.SymBool: bool, torch.SymFloat: float}[type(number)]
        return kind(expr) if expr.is_number or expr in (sympy.true, sympy.false) else _Unknown()
    return variable.as_python_constant() if variable.is_python_constant() else _Unknown()


def _handler(records: List[Any], function: Callable[..., Any], name: str, original: Callable[..., Any]):
    def handler(self, tx, *args, **kwargs):
        result = original(self, tx, *args, **kwargs)
        if result is not None and result.is_python_constant():
            records.append(Fold(function, name, tuple(map(_operand, args)) + tuple(map(_operand, kwargs.values())),
                                result.as_python_constant()))
        return result
    return handler


def _targets() -> List[Tuple[Callable[..., Any], str]]:
    found = [(getattr(symbolic_shapes, name), name) for name in QUERIES]
    return found + [(function, f"{function.__module__}.{function.__name__}")
                    for function in dynamo_torch.tracing_state_functions()]


@contextlib.contextmanager
def folds_recorded(records: List[Any]) -> Iterator[None]:
    """Every fold of the block appended to ``records`` (the module docstring); torch's handlers again at the end. A
    handler this torch does not have raises (``KeyError``): the capture fails and the call stays opaque."""
    handlers: Dict[Any, Any] = dynamo_torch.TorchInGraphFunctionVariable._get_handlers()
    saved: List[Tuple[Any, Any]] = []
    try:
        for function, name in _targets():
            original = handlers[function]
            saved.append((function, original))
            handlers[function] = _handler(records, function, name, original)
        yield
    finally:
        for function, original in reversed(saved):
            handlers[function] = original


def _same(native: Any, folded: Any) -> bool:
    return type(native) is type(folded) and native == folded


def unproved_fold(fold: Fold, forms: Dict[sympy.Symbol, sympy.Expr], batch: sympy.Symbol, start: int) -> Optional[str]:
    """Why ``fold`` is not the helper's native answer for every batch from ``start`` on; ``None`` when it is."""
    shown = f"{fold.name}({', '.join(map(str, fold.operands))}) to {fold.value!r}"
    if fold.name not in QUERIES:
        native, _ = guarded(lambda: fold.function())
        return None if _same(native, fold.value) else (
            f"its capture folds {shown}, where the helper's own run gives {native!r}")
    operand, rest = (fold.operands[0], fold.operands[1:]) if fold.operands else (_Unknown(), ())
    if any(isinstance(item, (_Unknown, sympy.Basic)) for item in rest) or isinstance(operand, _Unknown):
        return f"its capture folds {shown}, an operand the batch domain does not read"
    if not isinstance(operand, sympy.Basic):  # a Python value: the query's own answer on it
        native, _ = guarded(lambda: fold.function(operand, *rest))
        return None if _same(native, fold.value) else (
            f"its capture folds {shown}, where the helper's own run gives {native!r}")
    if fold.name == "has_static_value":
        statement: Any = sympy.false if fold.value is not True else sympy.true
    elif fold.name == "size_hint" or (fold.name == "guard_scalar" and type(fold.value) is not bool):
        statement = sympy.Eq(operand, fold.value) if type(fold.value) is int else sympy.false
    elif type(fold.value) is not bool:
        statement = sympy.false
    elif fold.name in _VALUE:
        statement = operand if fold.value else sympy.Not(operand)
    else:  # statically_known_false: natively the negation of the operand's value
        statement = sympy.Not(operand) if fold.value else operand
    if not prove(statement, forms, batch, start):
        return (f"its capture folds {shown} from the batch, which the affine batch domain does not prove the helper's "
                "own run gives")
    return None
