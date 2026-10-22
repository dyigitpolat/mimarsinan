"""The affine batch domain: the only arithmetic a batch-dependent scalar is folded with (review round 14).

The owner's shape-proof domain. Torch's symbolic engine (sympy through ShapeEnv) simplifies some integer expressions
of the batch size wrongly with no guard (``sym_max(6, 2B - 12) % 6`` to 0, ``(B & 3) < (B & 3)`` to True), so no fold
rests on it. The toolchain computes batch-dependent scalars itself, here, by integer arithmetic.

Values (``meaning``: the Python value a domain value stands for at a batch B):

- a batch-free value is a Python value (``int``, ``bool``, ``float``, ``None``, a ``str``, a tuple of values ...): its
  meaning is itself at every batch;
- ``Affine(a, c)``, with Python integers ``a != 0`` and ``c``, is the integer ``a*B + c`` at every batch B >= 1;
- ``Opaque(reason)`` is a batch-dependent value the domain does not compute. It has no meaning here: it is computed
  on the host at run time, as natively, and nothing is folded from it.

The operations on these values, and the domain's soundness lemma, are in ``operations.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

__all__ = ["Affine", "BATCH", "Opaque", "affine", "batch_dependent", "describe", "form", "meaning", "opaque_in",
           "SEQUENCES"]


@dataclass(frozen=True)
class Affine:
    """``a*B + c`` for every batch B >= 1 (``a != 0``)."""

    a: int
    c: int

    def at(self, batch: int) -> int:
        return self.a * batch + self.c

    def __repr__(self) -> str:
        return describe(self)


@dataclass(frozen=True, eq=False)
class Opaque:
    """A batch-dependent value the domain does not compute (``reason``: why); computed on the host at run time.
    ``origin``: what the symbolic pass attaches (the decision that made it opaque). Two opaque values are equal only
    when they are the same object: the domain never identifies two values it does not compute."""

    reason: str
    origin: Any = None


BATCH = Affine(1, 0)
"""The batch size B itself."""

SEQUENCES = (tuple, list, torch.Size)


def affine(a: int, c: int) -> Any:
    """The domain value of ``a*B + c``: the integer ``c`` when ``a`` is 0."""
    return c if a == 0 else Affine(a, c)


def describe(value: Any) -> str:
    """``2*B + 1`` style; a batch-free value as its ``repr``."""
    if not isinstance(value, Affine):
        return repr(value)
    head = "B" if value.a == 1 else "-B" if value.a == -1 else f"{value.a}*B"
    return head if value.c == 0 else f"{head} {'+' if value.c > 0 else '-'} {abs(value.c)}"


def form(value: Any) -> Optional[Tuple[int, int]]:
    """``(a, c)`` of an integer domain value: a form, or a batch-free ``int`` or ``bool`` (``a`` 0); else ``None``."""
    if isinstance(value, Affine):
        return value.a, value.c
    if type(value) in (int, bool):
        return 0, int(value)
    return None


def meaning(value: Any, batch: int) -> Any:
    """The Python value ``value`` stands for at ``batch`` (an ``Opaque`` value has none: ``ValueError``)."""
    if isinstance(value, Opaque):
        raise ValueError(f"an opaque value has no meaning in the domain ({value.reason})")
    if isinstance(value, Affine):
        return value.at(batch)
    if type(value) in SEQUENCES:
        return type(value)(meaning(item, batch) for item in value)
    return value


def _leaves(value: Any) -> list:
    if type(value) in SEQUENCES:
        return [leaf for item in value for leaf in _leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if type(value) is slice:
        return _leaves((value.start, value.stop, value.step))
    return [value]


def opaque_in(value: Any) -> Optional[Opaque]:
    """The first ``Opaque`` leaf of ``value``, through tuples, lists, ``torch.Size``, dicts and slices."""
    return next((leaf for leaf in _leaves(value) if isinstance(leaf, Opaque)), None)


def batch_dependent(value: Any) -> bool:
    return any(isinstance(leaf, (Affine, Opaque)) for leaf in _leaves(value))
