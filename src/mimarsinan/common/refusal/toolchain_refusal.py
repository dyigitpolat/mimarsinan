"""The one base type of every named refusal, so a refusal by name is told apart from a loud crash by type."""

from __future__ import annotations

from typing import Optional


class ToolchainRefusal(Exception):
    """A program the toolchain declines on purpose; the message names the operator, the node and the reason.

    Each refusal keeps its builtin base through multiple inheritance (``class X(ToolchainRefusal, RuntimeError)``),
    so ``except RuntimeError`` clauses keep catching it. Any exception that is not a ``ToolchainRefusal`` is a
    loud failure: an anonymous crash, never a refusal.
    """


def is_toolchain_refusal(exc: BaseException) -> bool:
    """``exc`` is a refusal by name (an instance of ``ToolchainRefusal``)."""
    return isinstance(exc, ToolchainRefusal)


def refusal_in_chain(exc: BaseException) -> Optional[ToolchainRefusal]:
    """The first ``ToolchainRefusal`` in ``exc``'s cause/context chain (``exc`` itself first), else ``None``.

    A refusal raised inside a forward pass reaches the caller wrapped (``[ModelRepresentation] forward failed``);
    the chain still carries its name.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ToolchainRefusal):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None
