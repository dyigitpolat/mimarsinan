"""Named refusals for operands the converter cannot carry (generic ops, module calls, tuple outputs, concatenation)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

import torch.fx as fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal


class BatchDecision(Protocol):
    """An opaque batch value of the symbolic pass (``symbolic.pass_result.Decision``), as a refusal names it."""

    @property
    def node(self) -> fx.Node: ...

    @property
    def operator(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def detail(self) -> str: ...


def op_name(fn) -> str:
    return getattr(fn, "__name__", None) or type(fn).__name__


def node_operator(node: fx.Node, modules: Optional[Mapping[str, nn.Module]] = None) -> str:
    """The operator an FX node applies: a function's name (a ``getattr`` with its attribute, ``getattr .shape``), a
    method's name, or a called module's class."""
    if node.op == "call_module" and modules is not None and str(node.target) in modules:
        return type(modules[str(node.target)]).__name__
    if node.op == "call_function" and node.target is getattr and len(node.args) >= 2 and isinstance(node.args[1], str):
        return f"getattr .{node.args[1]}"
    return op_name(node.target) if callable(node.target) else str(node.target)


def decided_reason(node: fx.Node, decision: BatchDecision) -> str:
    """Why ``node``'s per-sample shape, fixed at conversion, holds at the traced batch only: the opaque batch value it
    is computed from (``symbolic/pass_result.Decision``: a value outside the affine batch domain, or a symbolic run on
    a batch-dependent value that raised)."""
    who = "this op" if decision.node is node else f"{decision.operator} (FX node {decision.node.name!r})"
    if decision.kind == "failed":
        how = f"through {who}, whose symbolic run on a run-time extent raised {decision.detail}"
    else:
        how = f"through {who}, {decision.detail}"
    return f"its shape depends on the batch {how}, so the per-sample shape fixed at conversion holds at the traced " \
           "batch only"


class GenericOpArgumentError(ToolchainRefusal, RuntimeError):
    """A generic host op's FX argument is neither an activation with a mapper nor a bound tensor."""

    def __init__(self, op_name: str, node_name: str, argument_name: str, argument_kind: str):
        self.op_name = op_name
        self.node_name = node_name
        self.argument_name = argument_name
        self.argument_kind = argument_kind
        super().__init__(
            f"generic op '{op_name}' (FX node '{node_name}'): argument "
            f"'{argument_name}' ({argument_kind}) resolves to neither an activation "
            "with a mapper nor a parameter/buffer tensor; dropping it would evaluate "
            "a different call, so the op is refused"
        )

    @classmethod
    def for_node(cls, fn, node: fx.Node, argument: fx.Node) -> "GenericOpArgumentError":
        return cls(op_name(fn), node.name, argument.name, f"{argument.op} {argument.target!r}")


class ModuleCallArgumentError(ToolchainRefusal, RuntimeError):
    """A module call passes an operand its conversion rule cannot carry; dropping it would evaluate a different call."""

    def __init__(self, module: nn.Module, node: fx.Node, reason: str):
        self.operator = type(module).__name__
        self.node_name = node.name
        self.reason = reason
        super().__init__(
            f"{self.operator} (FX node {node.name!r}, module {str(node.target)!r}): {reason}; "
            "dropping or recasting the operand would evaluate a different call, so the conversion is refused"
        )


class TupleOutputIndexError(ToolchainRefusal, RuntimeError):
    """A consumed output of a tuple-returning module other than the one its conversion carries."""

    def __init__(self, node: fx.Node, producer: fx.Node, index: Any, carried: int):
        self.node_name = node.name
        self.producer_name = producer.name
        super().__init__(
            f"getitem (FX node {node.name!r}) reads output index {index} of {str(producer.target)!r} "
            f"(FX node {producer.name!r}), whose conversion carries output index {carried} only; "
            "reading the carried output instead would evaluate a different program, so the conversion is refused"
        )


class BatchAxisLayoutError(ToolchainRefusal, RuntimeError):
    """A layout method whose input or output does not carry the batch on axis 0 alone (a per-sample program's), or
    whose dimension argument is known at run time only; a mapper fixing a per-sample shape at conversion that reads a
    value an opaque batch value reaches (``decided_reason``, review round 14)."""

    def __init__(self, node: fx.Node, reason: str):
        self.node_name = node.name
        self.reason = reason
        users = ", ".join(repr(user.name) for user in node.users) or "no node"
        operator = node_operator(node)
        super().__init__(
            f"{operator} (FX node {node.name!r}, used by {users}): {reason}; a per-sample program keeps the "
            "batch axis leading, so the conversion is refused"
        )


class ConcatenationOperandError(ToolchainRefusal, RuntimeError):
    """A ``torch.cat`` the per-sample program cannot express."""

    def __init__(self, node: fx.Node, reason: str):
        self.node_name = node.name
        self.reason = reason
        super().__init__(f"cat (FX node {node.name!r}): {reason}, so the conversion is refused")
