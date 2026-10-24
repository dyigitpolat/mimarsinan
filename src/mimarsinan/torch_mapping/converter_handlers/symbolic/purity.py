"""The purity precondition of every static shape proof (review round 9).

The symbolic-batch pass, its folds and its layout proofs read a program as torch's own semantics define it. They are
sound only for a program that carries no user code and no user type; such a program is PURE:

- (P1) every ``call_function``, ``call_method`` and ``call_module`` target is in the declared operator set
  (``declared_operators.opaque_reason``; a method or attribute is read on a value whose run-time type, recorded by the
  tracer's shape propagation, is exactly ``torch.Tensor``, ``torch.nn.Parameter`` or ``torch.Size``), and the declared
  operators are torch's own code as imported (``imported_code.torch_rebinding``: no binding of a declared namespace or
  class rebound and none added to a declared class, no code binding rebound among the globals of a module defining
  declared code, no function's code, defaults or closure cells replaced, no mode of a class other than torch's own
  active; and ``module_call.call_code_change``: ``nn.Module``'s call and the globals it reads as imported);
- (P2) every literal operand (recursively), every ``get_attr`` value, and every parameter, buffer and attribute of a
  called module has a declared value type by exact type (``declared_operators.foreign_value``), and a called module's
  class is a declared leaf by exact type, its tables of the types ``nn.Module`` installs, without forward hooks (read as
  its call reads them, ``module_call.forward_hooks``) (``declared_operators.foreign_module``).

The traced program is the root's call: the tracer refuses a model whose call runs code FX does not trace
(``module_call.call_reason``: a hook on the root or a global module hook, an instance ``forward``, a class's own
``__call__``, a name the call reads bound by the class, a global it reads rebound; ``RootCallError``), so no conversion
path needs to represent it.

A pure program keeps the static machinery (the symbolic pass, static folds, layout proofs, the opaque batch values by
data flow). An IMPURE program has none of it, so its converted flow computes what the native model computes:

- the symbolic pass never runs, and no shape is proved or folded: every shape value is its run-time host op;
- every layout operation whose value depends on the input is the host op of its native method on its run-time
  operands (``LayoutMixin._native_layout``);
- a called module outside the declared set is the host op of its own call: no perceptron packaging, BatchNorm folding,
  absorption or Linear fusion reads its state; while torch's declared code is not the one imported, no conversion
  rule is its semantics, so every called module is;
- a host op takes its bound tensors at their native shapes and never runs at conversion
  (``MapperGraphFxMixin._batch_contract_adapter``);
- stage 2 refuses the value program by name (``ImpureShapeProgramError``), and the certificate records ``no static
  shape proof: impure program at <operator> (FX node <name>) (<reason>)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Mapping, Optional

import torch.fx as fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.converter_handlers.refusals import node_operator
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import (
    foreign_module,
    foreign_value,
    opaque_reason,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.imported_code import torch_rebinding
from mimarsinan.torch_mapping.converter_handlers.symbolic.module_call import call_code_change

__all__ = [
    "ImpureShapeProgramError", "Impurity", "ProgramPurity", "attribute_at", "declared_code_change", "node_impurity",
    "program_purity", "receiver_type",
]

_CALLS = ("call_function", "call_method", "call_module")


@dataclass(frozen=True)
class Impurity:
    """The first node (traced order) that makes a program impure: its FX node, operator and why."""

    node: str
    operator: str
    reason: str

    def statement(self) -> str:
        """The certificate's shape statement of an impure program."""
        return f"no static shape proof: impure program at {self.operator} (FX node {self.node!r}) ({self.reason})"


@dataclass(frozen=True)
class ProgramPurity:
    """``impurity``: the first impure node, ``None`` for a pure program; ``foreign_modules``: the ``call_module``
    targets outside the declared set (each converts as the host op of its own call)."""

    impurity: Optional[Impurity] = None
    foreign_modules: FrozenSet[str] = frozenset()

    @property
    def pure(self) -> bool:
        return self.impurity is None


def attribute_at(root: nn.Module, target: str) -> Any:
    """The attribute of ``root`` at the dotted path ``target`` (a ``get_attr`` or ``call_module`` target)."""
    value: Any = root
    for part in target.split("."):
        value = getattr(value, part)
    return value


def receiver_type(node: fx.Node) -> Optional[type]:
    """The run-time type of ``node``'s first argument, as the tracer's shape propagation recorded it."""
    receiver = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
    return receiver.meta.get("type") if receiver is not None else None


def declared_code_change() -> Optional[str]:
    """How torch's declared code or ``nn.Module``'s call differs from the one imported; ``None`` when neither does."""
    return torch_rebinding() or call_code_change()


def node_impurity(gm: fx.GraphModule, node: fx.Node, receiver_type: Optional[type], modules: Mapping[str, Any],
                  rebound: Optional[str]) -> Optional[str]:
    """Why ``node`` makes ``gm`` impure (P1, P2; ``receiver_type``: the run-time type of its first argument,
    ``rebound``: ``declared_code_change()``); ``None`` when it does not."""
    if node.op == "get_attr":
        found = foreign_value(attribute_at(gm, str(node.target)))
        return None if found is None else f"a get_attr value that is {found}"
    if node.op not in _CALLS:
        return None
    if rebound is not None:
        return f"the declared operators are not torch's own code as imported: {rebound}"
    return opaque_reason(node, receiver_type, modules)


def program_purity(gm: fx.GraphModule) -> ProgramPurity:
    """P1 and P2 over every node of ``gm`` (the traced program, before graph normalization rewrites any module); a
    rebound declared operator is named at the program's first call, and makes every called module foreign."""
    modules = dict(gm.named_modules())
    rebound = declared_code_change()
    first: Optional[Impurity] = None
    foreign_modules = set()
    for node in gm.graph.nodes:
        reason = node_impurity(gm, node, receiver_type(node), modules, rebound)
        module = modules.get(str(node.target)) if node.op == "call_module" else None
        if module is not None and (rebound is not None or foreign_module(module) is not None):
            foreign_modules.add(str(node.target))
        if reason is not None and first is None:
            first = Impurity(node.name, node_operator(node, modules), reason)
    return ProgramPurity(first, frozenset(foreign_modules))


class ImpureShapeProgramError(ToolchainRefusal, ValueError):
    """Stage 2 of an impure program: no per-sample shape of it is proved, and a value program fixes every one."""

    def __init__(self, impurity: Impurity):
        self.node_name = impurity.node
        self.operator = impurity.operator
        self.reason = impurity.reason
        super().__init__(
            f"{impurity.operator} (FX node {impurity.node!r}): {impurity.reason}; the program is impure, so no "
            "per-sample shape of it is proved, and a value program, which fixes every per-sample shape at "
            "conversion, is refused"
        )
