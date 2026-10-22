"""The rule of operand writes on a traced graph (review rounds 15 and 16; ``__init__.py`` states it and its lemma):
``define_writes`` makes every declared write a definition of its operand before the shape propagation runs, and
``settle_writes`` reads that run (``observed.WriteRecord``): every write, and every call outside the declared operator
set, runs after the nodes that read its operands' storage before it (``RUN_AFTER``); every reader of an operand after a
call whose write the run observes reads it through a value read after the call (``read_after``); any other read of a
value sharing a written operand's storage after the write is refused by name.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import torch.fx as fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.support.compute_modules import read_after
from mimarsinan.torch_mapping.converter_handlers.refusals import node_operator
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import opaque_reason
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.declared import (
    Write,
    declared_aliases,
    declared_writes,
    view_aliases,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.observed import WriteRecord

__all__ = [
    "COPY_GUARD", "KEPT_OPERANDS", "OperandWriteError", "RUN_AFTER", "check_record", "define_writes", "settle_writes",
]

COPY_GUARD = "mimarsinan_copy_guard"
"""The node metadata key of a call torch's schema lets share its operand's storage, which the traced program's run
shows copying, and on which the rule of a write relies (review round 16): ``(operand name, the operand's strides in
that run)``. The call is the host op of its native method, which checks at run time that its operand has those strides
and that its result is again a copy (``ComputeOpMapper.copy_guard``, ``compute_modules.HostOpAliasError``)."""

KEPT_OPERANDS = "mimarsinan_kept_operands"
"""The node metadata key of a call outside the declared operator set: the operands another node reads after it, which
its host op must not write in place at run time (``ComputeOpMapper.kept_inputs``); an operand whose write the run
observes is read after the call through ``read_after`` instead."""

RUN_AFTER = "mimarsinan_run_after"
"""The node metadata key of a write in place (declared or observed) and of a call outside the declared operator set:
the nodes that read a value sharing its operands' storage before it in traced order, which its host op runs after in
the converted model (``Mapper.run_after``, review round 16, targeted pass: it acted on a copy, which keeps the values
and not the layout or the storage), so a write acts on its operand itself after every earlier read, as natively."""

_CALLS = ("call_function", "call_method", "call_module")


class OperandWriteError(ToolchainRefusal):
    """A write of a tensor operand in place that the conversion cannot take as a definition of that operand: a value
    sharing the operand's storage other than the operand itself is read after the write, or the write is undeclared
    and not a call outside the declared operator set. ``operator`` and ``node_name`` name the writing call."""

    def __init__(self, operator: str, node_name: str, reason: str):
        self.operator, self.node_name, self.reason = operator, node_name, reason
        super().__init__(f"{operator} (FX node {node_name!r}): {reason}; the conversion computes every value once, "
                         "at its own node, so the model is refused")


def _order(gm: fx.GraphModule) -> Dict[fx.Node, int]:
    return {node: index for index, node in enumerate(gm.graph.nodes)}


def define_writes(gm: fx.GraphModule) -> List[Write]:
    """Rule 1: every node after a declared write that returns its operand reads the write's result in place of the
    operand (the same tensor, natively). Returns the declared writes in traced order."""
    modules, order, found = dict(gm.named_modules()), _order(gm), []
    for node in list(gm.graph.nodes):
        for write in declared_writes(node, modules):
            found.append(write)
            if not write.returned:
                continue
            for user in list(write.operand.users):
                if user is not node and order[user] > order[node]:
                    user.replace_input_with(write.operand, node)
    gm.recompile()
    return found


Edges = Dict[fx.Node, Set[fx.Node]]


def _link(edges: Edges, a: fx.Node, b: fx.Node) -> None:
    edges.setdefault(a, set()).add(b)
    edges.setdefault(b, set()).add(a)


def _alias_edges(gm: fx.GraphModule, modules: Mapping[str, nn.Module], record: WriteRecord) -> Tuple[Edges, Edges]:
    """The storage edges (a node and an operand its result may share storage with: torch's declarations and the
    run's sharing), and the COPY edges among them: a call torch's declaration lets share its operand's storage
    (``view_aliases``) whose result the run shows in another storage (review round 16: ``reshape`` of a permuted view,
    ``contiguous`` of a transposed one). A copy edge is no storage edge; a rule that relies on one guards it."""
    edges: Edges = {}
    copies: Edges = {}
    for node in gm.graph.nodes:
        shared = record.shares.get(node, [])
        views = view_aliases(node)
        for operand in [*declared_aliases(node, modules), *shared]:
            copied = (operand in views and operand not in shared and node.op in ("call_method", "call_function")
                      and node in record.strides and operand in record.strides)
            _link(copies if copied else edges, node, operand)
    return edges, copies


def _after(node: fx.Node, writer: fx.Node) -> bool:
    """``node`` is ``writer``'s result or the value read after it (``read_after(operand, writer)``)."""
    return node is writer or (node.target is read_after and writer in node.args[1:])


def _sharing(operand: fx.Node, writer: fx.Node, edges: Mapping[fx.Node, Set[fx.Node]]) -> Set[fx.Node]:
    """The values that may share ``operand``'s storage other than through ``writer``'s result."""
    found, stack = {operand}, [operand]
    while stack:
        for other in edges.get(stack.pop(), ()):
            if not _after(other, writer) and other not in found:
                found.add(other)
                stack.append(other)
    return found


def _relied(shared: Set[fx.Node], writer: fx.Node, edges: Edges, copies: Edges,
            guards: Set[Tuple[fx.Node, fx.Node]]) -> None:
    """The copy edges that separate ``shared`` (a closure of ``edges``) from values it would share storage with
    through them: where the closure through them has other readers, the rule relies on each such copy (``guards``:
    (call, operand))."""
    crossing = [(a, b) for a in shared for b in copies.get(a, ()) if b not in shared and b is not writer]
    if not crossing:
        return
    merged = {node: edges.get(node, set()) | copies.get(node, set()) for node in {*edges, *copies}}
    whole = _sharing(next(iter(shared)), writer, merged)
    if set(_readers(whole, writer)) != set(_readers(shared, writer)):
        guards.update((a, b) if b in a.all_input_nodes else (b, a) for a, b in crossing)


def _guard(guards: Set[Tuple[fx.Node, fx.Node]], record: WriteRecord) -> None:
    """Each copy a rule relies on is checked at run time: the call's host op copies again (``COPY_GUARD``)."""
    for node, operand in guards:
        node.meta[COPY_GUARD] = (operand.name, record.strides[operand])


def _readers(shared: Set[fx.Node], writer: fx.Node) -> List[Tuple[fx.Node, fx.Node]]:
    """``(reader, value)``: a node other than the writer (and the value read after it) that reads a value of
    ``shared`` and is not itself one."""
    return [(user, value) for value in shared for user in value.users
            if not _after(user, writer) and user not in shared]


_OBSERVED = "an undeclared write (observed in the traced program's run at batch 1) of"


def _read_after_reason(what: str, operand: fx.Node, user: fx.Node, value: fx.Node,
                       modules: Mapping[str, nn.Module]) -> str:
    return (f"{what} its tensor operand {operand.name!r} in place, and {node_operator(user, modules)} (FX node "
            f"{user.name!r}) reads {value.name!r}, which shares that operand's storage, after the write")


def _run_after(node: fx.Node, readers: Iterable[fx.Node], order: Mapping[fx.Node, int]) -> None:
    """``node``'s host op runs after ``readers``' in the converted model (``RUN_AFTER``, names in traced order)."""
    found = {*node.meta.get(RUN_AFTER, ()), *(reader.name for reader in readers)}
    if found:
        names = {n.name: n for n in order}
        node.meta[RUN_AFTER] = tuple(sorted(found, key=lambda name: order[names[name]]))


def _current(node: fx.Node, operand: fx.Node, routed: Mapping[fx.Node, fx.Node]) -> fx.Node:
    """The value ``node`` reads in place of ``operand`` once earlier observed writes routed it (``read_after``)."""
    while operand not in node.all_input_nodes and operand in routed:
        operand = routed[operand]
    return operand


def _route(gm: fx.GraphModule, write: Write, shared: Set[fx.Node], modules: Mapping[str, nn.Module],
           edges: Edges) -> Optional[fx.Node]:
    """Every node after ``write`` (an observed write of a call outside the declared set) that reads its operand reads
    ``read_after(operand, call)`` instead, a value ordered after the call (the same tensor); a node that reads another
    value of the operand's storage defined before the write, after it, is refused by name. Returns the routed value
    (``None``: no node reads the operand after the write)."""
    order = _order(gm)
    before = {value for value in shared if order[value] < order[write.node] and value is not write.operand}
    for node in list(gm.graph.nodes)[order[write.node] + 1:]:
        value = next((arg for arg in node.all_input_nodes if arg in before), None)
        if value is not None:
            raise OperandWriteError(node_operator(write.node, modules), write.node.name,
                                    _read_after_reason(_OBSERVED, write.operand, node, value, modules))
    later = [user for user in write.operand.users if order[user] > order[write.node]]
    if not later:
        return None
    with gm.graph.inserting_after(write.node):
        routed = gm.graph.call_function(read_after, (write.operand, write.node))
    for user in later:
        user.replace_input_with(write.operand, routed)
    _link(edges, routed, write.operand)
    return routed


def settle_writes(gm: fx.GraphModule, writes: List[Write], record: WriteRecord) -> Tuple[List[Write], bool]:
    """Rules 2 and 3 on the run ``record`` of the graph ``define_writes`` rewrote: the declared writes as they stand,
    and True when a reader was routed through ``read_after`` (the shape propagation then runs again)."""
    modules, order = dict(gm.named_modules()), _order(gm)
    edges, copies = _alias_edges(gm, modules, record)
    guards: Set[Tuple[fx.Node, fx.Node]] = set()
    declared = {(write.node, write.operand) for write in writes}
    observed = [Write(node, operand, False) for node, operands in record.written.items() for operand in operands
                if (node, operand) not in declared]
    routed: Dict[fx.Node, fx.Node] = {}
    accepted: Set[Tuple[fx.Node, fx.Node]] = set()
    settled: Dict[Write, Write] = {}
    for write in sorted([*writes, *observed], key=lambda found: order[found.node]):
        is_declared = (write.node, write.operand) in declared
        current = Write(write.node, _current(write.node, write.operand, routed), write.returned)
        order = _order(gm)
        shared = _sharing(current.operand, current.node, edges)
        _relied(shared, current.node, edges, copies, guards)
        readers = sorted(_readers(shared, current.node), key=lambda found: order[found[0]])
        later = [(user, value) for user, value in readers if order[user] > order[current.node]]
        what = "a declared write of" if is_declared else _OBSERVED
        opaque = not is_declared and _opaque(current.node, modules)
        if later and not opaque:
            raise OperandWriteError(node_operator(current.node, modules), current.node.name,
                                    _read_after_reason(what, current.operand, *later[0], modules))
        if readers and is_declared and not current.returned:
            user, value = readers[0]
            raise OperandWriteError(node_operator(current.node, modules), current.node.name, (
                f"{what} its tensor operand {current.operand.name!r} in place, which {node_operator(user, modules)} "
                f"(FX node {user.name!r}) reads (as {value.name!r}) before the write"))
        found = _route(gm, current, shared, modules, edges) if opaque else None
        if found is not None:
            routed[current.operand] = found
            accepted.add((current.node, current.operand))
        _run_after(current.node, [user for user, _ in readers if order[user] < order[current.node]], _order(gm))
        settled[write] = current
    if routed:
        gm.recompile()
    _keep_operands(gm, modules, edges, copies, guards, accepted)
    _guard(guards, record)
    return [settled.get(write, write) for write in writes], bool(routed)


def _opaque(node: fx.Node, modules: Mapping[str, nn.Module]) -> bool:
    """``node`` calls outside the declared operator set (a user helper, a module the trace does not enter); the
    toolchain's own ``read_after`` is none."""
    receiver = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
    return node.op in _CALLS and node.target is not read_after and opaque_reason(
        node, receiver.meta.get("type") if receiver is not None else None, modules) is not None


def _tensor_valued(node: fx.Node) -> bool:
    operand = node.args[0] if node.target is read_after and node.args else None
    return bool(node.meta.get("tensor_meta")) or (isinstance(operand, fx.Node) and _tensor_valued(operand))


def _keep_operands(gm: fx.GraphModule, modules: Mapping[str, nn.Module], edges: Edges, copies: Edges,
                   guards: Set[Tuple[fx.Node, fx.Node]], accepted: Set[Tuple[fx.Node, fx.Node]]) -> None:
    """Rule 3 at run time: a call outside the declared operator set may write an operand at another batch than the
    run's. It runs after every node that reads a value of its operand's storage before it (``RUN_AFTER``); an operand
    another node reads after it, whose write the run does not observe, is kept (``KEPT_OPERANDS``: the host op refuses
    a write of it by name)."""
    order = _order(gm)
    for node in gm.graph.nodes:
        if not _opaque(node, modules):
            continue
        kept = []
        for operand in node.all_input_nodes:
            shared = _sharing(operand, node, edges) if _tensor_valued(operand) else set()
            _relied(shared, node, edges, copies, guards)
            readers = _readers(shared, node)
            _run_after(node, [user for user, _ in readers if order[user] < order[node]], order)
            if any(order[user] > order[node] for user, _ in readers) and (node, operand) not in accepted:
                kept.append(operand.name)
        if kept:
            node.meta[KEPT_OPERANDS] = tuple(kept)


def _read_after_write(user: fx.Node, value: fx.Node, record: WriteRecord) -> bool:
    """``user`` reads ``value`` after a call it is ordered after wrote it (``read_after``, observed in ``record``)."""
    return user.target is read_after and user.args[:1] == (value,) and any(
        value in record.written.get(call, ()) for call in user.args[1:] if isinstance(call, fx.Node))


def check_record(gm: fx.GraphModule, writes: List[Write], record: WriteRecord) -> None:
    """The final run of the rewritten graph: no node reads a value changed in place since its node ran but a value
    read after the call that wrote it (``read_after``), and every declared write taken as a definition returned its
    operand."""
    modules = dict(gm.named_modules())
    stale = [(user, value) for user, value in record.stale if not _read_after_write(user, value, record)]
    if stale:
        user, value = stale[0]
        raise OperandWriteError(node_operator(user, modules), user.name, (
            f"it reads {value.name!r}, whose value changed in place after its node ran (observed in the traced "
            "program's run at batch 1)"))
    for write in writes:
        if write.returned and write.operand not in record.identical.get(write.node, []):
            raise OperandWriteError(node_operator(write.node, modules), write.node.name, (
                f"a declared write of its tensor operand {write.operand.name!r} whose result is not that operand"))
