"""A captured body's writes to tensors it allocates itself, computed out of place (coverage W8).

A write in place in a body stays the opaque call's own effect (``body_capture`` rule (d)), with one exception made
here: an ASSIGNMENT to a LOCAL tensor. An assignment is ``operator.setitem(t, index, value)``, ``t[index].zero_()``,
``t[index].fill_(value)``, ``t.zero_()`` or ``t.fill_(value)``, whose result is not read, with ``index`` a literal
basic index (integers and slices whose bounds and steps are integer literals or ``None``, at most one per axis) and
``value`` a literal number. A tensor ``t`` is local when the body allocates it: a call whose aten schema declares no
alias and no write (``operand_writes.schema_declares_no_alias``: a factory, ``clone``, an arithmetic kernel) or a
Python arithmetic operator on tensors, and whose value in the capture (a FakeTensor, which keeps storage identity)
holds the storage of no value before it (``x.type_as(x)`` is ``x``). Every assignment to ``t`` is
rewritten when no node reads ``t``, or a view of it, before ``t``'s last assignment, other than the assignments
themselves (``u = t[0]`` read after ``t[0, 0] = 5`` sees the write natively).

The out-of-place form (``assignment_nodes``) of ``t[e_0, ..., e_k] = v``: the views ``t_0 = t``,
``t_{j+1} = t_j[(:,) * a_j + (e_j,)]`` (``a_j``: the axis entry ``j`` indexes in ``t_j``, one more after a slice, the
same after an integer), the new innermost value ``torch.full_like(t_{k+1}, v)``, and outward, for each ``j`` from ``k``
to 0, ``torch.slice_scatter(t_j, new, a_j, start, stop, step)`` for a slice and ``torch.select_scatter(t_j, new, a_j,
e_j)`` for an integer; a whole-axis entry ``:`` is its operand's view and its source's scatter, so it adds neither (it
never scatters along the batch axis of ``y[:, k] = v``). Every later read of ``t`` reads its last assignment's result.

Lemma. Under T1 (torch as shipped), the rewritten body computes, at every node the original body's later nodes read,
the value the original computes there. ``t[index] = v`` with a basic index writes ``v``, converted to ``t``'s dtype as
``fill_`` converts it, into the view ``t[index]`` and changes no other element of ``t``; ``full_like`` converts ``v`` as
``fill_`` does, and ``slice_scatter`` (``select_scatter``) returns its input with the slice (the index) along ``dim``
replaced by its source, the same normalization of bounds as indexing, so the nested form equals ``t`` after the write
(induction over the entries, from the innermost). No other node observes ``t`` between its allocation and its last
assignment, ``t`` holds no storage of an operand, and the assignment's own result is not read, so no read is of a value
the original wrote and the rewrite does not; reads after the last assignment read the rewritten value. Nothing is
written in place any more, so rule (d) holds for the rewritten body.
"""

from __future__ import annotations

import contextlib
import operator
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

import torch
import torch.fx as fx
from torch.fx.node import map_arg
from torch.multiprocessing.reductions import StorageWeakRef

from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import declared_writes
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.declared import (
    schema_declares_no_alias,
    view_aliases,
)

__all__ = ["Assignment", "assignment_nodes", "rewrite_assignments"]

_ARITHMETIC = frozenset({operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod,
                         operator.pow, operator.neg, operator.matmul, operator.abs})
"""Python's operators on tensors: each returns a new tensor (``torch.Tensor.__add__`` and the others)."""

_FILLS = {"zero_": 0, "fill_": None}


class Assignment(NamedTuple):
    """``node`` assigns the literal ``value`` to ``base[where]``; ``view`` is the ``getitem`` node it writes through
    (``t[index].zero_()``), ``None`` for ``setitem`` and a write of ``base`` itself."""

    node: fx.Node
    base: fx.Node
    where: Tuple[Any, ...]
    value: Any
    view: Optional[fx.Node]


def _number(value: Any) -> bool:
    return type(value) in (int, float, bool)


def _literal_index(index: Any) -> Optional[Tuple[Any, ...]]:
    """``index`` as a tuple of basic entries (``int``, a ``slice`` of literal ``int`` or ``None`` bounds and a positive
    step), else ``None``."""
    entries = index if type(index) is tuple else (index,)
    for entry in entries:
        bounds = (entry.start, entry.stop, entry.step) if type(entry) is slice else (entry,)
        if type(entry) not in (int, slice) or not all(b is None or type(b) is int for b in bounds):
            return None
        if type(entry) is slice and entry.step is not None and entry.step < 1:
            return None
    return tuple(entries)


def _assignment(node: fx.Node) -> Optional[Assignment]:
    """``node`` as an assignment (the module docstring) when it is one, before the base is judged; else ``None``."""
    if node.op == "call_function" and node.target is operator.setitem and len(node.args) == 3 and not node.kwargs:
        base, index, value = node.args
        found = _literal_index(index)
        if isinstance(base, fx.Node) and found is not None and _number(value):
            return Assignment(node, base, found, value, None)
        return None
    if node.op != "call_method" or node.target not in _FILLS or node.kwargs or not isinstance(node.args[0], fx.Node):
        return None
    value = _FILLS[str(node.target)] if node.target == "zero_" else (node.args[1] if len(node.args) == 2 else None)
    if len(node.args) != (1 if node.target == "zero_" else 2) or not _number(value):
        return None
    target = node.args[0]
    if target.op == "call_function" and target.target is operator.getitem and isinstance(target.args[0], fx.Node):
        found = _literal_index(target.args[1])
        return None if found is None else Assignment(node, target.args[0], found, value, target)
    return Assignment(node, target, (), value, None)


def _storage(value: Any) -> Optional[StorageWeakRef]:
    return StorageWeakRef(value.untyped_storage()) if isinstance(value, torch.Tensor) else None


def _local(node: fx.Node, earlier: List[fx.Node]) -> bool:
    """``node`` allocates a new tensor: a call whose schema declares no alias and no write, or a Python arithmetic
    operator, whose value (Dynamo's FakeTensor) holds no storage of a value before it (``earlier``): ``type_as`` and
    ``to_dense`` declare no alias and return their operand where its type already fits."""
    value = node.meta.get("example_value")
    if not isinstance(value, torch.Tensor) or node.op not in ("call_function", "call_method"):
        return False
    own = _storage(value)
    if any(found is not None and found == own for found in (_storage(n.meta.get("example_value")) for n in earlier)):
        return False
    if node.op == "call_function" and node.target in _ARITHMETIC:
        return True
    return schema_declares_no_alias(node) and not view_aliases(node) and not declared_writes(node, {})


def _whole(entry: Any) -> bool:
    """A slice of every element of its axis (``:``): its view is its operand and its scatter is its source."""
    return type(entry) is slice and entry.start is None and entry.stop is None and entry.step in (None, 1)


def assignment_nodes(graph: fx.Graph, base: fx.Node, index: Sequence[Any], value: Any) -> fx.Node:
    """The out-of-place form of ``base[index] = value`` (the module docstring), added to ``graph`` at its insertion
    point; returns the new value of ``base``. A whole-axis entry (``:``) adds no view and no scatter."""
    views, axes, axis = [base], [], 0
    for entry in index:
        axes.append(axis)
        views.append(views[-1] if _whole(entry) else graph.call_function(
            operator.getitem, (views[-1], (slice(None),) * axis + (entry,))))
        axis += 1 if type(entry) is slice else 0
    new = graph.call_function(torch.full_like, (views[-1], value))
    for entry, parent, at in reversed(list(zip(index, views, axes))):
        if _whole(entry):
            continue
        if type(entry) is slice:
            new = graph.call_function(torch.slice_scatter, (parent, new), {
                "dim": at, "start": entry.start, "end": entry.stop, "step": 1 if entry.step is None else entry.step})
        else:
            new = graph.call_function(torch.select_scatter, (parent, new, at, entry))
    return new


def _plan(gm: fx.GraphModule) -> Tuple[Dict[fx.Node, List[Assignment]], Optional[str]]:
    """Every local tensor's assignments in graph order, or the reason the first assignment-form write is not taken."""
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    planned: Dict[fx.Node, List[Assignment]] = {}
    for node in gm.graph.nodes:
        found = _assignment(node)
        if found is None:
            if node.op == "call_function" and node.target is operator.setitem:
                return {}, (f"its body writes in place ({node.name}: setitem, at an index or of a value that is not "
                            "a literal)")
            continue
        planned.setdefault(found.base, []).append(found)
    for base, assignments in planned.items():
        where = f"its body writes in place ({assignments[0].node.name}: {_name(assignments[0].node)}"
        if not _local(base, [node for node in gm.graph.nodes if order[node] < order[base]]):
            return {}, f"{where}, a tensor the body does not allocate)"
        rank = len(base.meta["example_value"].shape)
        if any(len(a.where) > rank or a.node.users or (a.view is not None and len(a.view.users) != 1)
               for a in assignments):
            return {}, f"{where}, at an index or through a view the rewrite does not take)"
        last = order[assignments[-1].node]
        writes = {a.node for a in assignments} | {a.view for a in assignments if a.view is not None}
        early = [user for user in base.users if user not in writes and order[user] < last]
        if early:
            return {}, f"{where}, and {early[0].name} reads it before its last write)"
    return planned, None


def _name(node: fx.Node) -> str:
    return "setitem" if node.target is operator.setitem else str(node.target)


def _example(node: fx.Node) -> None:
    """``node``'s Dynamo metadata (``example_value``), from its call on its operands' FakeTensors: what the capture
    records of every node it makes, which the declared-operator check reads (a method's receiver type)."""
    args, kwargs = map_arg((node.args, node.kwargs), lambda arg: arg.meta["example_value"])
    fake = next(value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor))
    mode = getattr(fake, "fake_mode", None)
    shape_env = getattr(mode, "shape_env", None)
    with mode or contextlib.nullcontext(), shape_env.suppress_guards() if shape_env else contextlib.nullcontext():
        node.meta["example_value"] = cast(Any, node.target)(*args, **kwargs)  # no guard joins the capture's


def rewrite_assignments(gm: fx.GraphModule) -> Optional[str]:
    """Every assignment to a local tensor of ``gm`` (a captured body) out of place, when every assignment-form write
    is one; else ``gm`` unchanged and the reason the call stays opaque."""
    planned, reason = _plan(gm)
    if reason is not None:
        return reason
    graph, made = gm.graph, []
    for base, assignments in planned.items():
        readers = [user for user in base.users if user not in {a.node for a in assignments} | {
            a.view for a in assignments}]
        current = base
        for found in assignments:
            before = set(graph.nodes)
            with graph.inserting_before(found.node):
                current = assignment_nodes(graph, current, found.where, found.value)
            made += [node for node in graph.nodes if node not in before]
            graph.erase_node(found.node)
            if found.view is not None:
                graph.erase_node(found.view)
        for reader in readers:
            reader.replace_input_with(base, current)
    for node in made:
        _example(node)
    if planned:
        gm.recompile()
    return None
