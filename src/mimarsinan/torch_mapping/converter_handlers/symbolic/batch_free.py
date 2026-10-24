"""Batch-free values (coverage W6): a value computed from the program's state and literals only has no batch axis.

A call node is BATCH-FREE when no placeholder lies upstream of it (but through the receiver of a metadata read, below)
and every node upstream of it, itself included, is a
``get_attr`` of a declared value or a declared call (``purity.node_impurity``) that writes in place no value of the
state (a write torch declares, ``operand_writes.declared_writes``, or an in-place operator's name, ``in_place_form``, of
a ``get_attr`` or a value that may view one; a write of a tensor the computation itself made, an in-place ReLU, is
computed again with it) and draws no random number (an aten operator of its name tagged ``nondeterministic_seeded``, or,
on the program's run at conversion, any change of the random state while it runs); and when no node of the program
touches that computation other than by reading it: none writes the state it reads, or a value that views it; no call
outside the declared set reads that state (its effects are its own, T3), itself or through a value sharing its storage
(the tracer marks such a call, ``KEPT_OPERANDS`` where a node reads that storage after it, ``RUN_AFTER`` naming the
nodes that read it before, on the run's alias edges); none reads one of its values through
``read_after`` (a write an opaque call was seen making), is ordered around one (``RUN_AFTER``), keeps one
(``KEPT_OPERANDS``) or guards one as a copy (``COPY_GUARD``). A value so touched excludes the whole computation of it.

A METADATA READ (coverage W8) is a receiver factory (``method_rules.RECEIVER_FACTORIES``: ``x.new_zeros(size)``,
``new_ones``, ``new_full``) whose every operand but the receiver is a literal: its value is a function of the literals
and of the receiver's dtype and device alone, which are the program's (the converter follows the source model's dtype
on its device, and a body Dynamo captured is specialized to them). Its receiver is not part of the computation: a
metadata read is batch-free whatever declared operators compute its receiver (a receiver a call outside the declared
set reaches may change its dtype at run time, so it does not count), and the converted model computes it on a
zero-element stand-in of the receiver's traced dtype on the program's device, a buffer of the shared state in a child
named apart from every attribute of the program (no copied value shares its path), so it follows the converted model's
casts and moves as the state does (Swin's attention mask, ``x.new_zeros((pad_H, pad_W))`` filled at literal indices,
computed out of place by ``fx_rewrites/body_rules/local_writes``).

Lemma. Under the trust base (T1-T3) and trace fidelity, a batch-free value is, at every batch and input of a call, the
value its traced operators compute from the state as it stands at the start of the call (and from the program's dtype
and device, for a metadata read: its stand-in has them): no placeholder lies upstream,
so no input value or extent reaches it, and its shape is its traced shape; no node of the program writes that state,
so every one of its nodes reads the same state whenever it runs within the call; no node reads one of its values after
a write of it but through the write's own result (``define_writes``; the tracer refuses any other read of a written
storage), so every write a reader sees is a write upstream of that reader, computed again with it; its operators are
torch's own, deterministic. Computing a value again by the same operators from the same state, when any of its readers
runs, therefore gives the value that reader reads natively. A computation for which any condition fails keeps the
base's path (a host op, or a refusal by name).

The converter maps no tensor-valued batch-free node: every reader takes it as a bound operand computed at every call
(``mapping/support/batch_free_value.BatchFreeValue``) from the converted model's own copy of the state it reads, one
copy shared by every such value (``BatchFreeValues``), so gradients reach that state and ties hold; the deployed
program holds it as a constant (``deployed_payload``).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import map_arg

from mimarsinan.mapping.support.batch_free_value import BatchFreeValue
from mimarsinan.mapping.support.compute_modules import read_after
from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import RECEIVER_FACTORIES
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import (
    COPY_GUARD,
    KEPT_OPERANDS,
    RUN_AFTER,
    declared_writes,
    in_place_form,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.declared import view_aliases
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import (
    attribute_at,
    declared_code_change,
    node_impurity,
    receiver_type,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import graph_device, rng_neutral

__all__ = ["BatchFreeValues", "batch_free_nodes", "metadata_read"]

_CALLS = ("call_function", "call_method", "call_module")
_MARKS = (RUN_AFTER, KEPT_OPERANDS, COPY_GUARD)


def _closure(start: Iterable[fx.Node], step: Callable[[fx.Node], Iterable[fx.Node]]) -> Set[fx.Node]:
    """``start`` and every node ``step`` reaches from it, transitively."""
    found: Set[fx.Node] = set()
    stack = list(start)
    while stack:
        current = stack.pop()
        if current not in found:
            found.add(current)
            stack.extend(step(current))
    return found


def _seeded(node: fx.Node) -> bool:
    """An aten operator of the call's name draws random numbers (``torch.Tag.nondeterministic_seeded``)."""
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        return torch.Tag.nondeterministic_seeded in target.tags
    name = target if node.op == "call_method" else getattr(target, "__name__", None)
    try:
        packet = getattr(torch.ops.aten, str(name)) if isinstance(name, str) and name.isidentifier() else None
    except (AttributeError, RuntimeError):
        packet = None
    return packet is not None and any(
        torch.Tag.nondeterministic_seeded in getattr(packet, overload).tags for overload in packet.overloads())


def _written(node: fx.Node, modules: Mapping[str, nn.Module]) -> List[fx.Node]:
    found = [write.operand for write in declared_writes(node, modules)] + [in_place_form(node)]
    return [operand for operand in found if operand is not None]


def _writes_state(node: fx.Node, modules: Mapping[str, nn.Module]) -> Set[fx.Node]:
    """What ``node`` writes of the state: every value it writes in place that may view a ``get_attr``, with the values
    it views; empty for a write of a tensor the program made."""
    found: Set[fx.Node] = set()
    for operand in _written(node, modules):
        viewed = _closure([operand], view_aliases)  # the operand and every value it may view, up to the state
        found |= viewed if any(value.op == "get_attr" for value in viewed) else set()
    return found


def _touched(gm: fx.GraphModule, modules: Mapping[str, nn.Module], rebound: Optional[str]) -> Set[fx.Node]:
    """The state a node writes in place (with the values viewing it), the state a call outside the declared set reads,
    and the values a node reads through ``read_after``, is ordered around or keeps."""
    names = {node.name: node for node in gm.graph.nodes}
    touched: Set[fx.Node] = set()
    for node in gm.graph.nodes:
        touched |= _writes_state(node, modules)
        if node.target is read_after and node.args and isinstance(node.args[0], fx.Node):
            touched.add(node.args[0])
        for mark in (RUN_AFTER, KEPT_OPERANDS):
            touched |= {names[name] for name in node.meta.get(mark, ()) if name in names}
        if node.op in _CALLS and node_impurity(gm, node, receiver_type(node), modules, rebound) is not None:
            touched |= {arg for arg in node.all_input_nodes if arg.op == "get_attr"}
    return touched


def _declared_deterministic(gm: fx.GraphModule, node: fx.Node, modules: Mapping[str, nn.Module],
                            rebound: Optional[str]) -> bool:
    return (node.target is not read_after and node_impurity(gm, node, receiver_type(node), modules, rebound) is None
            and not _writes_state(node, modules) and not _seeded(node)
            and not any(mark in node.meta for mark in _MARKS))


def _inputs(node: fx.Node) -> List[fx.Node]:
    return node.all_input_nodes


def metadata_read(node: fx.Node) -> bool:
    """``node`` is a receiver factory whose every operand but the receiver is a literal (the module docstring)."""
    return node.op == "call_method" and node.target in RECEIVER_FACTORIES and bool(node.args) and isinstance(
        node.args[0], fx.Node) and node.all_input_nodes == [node.args[0]] and hasattr(
        node.args[0].meta.get("tensor_meta"), "dtype")


def _cone_inputs(node: fx.Node) -> List[fx.Node]:
    """The inputs of ``node``'s computation: all of them, but the receiver of a metadata read."""
    return [] if metadata_read(node) else node.all_input_nodes


def stand_in(node: fx.Node, device: torch.device) -> torch.Tensor:
    """A zero-element tensor of the dtype the metadata read ``node``'s receiver was traced with, on ``device``."""
    return torch.empty(0, dtype=node.args[0].meta["tensor_meta"].dtype, device=device)  # type: ignore[union-attr]


def _random_state(device: torch.device) -> Tuple[torch.Tensor, ...]:
    cuda = (torch.cuda.get_rng_state(device),) if device.type == "cuda" else ()
    return (torch.random.get_rng_state(), *cuda)


def _drawing(gm: fx.GraphModule, free: Sequence[fx.Node]) -> Set[fx.Node]:
    """The nodes of ``free`` (graph order) whose run changes the random state: each run once on the program's state,
    no gradient, the random state restored (``rng_neutral``)."""
    values: Dict[fx.Node, Any] = {}
    drawing: Set[fx.Node] = set()
    if not free:
        return drawing
    interpreter, device = fx.Interpreter(gm), graph_device(gm)
    with rng_neutral(gm), torch.no_grad():
        for node in free:
            for arg in _cone_inputs(node):
                if arg.op == "get_attr" and arg not in values:
                    values[arg] = attribute_at(gm, str(arg.target))
            found = {**values, cast(fx.Node, node.args[0]): stand_in(node, device)} if metadata_read(node) else values
            args, kwargs = map_arg((node.args, node.kwargs), found.__getitem__)
            before = _random_state(device)
            values[node] = getattr(interpreter, node.op)(node.target, args, kwargs)
            if not all(torch.equal(a, b) for a, b in zip(before, _random_state(device))):
                drawing.add(node)
    return drawing


def batch_free_nodes(gm: fx.GraphModule) -> FrozenSet[fx.Node]:
    """Every batch-free call node of ``gm`` (the module docstring), in no particular order; the ``get_attr`` nodes, the
    state itself, are not among them."""
    modules, rebound = dict(gm.named_modules()), declared_code_change()
    opaque = _closure([n for n in gm.graph.nodes if n.op in _CALLS and node_impurity(  # a call outside the set, and
        gm, n, receiver_type(n), modules, rebound) is not None], lambda n: list(n.users))  # every value it reaches
    stateful = set()  # no placeholder upstream; a metadata read's receiver computed by declared operators
    for node in gm.graph.nodes:
        if node.op == "get_attr" or (node.op in _CALLS and all(arg in stateful for arg in _cone_inputs(node))
                                     and not (metadata_read(node) and node.args[0] in opaque)):
            stateful.add(node)
    touched = _touched(gm, modules, rebound)
    excluded = _closure(touched & stateful, _inputs)
    leaves: Set[fx.Node] = set()
    free: Dict[fx.Node, None] = {}
    for node in gm.graph.nodes:
        if node in excluded or node not in stateful:
            continue
        if node.op == "get_attr":
            leaves |= {node} if node_impurity(gm, node, None, modules, rebound) is None else set()
        elif all(arg in free or arg in leaves for arg in _cone_inputs(node)) \
                and _declared_deterministic(gm, node, modules, rebound):
            free[node] = None
    drawing = _drawing(gm, list(free))
    kept: Set[fx.Node] = set()
    for node in free:
        if node not in drawing and all(arg in kept or arg in leaves for arg in _cone_inputs(node)):
            kept.add(node)
    return frozenset(kept)


def _tensor_valued(node: fx.Node) -> bool:
    meta = node.meta.get("tensor_meta")
    return meta is not None and hasattr(meta, "shape") and hasattr(meta, "dtype")


class BatchFreeValues:
    """The tensor-valued batch-free nodes of a program the converter maps (``nodes``), and the module computing each
    at every call from one copy of the state they read (``value``)."""

    def __init__(self, gm: fx.GraphModule) -> None:
        self._gm = gm
        self.nodes: FrozenSet[fx.Node] = frozenset(node for node in batch_free_nodes(gm) if _tensor_valued(node))
        self._state = nn.Module()
        self._memo: Dict[int, Any] = {}
        self._programs: Dict[fx.Node, fx.GraphModule] = {}
        self._values: Dict[Tuple[fx.Node, Tuple[int, ...]], BatchFreeValue] = {}

    def evaluate(self, node: fx.Node) -> torch.Tensor:
        """``node``'s value now, from the copied state (no gradient, the random state untouched)."""
        with rng_neutral(self._gm), torch.no_grad():
            return self._program(node)()

    def value(self, node: fx.Node, shape: Sequence[int]) -> BatchFreeValue:
        """The bound operand computing ``node`` at every call, in the stored form ``shape``; one per form."""
        key = (node, tuple(int(d) for d in shape))
        if key not in self._values:
            self._values[key] = BatchFreeValue(self._program(node), key[1])
        return self._values[key]

    def _program(self, value: fx.Node) -> fx.GraphModule:
        """The traced operators computing ``value``, in traced order, over the copied state; no placeholder."""
        if value not in self._programs:
            cone, graph, env = _closure([value], _cone_inputs), fx.Graph(), {}
            for node in self._gm.graph.nodes:
                if node in cone:
                    if node.op in ("get_attr", "call_module"):
                        self._install(str(node.target), module=node.op == "call_module")
                    if metadata_read(node):  # its receiver: the stand-in of the receiver's dtype, in the state
                        env[node.args[0]] = graph.get_attr(self._stand_in(node))
                    env[node] = graph.node_copy(node, env.__getitem__)
            graph.output(env[value])
            self._programs[value] = fx.GraphModule(self._state, graph).eval()
        return self._programs[value]

    def _stand_in(self, node: fx.Node) -> str:
        """The stand-in of the dtype the metadata read ``node`` reads, in a child no attribute of the program names."""
        holder = "_receivers"
        while hasattr(self._gm, holder):
            holder += "_"
        child = cast(nn.Module, self._state._modules.setdefault(holder, nn.Module()))
        name = f"_receiver_{str(node.args[0].meta['tensor_meta'].dtype).split('.')[-1]}"  # type: ignore[union-attr]
        if name not in child._buffers:
            child.register_buffer(name, stand_in(node, graph_device(self._gm)))
        return f"{holder}.{name}"

    def _install(self, target: str, *, module: bool) -> None:
        """A copy of the program's ``target`` in the shared state, at its path (one memo, so a tie stays one tensor);
        a plain container a shorter path made gives way to a called module's copy."""
        *path, last = target.split(".")
        parent = self._state
        for part in path:
            if not isinstance(getattr(parent, part, None), nn.Module):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        existing, source = getattr(parent, last, None), attribute_at(self._gm, target)
        if existing is not None and not (module and type(existing) is nn.Module):
            if isinstance(source, (torch.Tensor, nn.Module)) and self._memo.get(id(source)) is not existing:
                raise RuntimeError(f"BatchFreeValues: the copied state holds at {target!r} a value not the copy of it")
            return
        copied = copy.deepcopy(source, self._memo)
        if isinstance(copied, nn.Parameter):
            parent.register_parameter(last, copied)
        elif isinstance(copied, torch.Tensor):
            parent.register_buffer(last, copied)
        elif isinstance(copied, nn.Module):
            parent.add_module(last, copied)
        else:
            setattr(parent, last, copied)
