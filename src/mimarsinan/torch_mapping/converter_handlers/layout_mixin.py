"""Per-sample layouts: the converter's rules for a layout method of an activation (review rounds 3 to 9, 14).

A layout method (``view``, ``reshape``, ``flatten``, ``permute``, ``transpose``, ``unsqueeze``, ``squeeze``,
``expand``, ``torch.flatten``, ``nn.Flatten``) of a PURE program (``symbolic/purity.py``) is lowered with the
per-sample shape fixed at conversion: a ``ReshapeMapper``, a ``PermuteMapper``, or its source carried as an alias. Two
rules keep that exact at every batch:

- its input and output carry the batch on axis 0 alone (``BatchLayouts.violation``), else ``BatchAxisLayoutError``;
- a per-sample shape an opaque batch value reaches (``BatchLayouts.shape_decision``: a value outside the affine batch
  domain, review round 14) holds at the traced batch only, so such a layout fixes no shape: it is the host op of its
  native method on its run-time operands (``_native_layout``), as every node that depends on an opaque value in a
  shape position runs at run time. A value whose per-sample shape such a decision reaches may then be read only by a
  host op (it runs the native function on the run-time value) or by a pure reshape that restores a per-sample shape the
  decision does not reach (``torch.flatten(z.view(B, 6 // g, g), 1)``); a static per-sample layout it reaches (a
  perceptron, a convolution, a static layout or concatenation mapper) is refused by name, naming the decision
  (``_require_static_reads``).

Every layout method of an IMPURE program whose value depends on the input (an activation's, a parameter's ``expand``
to the batch) fixes no shape either: it is the host op of its native method with its run-time operands, so its value
is the native one at every batch, whatever user code or user type the program carries
(``regroup(y).reshape(B, h.size(2), h.size(1))``, whose per-sample shape changes from batch 9 on).
Every other mapper the converter emits applies to its operands' run-time values as natively (a host op, a
``PermuteMapper``, a ``ConcatMapper``, a perceptron or a convolution), and stage 2 refuses the program's value program
by name (``ImpureShapeProgramError``; ``OpaqueBatchValueError`` for a pure program with an opaque value).

A host op reads its activation operands' values as natively (``_native_operands``, review round 14): an operand
carried as a layout alias of another shape (an ``unsqueeze``, ``squeeze`` or ``expand`` of an activation, carried as
its source's value) is first lowered as the host op of its native method on its run-time operands. [round 13 refused
such a host op by name where an opaque value reached it, and elsewhere handed it the alias: an LSTM read
``y.unsqueeze(1)`` as ``y``, a sequence of length B.]
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError, decided_reason, node_operator
from mimarsinan.torch_mapping.converter_handlers.shape_value_mixin import depends_on_input
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import (
    COPY_GUARD,
    KEPT_OPERANDS,
    RUN_AFTER,
    OperandWriteError,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.shape_decisions import reshape_kind
from mimarsinan.torch_mapping.fx_shape_utils import node_output_shape

_RESHAPES = frozenset({"view", "reshape", "flatten"})


def _count(shape: Any) -> Optional[int]:
    """The per-sample element count of a traced shape (``None`` without one)."""
    return None if shape is None else math.prod(tuple(shape)[1:])


class LayoutMixin(ConverterContract):
    """Batch-axis and shape-decision rules of a layout method; the native layouts of an impure program."""

    def _layout_violation(self, node: fx.Node) -> Optional[str]:
        """Why an activation does not carry the batch on axis 0 alone (observed or proved); else None."""
        return self._batch_layouts.violation(node) if depends_on_input(node) else None

    def _require_batch_leading(self, node: fx.Node) -> None:
        """A layout method of an activation maps one sample to one sample: its input and output must carry the
        batch on axis 0 alone. A parameter's layout (a class token's ``expand``) is not a per-sample mapping."""
        tensor = node.args[0] if node.args else None
        if not isinstance(tensor, fx.Node) or not depends_on_input(tensor):
            return
        for role, value in (("input", tensor), ("output", node)):
            violation = self._batch_layouts.violation(value)
            if violation is not None:
                raise BatchAxisLayoutError(node, f"in its {role}, {violation}")

    def _native_layout(self, node: fx.Node, op: Any) -> bool:
        """Lower a layout as the host op ``op`` (its native method) on its run-time operands: in an impure program a
        layout whose value depends on the input (an activation's, or a parameter's ``expand`` to the batch), in a pure
        one a layout an opaque batch value reaches (``_require_restored``); False for any other layout."""
        native = self._batch_layouts.impurity is not None or node in self._native_decided()
        if not native or not depends_on_input(node):
            return False
        self._emit_generic_compute_op(node, op)
        return True

    def _native_decided(self) -> set:
        found = getattr(self, "_decided_layouts", None)
        if found is None:
            found = set()
            setattr(self, "_decided_layouts", found)
        return found

    def _require_restored(self, node: fx.Node) -> None:
        """A layout whose per-sample shape an opaque batch value reaches fixes no shape: it is lowered as the host op
        of its native method (``_native_layout``), and its readers are checked by ``_require_static_reads``."""
        tensor = node.args[0] if node.args else None
        if self._batch_layouts.shape_decision(node) is not None and isinstance(tensor, fx.Node) \
                and depends_on_input(tensor):
            self._native_decided().add(node)

    def _require_static_reads(self) -> None:
        """No mapper that fixes a per-sample shape at conversion reads a value whose per-sample shape an opaque batch
        value reaches: every reader of such a value is a host op on its run-time value, the output, an alias of it
        (itself so read), or a pure reshape restoring a per-sample shape the decision does not reach; else the
        conversion is refused by name, naming the decision."""
        for node in self.gm.graph.nodes:
            decision = self._batch_layouts.shape_decision(node)
            mapper = self._node_to_mapper.get(node)
            if decision is None or mapper is None or not depends_on_input(node):
                continue
            for user in node.users:
                used = self._node_to_mapper.get(user)
                restored = (self._batch_layouts.shape_decision(user) is None and user.args[:1] == (node,)
                            and reshape_kind(user, self._modules) is not None)
                native = isinstance(used, ComputeOpMapper) and any(
                    source is mapper for source in getattr(used, "_sources_list", ()))
                folded = used is None and user in self._shape_values  # a shape value computed at conversion
                if user.op == "output" or restored or native or used is mapper or folded:
                    continue
                reason = decided_reason(node, decision)
                if self._batch_layouts.batch_axis_decision(node) is not None:
                    reason += ", its axis 0 (the batch axis) included"
                raise BatchAxisLayoutError(node, f"{reason}; it reaches {node_operator(user, self._modules)} (FX node "
                                                 f"{user.name!r}), which fixes its per-sample shape at conversion, "
                                                 "before a pure reshape restores a per-sample shape the decision does "
                                                 "not reach")

    def _native_operands(self, node: fx.Node, operands: Optional[list] = None) -> None:
        """Before a host op (or the output) reads them, every activation operand of ``node`` (``operands``, default
        every input node) carried as a layout alias of another shape is lowered as the host op of its native method on
        its run-time operands, so the reader gets the native value (the alias's earlier readers keep the alias)."""
        for arg in node.all_input_nodes if operands is None else operands:
            if arg.op == "call_method" and self._is_layout_alias(arg):
                self._emit_generic_compute_op(arg, getattr(torch.Tensor, str(arg.target)))

    def _native_static_operand(self, node: fx.Node) -> None:
        """Before a mapper that fixes a per-sample shape at conversion reads its activation operand (a layout method, a
        perceptron, a convolution, a ``Flatten``; review round 15): an operand carried as a layout alias of another
        shape is lowered as for a host op (``_native_operands``), unless the reader is a reshape of the alias's element
        count (``view``, ``reshape``, ``flatten``), which reads the element sequence alone, the alias's own."""
        operand = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
        if operand is None or operand.op != "call_method" or not self._is_layout_alias(operand):
            return
        if self._reads_elements(node) and _count(node_output_shape(operand)) == _count(
                node_output_shape(self._alias_root(operand))):
            return
        self._native_operands(node, [operand])

    def _reads_elements(self, node: fx.Node) -> bool:
        if node.op == "call_module":
            return type(self._modules.get(str(node.target))) is torch.nn.Flatten
        return node.target in _RESHAPES if node.op == "call_method" else node.target in (torch.flatten, torch.reshape)

    def _alias_root(self, arg: fx.Node) -> fx.Node:
        """The value a chain of layout aliases carries (``y`` of ``y.unsqueeze(1).expand(-1, 3, -1)``, and of
        ``y.unsqueeze(1).contiguous()``, review round 16): the node whose value the shared mapper computes."""
        mapper = self._node_to_mapper.get(arg)
        while mapper is not None and arg.args and isinstance(arg.args[0], fx.Node) \
                and self._get_mapper(arg.args[0]) is mapper:
            arg = arg.args[0]
        return arg

    def _is_layout_alias(self, arg: fx.Node) -> bool:
        """``arg`` is carried as the value of another node of another shape: it shares its source's mapper, and its
        shape differs from its source's or from that of the value the mapper computes (review round 16: a
        ``contiguous()`` of an alias, which reuses its source's mapper, has its source's shape and not the value's)."""
        source = arg.args[0] if arg.args else None
        mapper = self._node_to_mapper.get(arg)
        if not isinstance(source, fx.Node) or mapper is None or mapper is not self._get_mapper(source):
            return False
        shape = node_output_shape(arg)
        return shape != node_output_shape(source) or shape != node_output_shape(self._alias_root(arg))

    def _guarded_copy(self, node: fx.Node) -> bool:
        """A call the rule of a write takes as a copy (``operand_writes.COPY_GUARD``, review round 16: a ``reshape``
        or ``contiguous()`` torch's schema lets share its operand's storage, which the traced run showed copying) is
        the host op of its native method on its run-time operand, checking at run time that it copies again
        (``ComputeOpMapper.copy_guard``); False for any other node."""
        guard = node.meta.get(COPY_GUARD)
        if guard is None or node.op not in ("call_method", "call_function"):
            return False
        self._emit_generic_compute_op(node, getattr(torch.Tensor, str(node.target)) if node.op == "call_method"
                                      else node.target)
        mapper = self._node_to_mapper.get(node)
        operand = next((arg for arg in node.all_input_nodes if arg.name == guard[0]), None)
        source = self._get_mapper(operand) if operand is not None else None
        sources = list(getattr(mapper, "_sources_list", ()))
        index = next((i for i, found in enumerate(sources) if found is source and source is not None), None)
        if not isinstance(mapper, ComputeOpMapper) or index is None:
            raise OperandWriteError(node_operator(node, self._modules), node.name, (
                "a call the rule of a write takes as a copy, whose copy the converted model cannot check at run time"))
        mapper.copy_guard = (index, tuple(guard[1]), node_operator(node, self._modules))
        return True

    def _mark_kept_inputs(self, node: fx.Node) -> None:
        """A call outside the declared operator set whose operand another node reads after it (the tracer's
        ``KEPT_OPERANDS``, ``operand_writes``) must not write that operand in place at run time: its host op checks it
        (``ComputeOpMapper.kept_inputs``, review round 15)."""
        mapper = self._node_to_mapper.get(node)
        kept = node.meta.get(KEPT_OPERANDS, ())
        if not isinstance(mapper, ComputeOpMapper) or not kept:
            return
        sources = [self._get_mapper(arg) for arg in node.all_input_nodes if arg.name in kept]
        mapper.kept_inputs = tuple(i for i, found in enumerate(mapper._sources_list)
                                   if any(found is source for source in sources))
        mapper.kept_operator = node_operator(node, self._modules)

    def _order_runs(self, output_mapper: Any) -> None:
        """Every node's host op runs after those of the nodes the tracer orders it after (``operand_writes.RUN_AFTER``:
        the earlier readers of a written operand's storage; review round 16, targeted pass): its own mapper's
        ``run_after``, or, for a node carried as another's value, its readers'. A cycle is refused by name."""
        names = {node.name: node for node in self.gm.graph.nodes}
        ordered = [node for node in self.gm.graph.nodes if node.meta.get(RUN_AFTER)]
        for node in ordered:
            before = [self._node_to_mapper.get(names[name]) for name in node.meta[RUN_AFTER] if name in names]
            self._run_after(node, [mapper for mapper in before if mapper is not None], set())
        cycle = _cycle(output_mapper) if ordered else None
        if cycle is not None:
            node = next((n for n, m in self._node_to_mapper.items() if m is cycle), None)
            raise OperandWriteError(node_operator(node, self._modules) if node is not None else type(cycle).__name__,
                                    node.name if node is not None else str(getattr(cycle, "name", None)),
                                    "the order of its reads and writes in place makes a cycle of the converted model")

    def _run_after(self, node: fx.Node, mappers: list, seen: set) -> None:
        mapper = self._node_to_mapper.get(node)
        if not mappers:
            return
        if mapper is not None and not any(mapper is self._get_mapper(arg) for arg in node.all_input_nodes):
            found = [m for m in mappers if m is not mapper and not any(m is r for r in mapper.run_after)]
            mapper.run_after = (*mapper.run_after, *found)
            return
        for user in node.users:
            if user not in seen:
                seen.add(user)
                self._run_after(user, mappers, seen)

    def _mark_host_value_layout(self, node: fx.Node) -> None:
        """Mark ``node``'s own host op whose value has no value-program form: it does not carry the batch on axis 0
        alone, or it is not a tensor; the mark names the operator for the refusal."""
        mapper = self._node_to_mapper.get(node)
        if not isinstance(mapper, ComputeOpMapper) or any(mapper is self._get_mapper(a) for a in node.all_input_nodes):
            return
        violation = self._layout_violation(node)
        if violation is not None:
            mapper.batch_layout_violation = violation
        if violation is not None or node_output_shape(node) is None:
            mapper.host_operator = node_operator(node, self._modules)


def _cycle(output_mapper: Any) -> Any:
    """A mapper on a cycle of the mapper graph's sources and ``run_after`` edges, or ``None``."""
    state: dict = {}
    stack = [(output_mapper, False)]
    while stack:
        mapper, done = stack.pop()
        if done:
            state[id(mapper)] = 2
            continue
        if state.get(id(mapper)) == 2:
            continue
        state[id(mapper)] = 1
        stack.append((mapper, True))
        for dep in [*mapper.get_source_mappers(), *getattr(mapper, "run_after", ())]:
            if dep is None:
                continue
            if state.get(id(dep)) == 1:
                return dep
            if state.get(id(dep)) is None:
                stack.append((dep, False))
    return None
