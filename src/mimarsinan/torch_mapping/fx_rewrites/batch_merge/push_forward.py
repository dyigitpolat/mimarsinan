"""The batch-merge push-forward (coverage W7): a reshape that merges the batch into another axis is rewritten so that
the merged value is carried split, batch-leading, until a reshape restores a batch-leading value.

A MERGE is a ``view``, ``reshape`` or ``flatten`` of a batch-leading value (the batch domain proves its extents
``(B, k_1, ...)``, every ``k_i`` an integer) whose proved output extents are ``(a*B, m_1, ...)`` with ``a >= 2``:
torchvision's window partition ``reshape(B * num_windows, N, C)``. The converter refuses it by name, since its axis 0 is
not the batch axis (``_require_batch_leading``). Its split value is ``reshape(-1, a, m_1, ...)`` of the same input:
the same elements in the same order, the batch on axis 0. From each merge, the REGION is every value computed from a
merged value by an operation ``axes.plan_of`` takes, each carried split with its merged axis as two (``axes.py``);
it ends at a reshape that restores a batch-leading value from a value merged on axis 0 (an END, the window reverse
``view(B, ...)``), and is read otherwise only as an extent (``size``) or for its dtype and device. Regions joined by an
operation reading two merged values are one. A region is rewritten when every reader of every merged value in it is
taken, no node of it is ordered by a write in place (``rewrite_support.rewritable``), and the domain's proved extents of
every merged value agree with the rules (axis ``at`` of extent ``a*B``, every other axis an integer); else nothing of
it is rewritten and the merge stays refused by name. Only a pure program has proved extents (``symbolic/purity.py``),
so an impure one is never rewritten.

Lemma. Under T1 (torch as shipped) and the batch domain's lemma, the rewritten program computes, at every batch B >= 1,
the value the traced program computes at every node outside a rewritten region: by induction over a region in traced
order, each split value is its merged value with the merged axis as two (a merge: a reshape keeps the row-major order
of the elements, so ``(a*B, m...)`` and ``(B, a, m...)`` of one input are the same elements; a taken operation: its
rule in ``axes.py``), an extent is the merged value's own (``a`` times the split extent at the merged axis), a dtype or
device the merged value's, and an end reshapes the same elements in the same order to the same extents.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Set, cast

import torch
import torch.fx as fx
from torch.fx.node import map_arg

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH, Affine
from mimarsinan.torch_mapping.fx_rewrites.batch_merge.axes import plan_of
from mimarsinan.torch_mapping.fx_rewrites.batch_merge.layouts import (
    LAYOUTS, Merged, Plan, reshape_target, shape_operands, split_axis,
)
from mimarsinan.torch_mapping.fx_rewrites.rewrite_support import trace_meta

__all__ = ["push_batch_merge_forward"]

_RESHAPES = (torch.reshape, torch.flatten)


def _reshape(node: fx.Node) -> bool:
    return node.op == "call_method" and node.target in LAYOUTS or node.op == "call_function" and \
        node.target in _RESHAPES


def _maybe_merge(node: fx.Node) -> bool:
    """A reshape whose traced output (batch 1) has another leading extent than its input: a merge candidate."""
    source = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
    if source is None or not _reshape(node) or not shape_operands(node):
        return False
    before, after = (getattr(n.meta.get("tensor_meta"), "shape", None) for n in (source, node))
    return bool(before) and bool(after) and cast(Any, after)[0] != cast(Any, before)[0]


def _merge_plan(node: fx.Node, layouts: BatchLayouts) -> Optional[Plan]:
    """``node``'s split value when it merges a batch-leading value (the module docstring)."""
    source = cast(fx.Node, node.args[0])
    before = layouts.extents(source)
    if before is None or not before or before[0] != BATCH or any(type(d) is not int for d in before[1:]):
        return None
    found = reshape_target(Merged(0, 1, len(before)), layouts.extents(node), ())
    if found is None or found[0] is None:
        return None
    method = "view" if node.op == "call_method" and node.target == "view" else "reshape"
    return Plan("call_method", method, (source, *found[2]), {}, found[0])


def _agrees(node: fx.Node, plan: Plan, layouts: BatchLayouts) -> bool:
    """The domain's proved extents of ``node`` are its plan's: the merged axis ``a*B``, every other an integer."""
    out = plan.out
    if out is None or out.elements:
        return True
    dims = layouts.extents(node)
    return dims is not None and len(dims) == out.rank and dims[out.at] == Affine(out.a, 0) and all(
        type(d) is int for i, d in enumerate(dims) if i != out.at)


class _Regions:
    """Union-find over the merges; a rejected merge rejects its whole region."""

    def __init__(self) -> None:
        self.parent: Dict[fx.Node, fx.Node] = {}
        self.rejected: Set[fx.Node] = set()

    def root(self, node: fx.Node) -> fx.Node:
        while self.parent[node] is not node:
            node = self.parent[node]
        return node

    def join(self, nodes: List[fx.Node]) -> fx.Node:
        roots = [self.root(node) for node in nodes]
        for other in roots[1:]:
            self.parent[other] = roots[0]
            if other in self.rejected:
                self.rejected.add(roots[0])
        return roots[0]


def _emit(gm: fx.GraphModule, node: fx.Node, plan: Plan, carried: Dict[fx.Node, fx.Node],
          infos: Dict[fx.Node, Merged], modules) -> fx.Node:
    """The split value of ``node`` (or, for an extent, its value), added before it."""
    mapped = lambda value: map_arg(value, lambda n: carried.get(n, n))  # noqa: E731
    with gm.graph.inserting_before(node):
        if plan.op == "size":
            read = cast(fx.Node, node.args[0])
            source, info = carried[read], infos[read]
            if plan.target == info.at:  # the merged extent: a times the split value's batch extent
                return gm.graph.call_function(operator.mul, (gm.graph.call_method("size", (source, info.at)), info.a))
            return gm.graph.call_method("size", (source, split_axis(plan.target, info.at)))
        new = gm.graph.create_node(plan.op, plan.target, mapped(plan.args), mapped(plan.kwargs))
    trace_meta(new, modules.get(str(new.target)) if new.op == "call_module" else None)
    return new


def _rewrite(gm: fx.GraphModule, region: List[fx.Node], plans: Dict[fx.Node, Plan], infos: Dict[fx.Node, Merged],
             modules) -> bool:
    """Every node of one region on its split values; the graph unchanged and False when a split value fails."""
    carried: Dict[fx.Node, fx.Node] = {}
    made: List[fx.Node] = []

    def step(node: fx.Node, plan: Plan) -> None:
        if plan.out is not None or plan.op == "size":
            carried[node] = _emit(gm, node, plan, carried, infos, modules)
        else:  # an end or a read: the same call on the split value must run on its metadata
            _probe(node, carried, modules)

    for node in region:
        before = node.prev
        _, failure = guarded(lambda: step(node, plans[node]))  # noqa: B023 - called at once, in this iteration
        made += _between(before, node)
        if failure is not None:  # a split value torch's kernels do not take: the region stays as traced
            for new in reversed(made):
                gm.graph.erase_node(new)
            return False
    for node in region:
        plan = plans[node]
        if plan.op == "size" or plan.out is not None:
            node.replace_all_uses_with(carried[node])
        else:  # an end, or a read of a dtype or device: the same call on the split value
            node.args = map_arg(node.args, lambda n: carried.get(n, n))
            node.kwargs = map_arg(node.kwargs, lambda n: carried.get(n, n))
    for node in reversed(region):
        if (plans[node].op == "size" or plans[node].out is not None) and not node.users:
            gm.graph.erase_node(node)
    return True


def _concrete(value: Any) -> Any:
    """``value`` at the trace batch for a run on metadata: a tensor its meta tensor, an extent (``size``) and Python
    arithmetic on extents their integers; ``ValueError`` for anything else."""
    if not isinstance(value, fx.Node):
        return value
    meta = value.meta.get("tensor_meta")
    if meta is not None:
        return torch.empty_strided(tuple(meta.shape), tuple(meta.stride), dtype=meta.dtype, device="meta")
    args = [_concrete(arg) for arg in value.args]
    if value.op == "call_method" and value.target == "size" and len(args) == 2:
        return int(args[0].shape[args[1]])
    if value.op == "call_function" and value.target in (operator.mul, operator.floordiv, operator.add, operator.sub):
        return cast(Any, value.target)(*args)
    raise ValueError(f"FX node {value.name!r} has no value on metadata")


def _probe(node: fx.Node, carried: Dict[fx.Node, fx.Node], modules) -> None:
    """``node``'s call on its operands' split values, on metadata: raises where torch's kernel does not take them
    (a ``view`` the split value's strides do not allow)."""
    args, kwargs = map_arg((node.args, node.kwargs), lambda n: _concrete(carried.get(n, n)))
    if node.op == "call_method":
        getattr(args[0], str(node.target))(*args[1:], **kwargs)
    elif node.op == "call_function":
        cast(Any, node.target)(*args, **kwargs)  # a call_function node's callable


def _between(first: fx.Node, last: fx.Node) -> List[fx.Node]:
    """The nodes after ``first`` and before ``last``."""
    found, node = [], first.next
    while node is not last:
        found.append(node)
        node = node.next
    return found


def push_batch_merge_forward(gm: fx.GraphModule, ordered: Set[str]) -> bool:
    """Rewrite every region of ``gm`` (the module docstring); True when one was rewritten. The batch domain's proof
    (``BatchLayouts``) is computed only when a reshape of a value computed from the input is a merge candidate."""
    reached: Set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder" or any(arg in reached for arg in node.all_input_nodes):
            reached.add(node)
    if not any(_maybe_merge(node) and node.args[0] in reached for node in gm.graph.nodes):
        return False
    layouts = BatchLayouts(gm)
    if not layouts.purity.pure:
        return False
    modules = dict(gm.named_modules())
    infos: Dict[fx.Node, Merged] = {}
    plans: Dict[fx.Node, Plan] = {}
    regions = _Regions()
    for node in gm.graph.nodes:
        merged = [arg for arg in node.all_input_nodes if arg in infos]
        plan = _merge_plan(node, layouts) if not merged and _maybe_merge(node) else None
        if plan is None and not merged:
            continue
        root = regions.join(merged) if merged else node
        regions.parent.setdefault(node, root)
        if plan is None:
            plan = plan_of(node, infos, layouts.extents, lambda n: n not in reached, modules)
        if plan is None or node.name in ordered or not _agrees(node, plan, layouts):
            regions.rejected.add(regions.root(root))
            continue
        plans[node] = plan
        if plan.out is not None:
            infos[node] = plan.out
    accepted: Dict[fx.Node, List[fx.Node]] = {}
    for node in plans:
        root = regions.root(node)
        if root not in regions.rejected:
            accepted.setdefault(root, []).append(node)
    changed = False
    for region in accepted.values():
        changed = _rewrite(gm, region, plans, infos, modules) or changed
    if changed:
        gm.graph.lint()
        gm.recompile()
    return changed
