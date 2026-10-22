"""The operands of a leaf call to inline (``leaf_call_inlining``): why one keeps the call opaque (a value outside the
declared set upstream, ``computed_from``; samples that are not one tensor's, ``tensor_reason``; a literal outside the
declared types, ``operand_reason``), and the extents of an inlined call's operands, proved in the converter's own
affine batch domain (``unproved_operands``; fix round 2 of the coverage fixes, review round 2).

A capture's batch symbols stand for the affine forms the samples at batches 1, 2 and 3 give each operand's extents
(``body_capture``). A sample fit is an operand's form only where its extent is affine in the batch: ``h[:8]`` has
``min(B, 8)`` rows, which the samples fit as ``B``, and ``W[: B // 7 + 3]`` has 3 rows at every sampled batch. A body
that reads such an extent only in a decision left the operand dead after the splice, so the converter's batch domain
never saw it, and the call was certified while wrong from batch 9 (and from batch 7).

So every tensor operand of an inlined call computed from a graph input is proved before the call is inlined
(``unproved_operands``): on the trial program (the traced program with every inlined body spliced in), restricted to
the operands' cone, the converter's symbolic batch pass (``symbolic_batch.symbolic_shapes``, on the cone's seeded
traces at batches 1, 2 and 3) holds each operand's shape, proved (no decision and no data-dependent size upstream), as
integers and affine forms ``a*B + c`` for every batch B >= 1, equal to the samples at the trace batches; else the call
stays opaque, the extent named. An operand computed from no graph input (parameters, literals) is batch-free.

Soundness, by induction over the inlined calls in traced order. The cone of a call is pure: its operands are
computed by declared operators and by bodies inlined before it (``leaf_call_inlining``), and the purity is checked.
When the bodies in it compute the native values at every batch, the pass's lemma gives each operand's extents for
every batch B >= 1; an affine extent equal to the samples at three batches is their affine fit, the form the
capture's guards and comparisons were judged on (``body_capture``), so the call's body computes the native value at
every batch too.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.fx as fx
from torch.fx.node import map_arg
from torch.fx.passes.shape_prop import ShapeProp

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import Affine, Opaque, describe, meaning
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import DECLARED_TENSOR_TYPES, foreign_value
from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import SymbolicPass
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import node_impurity, program_purity
from mimarsinan.torch_mapping.converter_handlers.symbolic.symbolic_batch import rng_neutral, symbolic_shapes
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import TRACE_BATCHES, trace
from mimarsinan.torch_mapping.fx_rewrites.body_output import first_line
from mimarsinan.torch_mapping.fx_rewrites.operand_samples import OperandSamples, TensorSample

__all__ = ["computed_from", "impure_nodes", "operand_reason", "tensor_reason", "unproved_operands"]


def operand_reason(name: str, value: Any) -> Optional[str]:
    """Why a non-node operand is not a literal the body can be captured with."""
    held: list = []
    map_arg(value, held.append)
    if held:
        return f"its operand {name!r} holds a traced value inside a container"
    found = foreign_value(value)
    return None if found is None else f"its operand {name!r} is {found}"


def tensor_reason(name: str, rows: list) -> Optional[str]:
    """Why a traced operand's samples are not those of one tensor at every trace batch."""
    if not all(isinstance(row, TensorSample) for row in rows):
        kind = next(row for row in rows if not isinstance(row, TensorSample))
        return f"its operand {name!r} is not a strided tensor (a value of type {getattr(kind, '__name__', kind)})"
    if any(row.kind not in DECLARED_TENSOR_TYPES for row in rows):
        return f"its operand {name!r} is a tensor of type {rows[0].kind.__name__}"
    if len({(row.dtype, row.device, row.requires_grad, len(row.shape)) for row in rows}) > 1:
        return f"its operand {name!r} changes its dtype, device or rank between batches"
    return None


def impure_nodes(gm: fx.GraphModule, samples: OperandSamples, calls: Set[fx.Node]) -> Dict[fx.Node, str]:
    """Every node outside the declared set (``purity.node_impurity``, receivers typed by the batch-1 run) other than
    the calls to decide, with its reason."""
    modules = dict(gm.named_modules())
    found: Dict[fx.Node, str] = {}
    for node in gm.graph.nodes:
        if node in calls or node.op in ("placeholder", "output"):
            continue
        receiver = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
        if node.op != "get_attr" and node not in samples.types:
            found[node] = "a value the seeded run at batch 1 does not reach"
            continue
        reason = node_impurity(gm, node, None if receiver is None else samples.types.get(receiver), modules, None)
        if reason is not None:
            found[node] = reason
    return found


def computed_from(node: fx.Node, impure: Dict[fx.Node, str]) -> Optional[str]:
    """The first value outside the declared set ``node`` is computed from, named."""
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in impure:
            return f"{impure[current]} (node {current.name!r})"
        stack.extend(current.all_input_nodes)
    return None



def _cone(graph: fx.Graph, roots: Set[fx.Node]) -> Tuple[Set[fx.Node], Set[fx.Node]]:
    """The nodes ``roots`` are computed from (``roots`` included), and the nodes of ``graph`` computed from a graph
    input."""
    found, stack = set(), list(roots)
    while stack:
        node = stack.pop()
        if node not in found:
            found.add(node)
            stack.extend(node.all_input_nodes)
    batched: Set[fx.Node] = set()
    for node in graph.nodes:
        if node.op == "placeholder" or batched.intersection(node.all_input_nodes):
            batched.add(node)
    return found, batched


def _cone_module(trial: fx.GraphModule, cone: Set[fx.Node], roots: Sequence[fx.Node]):
    """The program computing ``roots`` alone (their cone, in traced order), and the map to its nodes."""
    graph, env = fx.Graph(), {}
    for node in trial.graph.nodes:
        if node in cone:
            env[node] = graph.node_copy(node, env.__getitem__)
    graph.output(tuple(env[root] for root in roots))
    return fx.GraphModule(trial, graph), env


def _proof(gm: fx.GraphModule, input_shape: Tuple[int, ...], dtype: torch.dtype, device: Any) -> SymbolicPass:
    """The symbolic batch pass of ``gm`` (the tracer's shape propagation at batch 1 first, its seeded traces); a
    ``ValueError`` naming the first impure node of an impure ``gm``."""
    example = torch.randn((1, *input_shape), generator=torch.Generator().manual_seed(1), dtype=dtype).to(device)
    with rng_neutral(gm), torch.no_grad():
        ShapeProp(gm).propagate(example)
    impurity = program_purity(gm).impurity
    if impurity is not None:
        raise ValueError(impurity.statement())
    return symbolic_shapes(gm, tuple(trace(gm, batch) for batch in TRACE_BATCHES))


def _shown(dims: Tuple[Any, ...]) -> str:
    return "(" + ", ".join("?" if isinstance(d, Opaque) else describe(d) for d in dims) + ")"


def _reason(proof: SymbolicPass, node: fx.Node, rows: Sequence[TensorSample]) -> Optional[str]:
    """Why ``node``'s extents are not the domain's affine forms equal to its samples ``rows``; ``None`` if they are."""
    symbolic = proof.shapes.get(node)
    if symbolic is None or not proof.proved(node):
        decision = proof.batch_decision(node)
        why = "no shape of it is computed" if decision is None else (
            f"{decision.operator}, FX node {decision.node.name!r}: {decision.detail}")
        return f"has extents the converter's batch domain does not prove for every batch B >= 1 ({why})"
    dims = symbolic.dims
    if not all(type(d) is int or isinstance(d, Affine) for d in dims):
        return f"has extents {_shown(dims)}, not affine in the batch in the converter's batch domain"
    for batch, row in zip(TRACE_BATCHES, rows):
        if tuple(meaning(d, batch) for d in dims) != row.shape:
            return f"has extents {_shown(dims)} in the converter's batch domain, {row.shape} at batch {batch}"
    return None


def unproved_operands(trial: fx.GraphModule, operands: Mapping[Any, Mapping[str, fx.Node]],
                      samples: Mapping[Any, Mapping[str, Sequence[TensorSample]]], input_shape: Tuple[int, ...],
                      dtype: torch.dtype, device: Any) -> Dict[Any, str]:
    """Per inlined call (a key of ``operands``: its tensor operands' nodes in ``trial``, their samples in
    ``samples``), why an operand's extents are not proved; empty when every one is."""
    roots = list(dict.fromkeys(node for nodes in operands.values() for node in nodes.values()))
    cone, batched = _cone(trial.graph, set(roots))
    roots = [root for root in roots if root in batched]
    if not roots:
        return {}
    gm, env = _cone_module(trial, cone, roots)
    proof, exc = guarded(lambda: _proof(gm, input_shape, dtype, device))
    found: Dict[Any, str] = {}
    for key, nodes in operands.items():
        for name, node in nodes.items():
            if node not in batched:
                continue
            reason = (f"has extents the converter's batch domain could not compute "
                      f"({first_line(exc) if exc is not None else 'no proof'})" if proof is None
                      else _reason(proof, env[node], samples[key][name]))
            if reason is not None:
                found[key] = f"its operand {name!r} {reason}"
                break
    return found
