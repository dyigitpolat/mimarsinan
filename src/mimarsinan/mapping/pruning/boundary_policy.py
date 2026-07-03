"""Model-level I/O boundary policy for IR pruning and torch mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Set, Tuple

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore

__all__ = [
    "PruningBoundaryPolicy",
    "assert_unified_ir_for_pruning",
    "build_boundary_ir_graph",
    "compute_model_io_boundary_policy",
    "build_computeop_producer_map",
    "build_computeop_referenced_neurons",
    "compute_perceptron_io_exemption_indices",
]


@dataclass(frozen=True)
class PruningBoundaryPolicy:
    """Per-node row/column indices that must never be pruned (model I/O only)."""

    exempt_rows_per_node: Mapping[int, frozenset[int]]
    exempt_cols_per_node: Mapping[int, frozenset[int]]


def compute_model_io_boundary_policy(ir_graph: IRGraph) -> PruningBoundaryPolicy:
    """Return hard-exempt rows/cols: model input data axons (-2) and output logits."""
    exempt_rows: Dict[int, Set[int]] = {}
    exempt_cols: Dict[int, Set[int]] = {}

    output_keys: Set[Tuple[int, int]] = set()
    if ir_graph.output_sources.size:
        for src in ir_graph.output_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                output_keys.add((src.node_id, src.index))

    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore) or not hasattr(node, "input_sources"):
            continue
        in_rows: Set[int] = set()
        for i, src in enumerate(node.input_sources.flatten()):
            if isinstance(src, IRSource) and src.is_input():
                in_rows.add(i)
        exempt_rows[node.id] = in_rows

        out_cols: Set[int] = set()
        for j in range(node.get_output_count()):
            if (node.id, j) in output_keys:
                out_cols.add(j)
        exempt_cols[node.id] = out_cols

    return PruningBoundaryPolicy(
        exempt_rows_per_node={nid: frozenset(s) for nid, s in exempt_rows.items()},
        exempt_cols_per_node={nid: frozenset(s) for nid, s in exempt_cols.items()},
    )


def _computeop_relays_deadness(op: ComputeOp) -> bool:
    """Whether upstream deadness may relay 1:1 through this op (output i == f(input i), f(0)=0).

    A general op's output index has no positional correspondence with its inputs
    (pool/linear/conv change shape), so relaying deadness through it drops LIVE
    signal from the deployed graph. Only declared-identity ops with matching
    flat input/output widths qualify.
    """
    if str(getattr(op, "op_type", "")).lower() != "identity":
        return False
    output_shape = getattr(op, "output_shape", None)
    if output_shape is None:
        return True
    n_out = 1
    for d in output_shape:
        n_out *= int(d)
    return n_out == len(op.input_sources.flatten())


def build_computeop_producer_map(
    ir_graph: IRGraph,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Map ``(compute_op_id, output_index)`` to upstream ``(neural_id, col)`` for
    the ops that qualify under :func:`_computeop_relays_deadness` (identity 1:1
    relays only); every other ComputeOp is a deadness barrier."""
    producer_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, ComputeOp):
            continue
        if not _computeop_relays_deadness(node):
            continue
        for out_idx, src in enumerate(node.input_sources.flatten()):
            if isinstance(src, IRSource) and src.node_id >= 0:
                producer_map[(node.id, out_idx)] = (src.node_id, src.index)
    return producer_map


def build_computeop_referenced_neurons(ir_graph: IRGraph) -> frozenset[Tuple[int, int]]:
    """Neurons referenced by any ComputeOp input (connectivity, not hard exempt)."""
    referenced: Set[Tuple[int, int]] = set()
    for node in ir_graph.nodes:
        if not isinstance(node, ComputeOp):
            continue
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                referenced.add((src.node_id, src.index))
    return frozenset(referenced)


def compute_perceptron_io_exemption_indices(
    ir_graph: IRGraph,
    perceptrons=None,
) -> Tuple[Set[int], Set[int]]:
    """Return ``(exempt_input_layers, exempt_output_layers)`` keyed by perceptron_index."""
    exempt_input: Set[int] = set()
    exempt_output: Set[int] = set()

    output_keys: Set[Tuple[int, int]] = set()
    if ir_graph.output_sources.size:
        for src in ir_graph.output_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                output_keys.add((src.node_id, src.index))

    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        pidx = getattr(node, "perceptron_index", None)
        if pidx is None:
            continue

        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.is_input():
                exempt_input.add(pidx)
                break

        for j in range(node.get_output_count()):
            if (node.id, j) in output_keys:
                exempt_output.add(pidx)
                break

    if perceptrons is not None:
        for idx, p in enumerate(perceptrons):
            if getattr(p, "is_encoding_layer", False):
                exempt_input.add(idx)

    return exempt_input, exempt_output


def assert_unified_ir_for_pruning(ir_graph: IRGraph) -> None:
    """Reject segment subgraphs that must not drive unified-graph pruning."""
    if getattr(ir_graph, "_is_segment_subgraph", False):
        raise ValueError(
            "prune_ir_graph must run on the unified pre-segmentation IRGraph, "
            "not on a hybrid segment subgraph."
        )


def build_boundary_ir_graph(model, pipeline) -> IRGraph:
    """Build unified IR for boundary/exemption queries (Pruning Step)."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.transformations.quantization_bounds import quantization_bounds

    bits = int(pipeline.config.get("weight_bits", 8))
    _, q_max = quantization_bounds(bits)
    mapper_repr = model.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    ir_mapping = IRMapping(
        q_max=q_max,
        firing_mode=str(pipeline.config.get("firing_mode", "Default")),
    )
    return ir_mapping.map(mapper_repr)

