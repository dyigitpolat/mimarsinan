"""Spike/value node classification and neural-segment partition of an exec graph."""

from __future__ import annotations

from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper


def perceptron_of(node):
    return getattr(node, "perceptron", None)


def is_encoding_perceptron(node) -> bool:
    p = perceptron_of(node)
    return p is not None and getattr(p, "is_encoding_layer", False)


def is_value_boundary(node) -> bool:
    """Host-side value producer (raw input + host ComputeOps).

    Matches deployment (HCM): a ComputeOp runs host-side once on decoded values,
    never per-cycle on spikes.
    """
    return isinstance(node, (InputMapper, ComputeOpMapper))


def classify_spike_producers(exec_order, deps_map) -> dict:
    """Map each node to whether it carries on-chip spikes (vs host-side values)."""
    produces: dict = {}
    for node in exec_order:  # exec_order is topological: deps precede node
        if perceptron_of(node) is not None:
            produces[node] = True
        elif is_value_boundary(node):
            produces[node] = False
        else:
            produces[node] = any(produces.get(d, False) for d in deps_map.get(node, []))
    return produces


def partition_spike_segments(exec_order, deps_map):
    """Group spike-producing nodes into segments (maximal connected regions).

    Returns ``({node: segment_root}, produces_spikes)``.
    """
    produces = classify_spike_producers(exec_order, deps_map)
    parent: dict = {}

    def find(n):
        root = n
        while parent[root] != root:
            root = parent[root]
        while parent[n] != root:
            parent[n], n = root, parent[n]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra is not rb:
            parent[ra] = rb

    spike_nodes = [n for n in exec_order if produces[n]]
    for n in spike_nodes:
        parent.setdefault(n, n)
    for n in spike_nodes:
        # An encoding layer consumes a decoded value, so it starts a fresh segment (never unioned with its source).
        if is_encoding_perceptron(n):
            continue
        for d in deps_map.get(n, []):
            if produces.get(d, False):
                parent.setdefault(d, d)
                union(n, d)
    return {n: find(n) for n in spike_nodes}, produces


def partition_perceptron_segments(exec_order, deps_map):
    """Perceptron-only view of :func:`partition_spike_segments` (for tests/introspection)."""
    seg_of, _ = partition_spike_segments(exec_order, deps_map)
    return {n: r for n, r in seg_of.items() if perceptron_of(n) is not None}


def spike_segment_count(model_repr) -> int:
    """Number of neural (spike) segments in a ``ModelRepresentation`` exec graph."""
    model_repr._ensure_exec_graph()
    seg_of, _ = partition_spike_segments(model_repr._exec_order, model_repr._deps)
    return len(set(seg_of.values()))
