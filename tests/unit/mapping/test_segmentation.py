"""Pins the single-source segmentation module: the ordered neural/host
partition used by the HCM packer agrees with the dependency-graph segment ids
used by the layout finalizer, and both stay stable across the config matrix."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from integration._placement_signature import CONFIGS  # noqa: E402

from mimarsinan.mapping.ir import ComputeOp, NeuralCore  # noqa: E402
from mimarsinan.mapping.layout.segmentation import (  # noqa: E402
    HostSegment,
    NeuralSegment,
    compute_segment_ids,
    partition_ir_graph,
)


def _ir_dep_view(ir_graph):
    """Build (node_input_node_ids, node_is_neural) from a concrete IRGraph."""
    node_input_node_ids: dict[int, set] = {}
    node_is_neural: dict[int, bool] = {}
    for node in ir_graph.nodes:
        deps = {
            int(s.node_id)
            for s in node.input_sources.flatten()
            if int(getattr(s, "node_id", -1)) >= 0
        }
        node_input_node_ids[node.id] = deps
        node_is_neural[node.id] = isinstance(node, NeuralCore)
    return node_input_node_ids, node_is_neural


def test_partition_preserves_node_order_and_kinds():
    for name, (builder, _kwargs) in CONFIGS.items():
        ir = builder()
        segments = partition_ir_graph(ir)

        # Flattening the partition reproduces the exact node order of the graph.
        flat = []
        for seg in segments:
            if isinstance(seg, NeuralSegment):
                flat.extend(n.id for n in seg.nodes)
            else:
                flat.append(seg.compute_op.id)
        assert flat == [n.id for n in ir.nodes], f"order drift for {name}"

        # Neural segments are non-empty and contain only NeuralCores.
        for seg in segments:
            if isinstance(seg, NeuralSegment):
                assert seg.nodes, f"empty neural segment for {name}"
                assert all(isinstance(n, NeuralCore) for n in seg.nodes)
            else:
                assert isinstance(seg.compute_op, ComputeOp)


def test_partition_agrees_with_dependency_segment_ids():
    """Every node inside one HCM neural segment shares one dependency-based
    segment_id, and segment order is monotonic -- so the two segmentation views
    cannot disagree on membership."""
    for name, (builder, _kwargs) in CONFIGS.items():
        ir = builder()
        seg_ids = compute_segment_ids(*_ir_dep_view(ir))

        neural_segments = [
            s for s in partition_ir_graph(ir) if isinstance(s, NeuralSegment)
        ]
        seen_ids = []
        for seg in neural_segments:
            ids = {seg_ids[n.id] for n in seg.nodes}
            assert len(ids) == 1, (
                f"{name}: neural segment spans multiple dependency segment_ids {ids}"
            )
            seen_ids.append(next(iter(ids)))

        assert seen_ids == sorted(seen_ids), (
            f"{name}: HCM segment order disagrees with dependency segment_ids: {seen_ids}"
        )
        assert len(set(seen_ids)) == len(seen_ids), (
            f"{name}: dependency segment_ids not unique per HCM segment: {seen_ids}"
        )
