"""Latency calculation for IRGraph."""

from __future__ import annotations

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, ComputeOp
from mimarsinan.mapping.latency.upstream import iter_upstream_neural_ids


class IRLatency:
    """Calculate and assign per-core latency for an IRGraph: core depth = 1 + max(upstream core depths), O(nodes) with memoization."""

    def __init__(self, ir_graph: IRGraph):
        self.ir_graph = ir_graph
        self._depth: dict[int, int] = {}
        self._node_map: dict[int, NeuralCore | ComputeOp] = {
            n.id: n for n in ir_graph.nodes
        }

    def _get_depth(self, node_id: int) -> int:
        """Recursive depth with memoization."""
        if node_id in self._depth:
            return self._depth[node_id]

        node = self._node_map.get(node_id)
        if node is None:
            self._depth[node_id] = 0
            return 0

        upstream_ids = list(
            iter_upstream_neural_ids(
                (s for s in node.input_sources.flat if isinstance(s, IRSource)),
                skip_off=True,
            )
        )

        if not upstream_ids:
            depth = 1 if isinstance(node, NeuralCore) else 0
        else:
            max_up = max(self._get_depth(uid) for uid in upstream_ids)
            depth = (1 + max_up) if isinstance(node, NeuralCore) else max_up

        self._depth[node_id] = depth
        return depth

    def calculate(self) -> int:
        """Calculate latencies for all neural cores; returns the network's max latency (depth)."""
        self._depth = {}

        max_delay = 0
        for node in self.ir_graph.nodes:
            d = self._get_depth(node.id)
            if isinstance(node, NeuralCore):
                node.latency = d - 1
            max_delay = max(max_delay, d)

        return max_delay

