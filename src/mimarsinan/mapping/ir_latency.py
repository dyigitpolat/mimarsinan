"""
Latency calculation for IRGraph.

Similar to ChipLatency but operates on the unified IR representation.
"""

from __future__ import annotations

from collections import defaultdict

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore


class IRLatency:
    """
    Calculate and assign latency values for each NeuralCore in an IRGraph.

    Latency is the longest path (in hops through neural cores) from the network
    inputs to a given core's output.
    """

    def __init__(self, ir_graph: IRGraph):
        self.ir_graph = ir_graph
        self.memo: dict[tuple[int, int], int] = {}
        self._node_id_to_core: dict[int, NeuralCore] = {
            n.id: n for n in ir_graph.nodes if isinstance(n, NeuralCore)
        }

    def _get_non_zero_axon_sources(self, core: NeuralCore, neuron_idx: int) -> list[IRSource]:
        """Get input sources that have non-zero weights for a given neuron."""
        non_zero_sources = []
        for axon_idx, w in enumerate(core.core_matrix[:, neuron_idx]):
            if abs(w) > 0:
                src = core.input_sources.flat[axon_idx]
                non_zero_sources.append(src)
        return non_zero_sources

    def _is_direct_signal(self, source: IRSource) -> bool:
        """Check if source is a direct input (not from another core)."""
        return source.node_id < 0

    def get_delay_for(self, source: IRSource) -> int:
        """
        Get the delay (latency) for a given source.

        Returns 0 for direct inputs, or 1 + max(delays of inputs) for cores.
        """
        key = (source.node_id, source.index)

        if key in self.memo:
            return self.memo[key]

        if self._is_direct_signal(source):
            self.memo[key] = 0
            return 0

        core = self._node_id_to_core.get(source.node_id)
        if core is None:
            # Source is from a ComputeOp or unknown node - treat as direct signal
            self.memo[key] = 0
            return 0

        non_zero_sources = self._get_non_zero_axon_sources(core, source.index)

        if len(non_zero_sources) == 0:
            self.memo[key] = 0
            return 0

        result = 1 + max(self.get_delay_for(src) for src in non_zero_sources)
        self.memo[key] = result
        return result

    def calculate(self) -> int:
        """
        Calculate latencies for all neural cores in the graph.

        Returns the maximum latency (depth of the network).
        """
        self.memo = {}

        # Calculate delays for all output sources (backward traversal from outputs)
        max_delay = 0
        for src in self.ir_graph.output_sources.flat:
            if isinstance(src, IRSource):
                delay = self.get_delay_for(src)
                max_delay = max(max_delay, delay)

        # Also traverse from ALL neural cores to ensure we cover every node
        # (some nodes might not be on the output path but still need latency)
        for core in self._node_id_to_core.values():
            for neuron_idx in range(core.core_matrix.shape[1]):
                src = IRSource(node_id=core.id, index=neuron_idx)
                delay = self.get_delay_for(src)
                max_delay = max(max_delay, delay)

        # Assign latency to each neural core
        latencies: dict[int, int] = defaultdict(int)
        for key in self.memo:
            node_id, neuron_idx = key
            if node_id >= 0 and node_id in self._node_id_to_core:
                # Latency is delay - 1 (core itself adds 1 hop)
                latencies[node_id] = max(latencies[node_id], self.memo[key] - 1)

        # Ensure ALL neural cores have a latency value set
        # Cores not reached by any path get latency 0 (input layer)
        for node_id, core in self._node_id_to_core.items():
            if node_id in latencies:
                core.latency = latencies[node_id]
            else:
                # Core not in any path - this shouldn't happen but set to 0
                core.latency = 0

        return max_delay

