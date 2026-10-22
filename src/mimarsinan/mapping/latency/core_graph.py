"""ChipLatency's per-core latency over a stage's core graph, in O(V + E): the longest path, then the ALAP alignment."""

from __future__ import annotations

from typing import List, Sequence


def topological_order(sources: Sequence[Sequence[int]]) -> List[int]:
    """Core indices in an order where every core follows the stage cores it reads; a cycle refuses."""
    pending = [len(set(core_sources)) for core_sources in sources]
    readers: List[List[int]] = [[] for _ in sources]
    for index, core_sources in enumerate(sources):
        for source in set(core_sources):
            readers[int(source)].append(index)
    order = [index for index, count in enumerate(pending) if count == 0]
    for index in order:
        for reader in readers[index]:
            pending[reader] -= 1
            if pending[reader] == 0:
                order.append(reader)
    if len(order) != len(sources):
        raise ValueError("core graph has a cycle")
    return order


def longest_paths(sources: Sequence[Sequence[int]]) -> List[int]:
    """Per core, its longest path from a core that reads only stage inputs (0 there)."""
    depth = [0] * len(sources)
    for index in topological_order(sources):
        depth[index] = max((depth[int(s)] + 1 for s in sources[index]), default=0)
    return depth


def alap_latencies(sources: Sequence[Sequence[int]]) -> List[int]:
    """Per core, the latency ``ChipLatency.calculate`` gives core ``i`` reading the stage cores ``sources[i]``.

    Its invariant sweep leaves the longest path over cross-core sources (live weights never exceed it), and its shift
    alignment raises every core of a feed-forward stage to its deepest consumer's latency minus one."""
    order = topological_order(sources)
    latency = [0] * len(sources)
    for index in order:
        latency[index] = max((latency[int(s)] + 1 for s in sources[index]), default=0)
    deepest = [-1] * len(sources)
    for index in reversed(order):
        if deepest[index] >= 0:
            latency[index] = max(latency[index], deepest[index] - 1)
        for source in sources[index]:
            deepest[int(source)] = max(deepest[int(source)], latency[index])
    return latency
