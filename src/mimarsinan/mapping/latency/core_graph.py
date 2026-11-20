"""The orders of a core graph (cores reading cores), in O(V + E) or O((V + E) log V) keyed: ChipLatency's per-core
latency (the longest path, then the ALAP alignment) and [F2] the residency units' program order (a keyed topological
order, the strongly connected sets, the condensation in keyed order)."""

from __future__ import annotations

import heapq
from typing import Any, Callable, List, Optional, Sequence


def topological_order(sources: Sequence[Sequence[int]], key: Optional[Callable[[int], Any]] = None) -> List[int]:
    """Core indices in an order where every core follows the stage cores it reads; a cycle refuses. Without ``key``
    the ready cores in FIFO order (the latency passes'); [F2] with ``key`` the ready core of the smallest key next, so
    the order is the key order itself wherever that order is topological (its smallest remaining core then has every
    source emitted)."""
    pending = [len(set(core_sources)) for core_sources in sources]
    readers: List[List[int]] = [[] for _ in sources]
    for index, core_sources in enumerate(sources):
        for source in set(core_sources):
            readers[int(source)].append(index)
    if key is None:
        order = [index for index, count in enumerate(pending) if count == 0]
        for index in order:
            for reader in readers[index]:
                pending[reader] -= 1
                if pending[reader] == 0:
                    order.append(reader)
    else:
        ready = [(key(index), index) for index, count in enumerate(pending) if count == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            index = heapq.heappop(ready)[1]
            order.append(index)
            for reader in readers[index]:
                pending[reader] -= 1
                if pending[reader] == 0:
                    heapq.heappush(ready, (key(reader), reader))
    if len(order) != len(sources):
        raise ValueError("core graph has a cycle")
    return order


def strongly_connected_sets(sources: Sequence[Sequence[int]]) -> List[List[int]]:
    """[F2] The strongly connected sets of the graph (each core with every core it reaches and is reached by), each in
    index order: Tarjan's, iterative (no recursion limit on deep graphs)."""
    n = len(sources)
    index: List[int] = [-1] * n
    low: List[int] = [0] * n
    on_stack = [False] * n
    stack: List[int] = []
    sets: List[List[int]] = []
    counter = 0
    for root in range(n):
        if index[root] >= 0:
            continue
        work = [(root, iter(sorted(set(int(s) for s in sources[root]))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            core, pending = work[-1]
            descended = False
            for source in pending:
                if index[source] < 0:
                    index[source] = low[source] = counter
                    counter += 1
                    stack.append(source)
                    on_stack[source] = True
                    work.append((source, iter(sorted(set(int(s) for s in sources[source])))))
                    descended = True
                    break
                if on_stack[source]:
                    low[core] = min(low[core], index[source])
            if descended:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[core])
            if low[core] == index[core]:
                members = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    members.append(member)
                    if member == core:
                        break
                sets.append(sorted(members))
    return sets


def condensed_order(sources: Sequence[Sequence[int]], key: Callable[[int], Any]) -> List[List[int]]:
    """[F2] The strongly connected sets in a topological order of the condensation, the ready set of the smallest key
    next, a set's key its members' smallest (``topological_order`` with ``key``); on an acyclic graph whose key order
    is topological, every core alone in the key order."""
    sets = strongly_connected_sets(sources)
    of = {core: k for k, members in enumerate(sets) for core in members}
    condensed = [sorted({of[int(s)] for core in members for s in sources[core]} - {k})
                 for k, members in enumerate(sets)]
    keys = [min(key(core) for core in members) for members in sets]
    return [sets[k] for k in topological_order(condensed, key=keys.__getitem__)]


def longest_paths(sources: Sequence[Sequence[int]]) -> List[int]:
    """Per core, its longest path from a core that reads only stage inputs (0 there)."""
    depth = [0] * len(sources)
    for index in topological_order(sources):
        depth[index] = max((depth[int(s)] + 1 for s in sources[index]), default=0)
    return depth


def alap_latencies(sources: Sequence[Sequence[int]]) -> List[int]:
    """Per core, the latency ``ChipLatency.calculate`` gives core ``i`` reading the stage cores ``sources[i]``.

    ``sources[i]`` are core ``i``'s live sources under the one liveness predicate (``upstream.live_axons``: not off,
    not a stage input or the always-on line, a nonzero weight row), the set ChipLatency's sweep, alignment and delays
    read. Its invariant sweep leaves the longest path over them, and its shift alignment raises every core of a
    feed-forward stage to its deepest consumer's latency minus one."""
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
