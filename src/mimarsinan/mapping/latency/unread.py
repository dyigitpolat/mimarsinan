"""[U-TGAUGE6, U-LIVE] What the window rule's dead-read pass leaves unread, taken away with the neuron as the unit.

Turning dead in-segment reads off (``cycle_windows.silence_dead_segment_reads``) can leave a producer, or some of its
neurons, with no reader. A neuron that no node and no output reads is marked through the IR's own column compaction
(``pruned_col_mask``: soft-core compaction materializes it, the static check keys by it), and a whole core is removed
only where every neuron of it is unread, by the IR's whole-node deletion (``IRGraph.remove_nodes``), so the declared
and the unbounded geometry of one network hold the same neurons (U-TGAUGE6's check, m-1)."""

from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore

__all__ = ["drop_unread"]


def drop_unread(ir_graph: IRGraph, candidates: Set[int]) -> Tuple[int, int]:
    """[U-LIVE] The unit of removal is the neuron. A whole core of ``candidates`` is removed only where every neuron of
    it is unread (``_remove_unread``, which adds the producers it reaches to ``candidates``); a neuron of a remaining
    candidate that no node and no output source reads is marked compacted (the IR's own column compaction,
    ``pruned_col_mask``, which soft-core compaction materializes and the static check keys by), so the declared and
    the unbounded geometry hold the same neurons. A psum or coalescing group's member keeps its columns (the group's
    own wiring) and goes whole or stays. Returns ``(cores removed, neurons newly marked)``."""
    if not candidates:
        return 0, 0
    removed = _remove_unread(ir_graph, candidates)
    present = {int(n.id): n for n in ir_graph.nodes}
    candidates.intersection_update(present)
    read: Dict[int, Set[int]] = {}
    for node in ir_graph.nodes:
        for s in node.input_sources.flatten():
            if isinstance(s, IRSource) and s.node_id >= 0:
                read.setdefault(int(s.node_id), set()).add(int(s.index))
    for s in np.asarray(ir_graph.output_sources).flatten():
        if isinstance(s, IRSource) and s.node_id >= 0:
            read.setdefault(int(s.node_id), set()).add(int(s.index))
    marked = 0
    for nid in sorted(candidates):
        node = present[nid]
        if not isinstance(node, NeuralCore) or not read.get(nid):
            continue
        if getattr(node, "psum_group_id", None) is not None or getattr(node, "coalescing_group_id", None) is not None:
            continue
        shape = np.shape(node.get_core_matrix(ir_graph))
        axons, width = (int(shape[0]), int(shape[1])) if len(shape) == 2 else (-1, -1)
        if axons != len(node.input_sources.flatten()) or width != int(node.get_output_count()):
            continue  # masks that cannot describe the matrix compact nothing (``compaction_masks_apply``)
        cols = getattr(node, "pruned_col_mask", None)
        rows = getattr(node, "pruned_row_mask", None)
        cols = [False] * width if cols is None or rows is None or len(cols) != width else list(cols)
        fresh = [j for j in range(width) if not cols[j] and j not in read[nid]]
        if not fresh:
            continue
        for j in fresh:
            cols[j] = True
        node.pruned_col_mask = cols
        node.pruned_row_mask = [False] * axons if rows is None or len(rows) != axons else list(rows)
        marked += len(fresh)
    return removed, marked


def _remove_unread(ir_graph: IRGraph, candidates: Set[int]) -> int:
    """Remove every core of ``candidates`` that no node and no output source reads, with its own reads, and every core
    that removal leaves the same way, to a fixed point; returns the cores removed and adds every producer the fixed
    point reaches to ``candidates`` (whose neurons the removal may leave unread). One call of the IR's whole-node
    deletion (``IRGraph.remove_nodes``, the one IR pruning removes its dead cores with) removes them all, so a psum or
    coalescing group goes whole (no reader outside it) or stays. Host operations are never removed. A removed core is
    read by nothing, so no remaining read changes: the relay pass, ``ChipLatency`` and the assertion keep one edge
    set."""
    if not candidates:
        return 0
    readers: Dict[int, Set[int]] = {}
    for node in ir_graph.nodes:
        for s in node.input_sources.flatten():
            if isinstance(s, IRSource) and s.node_id >= 0:
                readers.setdefault(int(s.node_id), set()).add(int(node.id))
    for s in np.asarray(ir_graph.output_sources).flatten():
        if isinstance(s, IRSource) and s.node_id >= 0:
            readers.setdefault(int(s.node_id), set()).add(-1)  # an output source: never inside the removed set
    cores = {int(n.id): n for n in ir_graph.nodes if isinstance(n, NeuralCore)}
    groups: Dict[Tuple[str, int], Set[int]] = {}
    for nid, core in cores.items():
        for attr in ("psum_group_id", "coalescing_group_id"):
            if getattr(core, attr, None) is not None:
                groups.setdefault((attr, int(getattr(core, attr))), set()).add(nid)

    def unit(nid: int) -> Set[int]:
        members, pending = {nid}, [nid]
        while pending:
            core = cores[pending.pop()]
            for attr in ("psum_group_id", "coalescing_group_id"):
                gid = getattr(core, attr, None)
                fresh = set() if gid is None else groups[(attr, int(gid))] - members
                members |= fresh
                pending.extend(fresh)
        return members

    removed: Set[int] = set()
    pending = sorted(candidates)
    while pending:
        nid = pending.pop()
        if nid in removed or nid not in cores:
            continue
        members = unit(nid)
        if any(not readers.get(m, set()) <= removed | members for m in members):
            continue
        removed |= members
        reached = [int(s.node_id) for m in members for s in cores[m].input_sources.flatten()
                   if isinstance(s, IRSource) and s.node_id >= 0 and int(s.node_id) not in removed]
        candidates.update(reached)
        pending.extend(reached)
    if removed:
        ir_graph.remove_nodes(removed)
    return len(removed)
