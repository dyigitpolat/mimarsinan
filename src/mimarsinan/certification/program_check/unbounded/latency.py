"""[E1 fix round] The latency and the program stage at which an IR's identity program runs each neural node, read off
the IR: the identity builder runs one stage per ``partition_ir_graph`` segment and one hard core per neural node,
whose cross-core sources are the same-segment nodes its kept rows read at a kept column (soft-core compaction turns
every other such row off, and a row reading another segment is a stage input); the streamed LIF and cascaded TTFS
executors run each core at ChipLatency's latency over that core graph (``core_graph.alap_latencies``: the longest path
over the listed sources, then every core raised to its deepest consumer's latency minus one).

Which neurons share a node, and so whom the node's core feeds, is the tiling: two IRs equal neuron by neuron can run a
neuron at different latencies (``executed_latency``). The derivation is held, on every check, to the identity program
the check builds for the deployment's IR (``timing_derivation``, through ``IdentityFacts``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np

from mimarsinan.certification.program_check.unbounded.keys import kept, source_pairs
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.latency.core_graph import alap_latencies
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.packing.pass_timing import executed_core_latencies


@dataclass(frozen=True)
class IdentityFacts:
    """What the check's identity program of the deployment's IR executes, per IR neural node: the latency it runs the
    node's core at (None where the semantics is timing-free), its program stage, and the channel table it carries."""

    latency: Optional[Mapping[int, int]]
    stage: Mapping[int, int]
    channels: Mapping[int, Any]


def _groups(nodes: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    """``(producer, positions)`` per distinct producer node id at or above 0."""
    live = np.flatnonzero(nodes >= 0)
    order = live[np.argsort(nodes[live], kind="stable")]
    if not len(order):
        return
    starts = np.flatnonzero(np.r_[True, nodes[order][1:] != nodes[order][:-1]])
    for start, end in zip(starts.tolist(), np.r_[starts[1:], len(order)].tolist()):
        yield int(nodes[order[start]]), order[start:end]


def identity_timing(graph: IRGraph) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Per IR neural node: the latency and the program stage at which its identity program runs it."""
    latency: Dict[int, int] = {}
    stage: Dict[int, int] = {}
    for index, segment in enumerate(partition_ir_graph(graph)):
        if not isinstance(segment, NeuralSegment):
            continue
        position = {int(n.id): i for i, n in enumerate(segment.nodes)}
        keep = [kept(n) for n in segment.nodes]
        columns = {int(n.id): cols for n, (_rows, cols) in zip(segment.nodes, keep)}
        sources = []
        for node, (rows, _cols) in zip(segment.nodes, keep):
            flat = np.asarray(node.input_sources, dtype=object).flatten()
            producers, idx = source_pairs(flat[rows].tolist() if len(rows) else [])
            reads = {position[p] for p, sel in _groups(producers)
                     if p in position and bool(np.isin(idx[sel], columns[p]).any())}
            sources.append(sorted(reads))
        for node, value in zip(segment.nodes, alap_latencies(sources)):
            latency[int(node.id)], stage[int(node.id)] = int(value), index
    return latency, stage


def facts_of(identity: Any, *, timed: bool) -> IdentityFacts:
    """``IdentityFacts`` of a built identity program (one IR node per hard core, its placement records naming it)."""
    latency: Dict[int, int] = {}
    stage: Dict[int, int] = {}
    for index, program_stage in enumerate(identity.stages):
        hcm = program_stage.hard_core_mapping if program_stage.kind == "neural" else None
        if hcm is None:
            continue
        executed = executed_core_latencies(hcm) if timed else [0] * len(hcm.cores)
        records = list(getattr(hcm, "soft_core_placements_per_hard_core", None) or [])
        for h, placed in enumerate(records):
            for record in placed:
                latency[int(record["ir_node_id"])], stage[int(record["ir_node_id"])] = int(executed[h]), index
    return IdentityFacts(latency if timed else None, stage, dict(getattr(identity, "node_phase_channels", {}) or {}))
