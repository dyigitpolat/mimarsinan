"""Pass projection: every packed stage (each pass of a schedule segment, each single-pool stage) runs as the fused
segment restricted to its cores."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.latency.chip import PINNED_LATENCIES, ChipLatency
from mimarsinan.mapping.latency.core_graph import alap_latencies

#: A ``HybridStage`` (``hybrid_types`` imports the packers, which import this module).
HybridStage = Any

#: A pass stage's attribute: ``{input slice index: delay}``; the slice is its producer's raster read ``delay`` (< 0)
#: cycles late, as the fused segment delivers it to the cores that read that slice.
REPLAY_DELAYS = "replay_delays"


class PassTimingError(ToolchainRefusal, ValueError):
    """A pass whose fused-segment timing cannot be projected onto its own program, refused by name."""


def _special(source) -> bool:
    return bool(getattr(source, "is_off_", False) or getattr(source, "is_always_on_", False))


def _slice_index(stage: HybridStage) -> List[int]:
    """Per stage input index, the index of the ``input_map`` slice that holds it."""
    width = max((int(s.offset) + int(s.size) for s in stage.input_map), default=0)
    of = [-1] * width
    for k, s in enumerate(stage.input_map):
        for i in range(int(s.offset), int(s.offset) + int(s.size)):
            of[i] = k
    return of


class _Placed:
    """The IR node each axon and each neuron of a pass's hard cores belongs to (``soft_core_placements``)."""

    def __init__(self, stage: HybridStage) -> None:
        hcm = stage.hard_core_mapping
        assert hcm is not None
        self.cores = list(hcm.cores)
        rows = list(getattr(hcm, "soft_core_placements_per_hard_core", []) or [])
        self.placements = [rows[h] if h < len(rows) else [] for h in range(len(self.cores))]
        self._tables: Dict[Tuple[int, str], List[int]] = {}

    def nodes(self, h: int) -> Set[int]:
        return {int(p["ir_node_id"]) for p in self.placements[h]}

    def node_at(self, h: int, index: int, *, axis: str) -> int:
        """The node of the first placement of core ``h`` covering ``index`` on ``axis``, by table (O(1) per read)."""
        table = self._tables.get((h, axis))
        if table is None:
            offset, size = f"{axis}_offset", "axons" if axis == "axon" else "neurons"
            table = [-1] * max((int(p[offset]) + int(p[size]) for p in self.placements[h]), default=0)
            for p in reversed(self.placements[h]):           # the first placement in order wins, as a scan would
                lo, n = int(p[offset]), int(p[size])
                table[lo:lo + n] = [int(p["ir_node_id"])] * n
            self._tables[(h, axis)] = table
        node = table[index] if 0 <= index < len(table) else -1
        if node < 0:
            raise PassTimingError(f"pass projection: core {h} holds no soft core at {axis} {index}")
        return node


def _segment_graph(passes: Sequence[HybridStage]) -> Tuple[Dict[int, Set[int]], List[_Placed], Set[int]]:
    """The fused segment's IR-node graph read off its passes: in-pass edges through hard-core sources, cross-pass
    edges through carried input slices (a node an earlier pass of the segment produces)."""
    placed = [_Placed(stage) for stage in passes]
    produced: Set[int] = set()
    reads: Dict[int, Set[int]] = defaultdict(set)
    for stage, pl in zip(passes, placed):
        slice_of = _slice_index(stage)
        for h, core in enumerate(pl.cores):
            for n in pl.nodes(h):
                reads.setdefault(n, set())
            for a, src in enumerate(core.axon_sources):
                if _special(src):
                    continue
                consumer = pl.node_at(h, a, axis="axon")
                if getattr(src, "is_input_", False):
                    node = int(stage.input_map[slice_of[int(src.neuron_)]].node_id)
                    if node in produced:
                        reads[consumer].add(node)
                    continue
                reads[consumer].add(pl.node_at(int(src.core_), int(src.neuron_), axis="neuron"))
        produced |= {int(s.node_id) for s in stage.output_map}
    return reads, placed, produced


def fused_latencies(reads: Dict[int, Set[int]]) -> Dict[int, int]:
    """Per IR node, the latency the fused segment runs it at (``ChipLatency`` of its identity stage)."""
    ids = sorted(reads)
    index = {n: k for k, n in enumerate(ids)}
    return dict(zip(ids, alap_latencies([[index[s] for s in reads[n] if s in index] for n in ids])))


def _core_latencies(pl: _Placed, fused: Dict[int, int], label: str) -> List[int]:
    out = []
    for h in range(len(pl.cores)):
        held = {fused[n] for n in pl.nodes(h)}
        if len(held) > 1:
            raise PassTimingError(
                f"pass projection: {label} core {h} holds soft cores {sorted(pl.nodes(h))} that the fused segment runs "
                f"at latencies {sorted(held)}; one hard core starts at one latency")
        out.append(held.pop() if held else 0)
    return out


def _own_latencies(pl: _Placed) -> Tuple[List[int], List[List[int]]]:
    sources = [sorted({int(s.core_) for s in core.axon_sources if not _special(s) and not getattr(s, "is_input_", False)})
               for core in pl.cores]
    return alap_latencies(sources), sources


def _pin(stage: HybridStage, pl: _Placed, fused: Dict[int, int]) -> None:
    """Pin a pass at the fused latencies where its own timing moves an edge between its cores."""
    pinned = _core_latencies(pl, fused, stage.name)
    own, sources = _own_latencies(pl)
    if any(own[c] - own[p] != pinned[c] - pinned[p] for c, srcs in enumerate(sources) for p in srcs):
        base = min(pinned, default=0)
        setattr(stage.hard_core_mapping, PINNED_LATENCIES, [p - base for p in pinned])


def _delay(stage: HybridStage, pl: _Placed, fused: Dict[int, int], produced: Set[int]) -> None:
    """One input slice per (carried producer, delivery delay), the delay recorded where it is not zero."""
    pinned, slice_of = _core_latencies(pl, fused, stage.name), _slice_index(stage)
    uses: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for h, core in enumerate(pl.cores):
        for a, src in enumerate(core.axon_sources):
            if not getattr(src, "is_input_", False) or _special(src):
                continue
            k = slice_of[int(src.neuron_)]
            node = int(stage.input_map[k].node_id)
            if node in produced:
                uses[(k, pinned[h] - 1 - fused[node])].append((h, a))
    if any(d > 0 for _, d in uses):
        raise PassTimingError(
            f"pass projection: {stage.name} reads a carried producer {max(d for _, d in uses)} cycles early; the "
            f"fused segment runs ALAP, so a delivery delay is never positive (hold-last replay is unexercised)")
    delays: Dict[int, int] = {}
    end = max((int(s.offset) + int(s.size) for s in stage.input_map), default=0)
    kept: Set[int] = set()
    for (k, d), axons in sorted(uses.items(), key=lambda item: (item[0][0], -item[0][1])):
        if k not in kept:
            kept.add(k)
            if d:
                delays[k] = d
            continue
        old = stage.input_map[k]
        stage.input_map.append(type(old)(node_id=int(old.node_id), offset=end, size=int(old.size)))
        delays[len(stage.input_map) - 1] = d
        for h, a in axons:
            core = pl.cores[h]
            source = copy.copy(core.axon_sources[a])
            source.neuron_ = end + int(source.neuron_) - int(old.offset)
            core.axon_sources[a] = source
            core._axon_source_spans = None
        end += int(old.size)
    if delays:
        setattr(stage, REPLAY_DELAYS, delays)


def project_passes(stages: Sequence[HybridStage]) -> None:
    """Pins and delivery delays on every packed neural stage (R1, R2): the passes of each schedule segment, and each
    stage outside a schedule (the single pool) on its own, run at the fused segment's (the identity stage's) timing. A
    hard core holding soft cores that the fused segment runs at two latencies refuses by name, single pass included."""
    groups: Dict[Tuple[str, int], List[HybridStage]] = defaultdict(list)
    for index, stage in enumerate(stages):
        if stage.kind == "neural" and stage.hard_core_mapping is not None:
            segment = stage.schedule_segment_index
            groups[("segment", int(segment)) if segment is not None else ("stage", index)].append(stage)
    for passes in groups.values():
        reads, placed, _ = _segment_graph(passes)
        fused = fused_latencies(reads)
        produced: Set[int] = set()
        for stage, pl in zip(passes, placed):
            _pin(stage, pl, fused)
            _delay(stage, pl, fused, produced)
            produced |= {int(s.node_id) for s in stage.output_map}


@dataclass(frozen=True)
class PlannedProjection:
    """The census plane's projection of a planned program [U1 W10]: per pass, the latency span it runs pinned at
    (None: its own timing), and the delayed carried slices it adds beyond one per producer, ``(producer, width)``."""

    pinned_spans: Tuple[Optional[int], ...]
    extra_slices: Tuple[Tuple[Tuple[int, int], ...], ...]


def project_planned_passes(widths: Sequence[int], pass_members: Sequence[Sequence[int]],
                           pair_wires, segment_of: Sequence[int]) -> PlannedProjection:
    """The builder's projection rules read off the walk's wire census: fused latencies per segment over the
    specs' cross-spec wires, a pass pinned where its own timing moves an edge, one slice per (producer, delay)."""
    pass_of = {int(i): q for q, members in enumerate(pass_members) for i in members}
    reads: Dict[int, Set[int]] = {i: set() for i in pass_of}
    for p, c in pair_wires:
        if int(p) in pass_of and int(c) in pass_of and segment_of[int(p)] == segment_of[int(c)]:
            reads[int(c)].add(int(p))
    fused = fused_latencies(reads)
    spans: List[Optional[int]] = []
    extra: List[Tuple[Tuple[int, int], ...]] = []
    for q, members in enumerate(pass_members):
        held = sorted({int(i) for i in members})
        pos = {i: k for k, i in enumerate(held)}
        inside = {i: [p for p in reads[i] if pass_of[p] == q] for i in held}
        own = dict(zip(held, alap_latencies([[pos[p] for p in inside[i]] for i in held])))
        moved = any(own[c] - own[p] != fused[c] - fused[p] for c in held for p in inside[c])
        spans.append(max(fused[i] for i in held) - min(fused[i] for i in held) if moved and held else None)
        delays: Dict[int, Set[int]] = defaultdict(set)
        for c in held:
            for p in reads[c]:
                if pass_of[p] < q:
                    delays[p].add(fused[c] - 1 - fused[p])
        extra.append(tuple((p, int(widths[p])) for p, ds in sorted(delays.items()) for _ in range(len(ds) - 1)))
    return PlannedProjection(tuple(spans), tuple(extra))


def replay_delays(stage) -> Dict[int, int]:
    return dict(getattr(stage, REPLAY_DELAYS, None) or {})


def carried_nodes(stages: Sequence[HybridStage], stage: HybridStage) -> Set[int]:
    """The input nodes of ``stage`` that an earlier pass of its schedule segment produces: its carried slices (empty for
    a stage outside ``stages``, such as a level unit, which reads count re-encodes)."""
    segment, produced = getattr(stage, "schedule_segment_index", None), set()
    for other in stages:
        if other is stage:
            return {int(s.node_id) for s in stage.input_map} & produced if segment is not None else set()
        if other.kind == "neural" and getattr(other, "schedule_segment_index", None) == segment:
            produced |= {int(s.node_id) for s in other.output_map}
    return set()


def executed_core_latencies(hcm) -> List[int]:
    """Per hard core, the latency it executes at (``ChipLatency``, pins honoured); the cores' stored latencies are
    kept, so a judgement of the executed timing never rewrites the built program (its pickle, its record)."""
    stored = [core.latency for core in hcm.cores]
    try:
        ChipLatency(hcm).calculate()
        return [int(core.latency or 0) for core in hcm.cores]
    finally:
        for core, latency in zip(hcm.cores, stored):
            core.latency = latency


def late_stage_input_readers(stage: HybridStage, nodes: Optional[Set[int]] = None) -> List[Tuple[int, int]]:
    """``(core, latency)`` per hard core of ``stage`` that reads a stage-input row (of ``nodes``, or any) and starts
    above latency 0 as executed (``executed_core_latencies``): such a core reads cycle t of that row at global cycle
    ``latency + t`` (the stage-input timing rule, W33), and a backend that read a stage input at the global cycle
    would hand it the row ``latency`` cycles early and never its last ``latency`` cycles."""
    if nodes is not None and not nodes:
        return []
    slice_of = _slice_index(stage)
    rows = {k for k, s in enumerate(stage.input_map) if nodes is None or int(s.node_id) in nodes}
    hcm = stage.hard_core_mapping
    executed = executed_core_latencies(hcm)
    return [(h, latency) for h, (core, latency) in enumerate(zip(hcm.cores, executed)) if latency > 0
            and any(getattr(src, "is_input_", False) and not _special(src) and slice_of[int(src.neuron_)] in rows
                    for src in core.axon_sources)]
