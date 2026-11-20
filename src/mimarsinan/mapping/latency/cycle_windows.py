"""[U-TGAUGE5] The window rule of every per-cycle cascade: each live in-segment edge has a latency gap of one.

In a per-cycle cascade (``SpikingModePolicy.runs_per_cycle_cascade``: the LIF family and the timing family's cascaded
schedule) each core steps from its latency ``L``, which ``ChipLatency`` sets (the longest path over the stage's core
reads, then the alignment that raises each core to its deepest reader's latency minus one), and fills its axons at cycle
``L + k`` from each producer core's buffer of the previous cycle: that producer's relative cycle ``k + g - 1`` at a
latency gap ``g``. Only ``g = 1`` reads the producer's relative cycle ``k``: at ``g = 0`` (the alignment raises a
producer read at two depths to one below its deepest reader) the shallower reader reads it a cycle late and loses its
last cycle, at ``g >= 2`` it drops the producer's first ``g - 1`` cycles and re-reads its last buffer. The mapping makes
``g = 1`` on every live in-segment edge (dead reads off, then the depth-balancing relays: Soft Core Mapping before
quantization, ``depth_balancing.balance_cycle_windows`` for a program built outside it) or refuses the program by name,
and the per-cycle executor refuses a stage holding another gap. The analytical, quantized and synchronized timing
schedules run each core in dependency order over complete values (``chip_simulation/ttfs/ttfs_segment.py``): no window.

A pass of a fused segment (pass projection) runs its cores at the fused latencies and reads an earlier pass's producer
as a carried slice delayed ``d = g - 1`` cycles: a nonzero delivery delay is a read at a gap other than one, which the
executor reads with the in-pass edges (``carried_window_gaps``).

One edge set everywhere: an edge is an in-segment read that the one liveness predicate keeps ([U-LIVE]
``upstream.live_axons``: not off, not a segment input or the always-on line, a nonzero weight row), the set the relay
pass, ``ChipLatency`` and the assertion below read. ``silence_dead_segment_reads`` also turns every dead in-segment read
off before the relays, and takes away what those reads leave unread, the neuron being the unit: a neuron no node and no
output reads is marked compacted, and a core every neuron of which is unread carries nothing to anyone (under per-hop
re-timing it would be a level stage without outputs) and goes with its own reads, to a fixed point, by the IR's
whole-node deletion (``IRGraph.remove_nodes``). The latencies here are ``ChipLatency``'s: ``core_graph.alap_latencies``
over the live in-segment reads, locked to it, on the IR; the cores' own (pins honoured) on a hard-core mapping.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Set, Tuple

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph, IRSource
from mimarsinan.mapping.latency.core_graph import alap_latencies
from mimarsinan.mapping.latency.unread import drop_unread
from mimarsinan.mapping.latency.upstream import core_reads, ir_reads
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph

__all__ = [
    "DECLARED_OFF", "DeadReads", "NOT_ARMED", "RELAYS_LEFT_GAP", "RELAY_KEY", "UnbalancedFanInError", "WindowGap",
    "carried_window_gaps", "core_window_gaps", "ir_window_gaps", "mapping_cause", "require_cycle_windows",
    "silence_dead_segment_reads",
]

#: The declared option that switches the depth-balancing relays of a per-cycle cascade off (its recipe arms it).
RELAY_KEY = "lif_depth_balancing_relays"

#: The window refusal's three causes (one class, one edge read, one remedy).
DECLARED_OFF = f"the depth-balancing relays are declared off ({RELAY_KEY} = false)"
NOT_ARMED = (f"no mapping armed the depth-balancing relays on it (no conversion recipe folded {RELAY_KEY} and no "
             f"declaration set it, or the program was built outside Soft Core Mapping)")
RELAYS_LEFT_GAP = "the depth-balancing relays ran and left it (a defect of the mapping, never of the program)"


class UnbalancedFanInError(ToolchainRefusal, ValueError):
    """A live in-segment edge of a per-cycle cascade whose latency gap after the alignment is not one."""


class WindowGap(NamedTuple):
    producer: int
    consumer: int
    gap: int


def _gaps(live: Sequence[Sequence[int]], latency: Sequence[int], ids: Sequence[int]) -> List[WindowGap]:
    out: List[WindowGap] = []
    for consumer, producers in enumerate(live):
        for producer in sorted(set(int(p) for p in producers)):
            gap = int(latency[consumer]) - int(latency[producer])
            if gap != 1:
                out.append(WindowGap(int(ids[producer]), int(ids[consumer]), gap))
    return out


def _segment_reads(ir_graph: IRGraph, seg: NeuralSegment):
    """``(node index, flat sources, live, [(axon, producer index)])`` per node: its reads of segment cores that are not
    off, and per axon the one liveness predicate (``upstream.ir_live_reads`` with the segment as the unit)."""
    position = np.full(max(int(n.id) for n in ir_graph.nodes) + 1, -1, dtype=np.int64)  # node id -> segment index
    position[[int(node.id) for node in seg.nodes]] = np.arange(len(seg.nodes))
    for k, node in enumerate(seg.nodes):
        flat = node.input_sources.flatten()
        ids, live = ir_reads(node, ir_graph, members=position)
        at = np.flatnonzero((ids >= 0) & (ids < len(position)))
        at = at[position[ids[at]] >= 0]  # the reads of segment cores, with their producers' indices
        yield k, flat, live, list(zip(at.tolist(), position[ids[at]].tolist()))


class _Counts(NamedTuple):
    reads: int
    cores: int


class DeadReads(_Counts):
    """What turning dead in-segment reads off did: the reads turned off and the cores they left unread, removed (the
    two counts, compared as a pair), and [U-LIVE] ``neurons``, the neurons they left unread in a core whose other
    neurons are read (marked compacted)."""

    neurons: int

    def __new__(cls, reads: int, cores: int, neurons: int = 0) -> "DeadReads":
        self = super().__new__(cls, reads, cores)
        self.neurons = int(neurons)
        return self

    def record(self) -> Dict[str, int]:
        """The mapping record's statement of the pass: present only where a read was turned off, so a program without
        one keeps its record's bytes; the neurons marked only where there are any."""
        if not self.reads:
            return {}
        marked = {"unread_neurons_marked": self.neurons} if self.neurons else {}
        return {"dead_reads_off": self.reads, "unread_cores_removed": self.cores, **marked}


def silence_dead_segment_reads(ir_graph: IRGraph) -> DeadReads:
    """Turn every in-segment read that the one liveness predicate finds dead into an off source, in place: it carries
    nothing, so no value changes, and the relay pass, ``ChipLatency`` and the window assertion then read one edge set.
    In the same pass, take away what those reads leave unread (``unread.drop_unread``): [U-LIVE] the unit is the neuron, so
    both geometries of one network hold the same neurons. Marking a neuron can leave a row of its core charged in no
    kept column, so the two alternate to a fixed point. Returns ``DeadReads`` (zeros on a program without dead reads,
    which is left byte for byte)."""
    silenced = removed = marked = 0
    candidates: Set[int] = set()
    while True:
        reads, producers = _turn_dead_reads_off(ir_graph)
        silenced += reads
        candidates |= producers
        cores, neurons = drop_unread(ir_graph, candidates)
        removed += cores
        marked += neurons
        if not (cores or neurons):
            return DeadReads(silenced, removed, marked)


def _turn_dead_reads_off(ir_graph: IRGraph) -> Tuple[int, Set[int]]:
    """One sweep: every in-segment read that is not live turned off; ``(reads turned off, their producers)``."""
    silenced = 0
    producers: Set[int] = set()
    for seg in partition_ir_graph(ir_graph):
        if not isinstance(seg, NeuralSegment):
            continue
        for k, flat, live, inside in list(_segment_reads(ir_graph, seg)):
            dead = [int(i) for i, _producer in inside if not live[i]]
            if not dead:
                continue
            node = seg.nodes[k]
            for i in dead:
                producers.add(int(flat[i].node_id))
                flat[i] = IRSource(node_id=-1, index=0)
            node.input_sources = flat.reshape(node.input_sources.shape)
            silenced += len(dead)
    return silenced, producers


def ir_window_gaps(ir_graph: IRGraph, *, per_hop: bool = False) -> List[WindowGap]:
    """Every live in-segment edge of ``ir_graph`` whose gap after the alignment is not one, by IR node ids, over the
    units the hybrid build executes: its neural segments, or one depth level each under per-hop re-timing
    (``per_hop``)."""
    gaps: List[WindowGap] = []
    for seg in partition_ir_graph(ir_graph, per_hop=per_hop):
        if not isinstance(seg, NeuralSegment):
            continue
        live: List[List[int]] = [[] for _ in seg.nodes]
        for k, _flat, alive, inside in _segment_reads(ir_graph, seg):
            live[k] = sorted({p for i, p in inside if alive[i]})
        gaps.extend(_gaps(live, alap_latencies(live), [node.id for node in seg.nodes]))
    return gaps


def core_window_gaps(mapping) -> List[WindowGap]:
    """Every live core-to-core edge of one stage's hard-core mapping whose executed latencies (``ChipLatency``'s, pins
    honoured, already set on its cores) differ by other than one, by core index."""
    cores = list(mapping.cores)
    latency = [int(c.latency or 0) for c in cores]
    live: List[List[int]] = []
    for core in cores:
        if core.latency is None:  # never stepped: it reads nothing
            live.append([])
            continue
        reads = core_reads(core)
        live.append(sorted(set(int(c) for c in reads.core[reads.live])))
    return _gaps(live, latency, list(range(len(cores))))


def carried_window_gaps(mapping, input_map: Sequence, delays: Dict[int, int]) -> List[WindowGap]:
    """Every live read of a pass's carried slice whose delivery delay ``d`` (``delays``, by input slice, the carried
    slices only: an earlier pass of the same fused segment produces them) is not zero: a read of that producer at the
    fused latency gap ``d + 1``, by the slice's producer node id and the reading core's index."""
    if not any(delays.values()):
        return []
    of: Dict[int, int] = {}
    for k, s in enumerate(input_map):
        for i in range(int(s.offset), int(s.offset) + int(s.size)):
            of[i] = k
    gaps = set()
    for h, core in enumerate(mapping.cores):
        reads = core_reads(core, carried=lambda _neuron: True)
        neuron = reads.neuron
        for a in np.flatnonzero(reads.live & reads.stage_input):
            k = of.get(int(neuron[a]))
            if k is None:
                continue
            d = int(delays.get(k, 0))
            if d:
                gaps.add(WindowGap(int(input_map[k].node_id), h, d + 1))
    return sorted(gaps)


def mapping_cause(declared, *, armed: bool) -> str:
    """The cause of a gap found at mapping: the relays ran (``armed``), else the declared option's own state."""
    if armed:
        return RELAYS_LEFT_GAP
    return DECLARED_OFF if declared is False else NOT_ARMED


def require_cycle_windows(gaps: Sequence[WindowGap], *, where: str, cause: str) -> None:
    """Refuse a per-cycle cascade holding ``gaps`` by name (``UnbalancedFanInError``) with its ``cause``."""
    if not gaps:
        return
    shown = list(gaps)[:8]
    detail = ", ".join(f"{g.producer}->{g.consumer} (gap {g.gap})" for g in shown)
    more = f" and {len(gaps) - len(shown)} more" if len(gaps) > len(shown) else ""
    raise UnbalancedFanInError(
        f"{where}: a per-cycle cascade reads a producer of its segment at a latency gap other than one on "
        f"{len(gaps)} live edge(s): {detail}{more}; {cause}. A reader at gap 0 reads its producer a cycle late, at a "
        f"gap above one it drops the producer's early cycles and re-reads its last buffer. The remedy is the "
        f"depth-balancing relays (mapping/latency/depth_balancing.py), armed for every per-cycle cascade.")
