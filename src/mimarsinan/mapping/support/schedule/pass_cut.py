"""What a pass boundary must carry: the true graph cut of a segment's core DAG.

A pass is a SPATIAL cut of the segment, never a temporal one — every core runs its
whole cycle window inside one pass, so membrane state never crosses a boundary and only
spike rasters do. The cut is therefore every edge from a core in pass <= p to a core in
pass > p, which is not the same as "the last latency group": halving a group across
passes leaves the group's own INPUTS needed by the second half, and a wire read two
passes later stays live in between. Both are liveness facts, computed here once.

Duck-typed on purpose: the graph core needs four facts about a core, so it is testable
without building an IR (the ``LayoutStatsView`` precedent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

#: The raster crosses unchanged — streamed LIF, where rhythm is the signal.
VERBATIM = "verbatim"
#: Counts cross and are re-encoded — the windowed disciplines, which normalize
#: timing at every boundary anyway, so a pass boundary costs them nothing.
COLLAPSE = "collapse"
TRANSFER_DISCIPLINES: Tuple[str, ...] = (VERBATIM, COLLAPSE)


def transfer_for(*, streamed: bool) -> str:
    """The transfer discipline a pass boundary owes this execution semantics.

    Streamed execution has no interior transcode by definition, so a pass boundary
    inside a segment must replay the raster; collapsing it to counts would be the
    windowed transcode wearing a scheduling hat.
    """
    return VERBATIM if streamed else COLLAPSE


def raster_bytes(width: int, timesteps: int) -> int:
    """One BIT per (neuron, cycle) — a spike raster is binary by construction."""
    if width < 0 or timesteps < 0:
        raise ValueError(f"raster extent must be non-negative, got {width}x{timesteps}")
    return int(width) * ((int(timesteps) + 7) // 8)


@dataclass(frozen=True)
class CutNode:
    """The only facts about a core that a pass cut needs."""

    core_id: int
    latency: int
    sources: Tuple[int, ...]
    out_width: int


@dataclass(frozen=True)
class CarriedWire:
    """One producer's raster, and the span of passes it must survive."""

    producer: int
    width: int
    producer_latency: int
    produced_in: int
    last_consumed_in: int
    consumer_latencies: Tuple[int, ...] = ()

    @property
    def live_boundaries(self) -> Tuple[int, ...]:
        """Boundary ``b`` sits between pass ``b`` and ``b+1``."""
        return tuple(range(self.produced_in, self.last_consumed_in))

    @property
    def is_skew_free(self) -> bool:
        """Whether every consumer sits exactly one cycle after the producer.

        A carried wire re-enters its consumer pass as a SEGMENT INPUT, aligned to
        each consuming core's own start. Inside one pass the same wire is a live
        buffer handoff with a fixed one-cycle delay and NO realignment. The two
        agree only when every consumer sits at ``producer_latency + 1``; otherwise
        cutting here would silently remove a skew the fused execution has.
        """
        return all(
            latency == self.producer_latency + 1 for latency in self.consumer_latencies
        )


def _validate(nodes: Sequence[CutNode], passes: Sequence[Sequence[int]]) -> Dict[int, int]:
    known = {node.core_id for node in nodes}
    assignment: Dict[int, int] = {}
    for index, members in enumerate(passes):
        if not members:
            raise ValueError(
                f"pass {index} is empty; a pass with no cores is a reprogram that "
                f"computes nothing"
            )
        for core_id in members:
            if core_id not in known:
                raise ValueError(f"{core_id!r} is not a core of this segment")
            if core_id in assignment:
                raise ValueError(
                    f"core {core_id} is assigned to passes {assignment[core_id]} and "
                    f"{index}; every core belongs to exactly once pass"
                )
            assignment[core_id] = index
    missing = sorted(known - set(assignment))
    if missing:
        raise ValueError(
            f"cores {missing} are assigned to no pass; every core must appear "
            f"exactly once, or the segment is not fully scheduled"
        )
    for node in nodes:
        for source in node.sources:
            if source in assignment and assignment[source] > assignment[node.core_id]:
                raise ValueError(
                    f"pass assignment is not monotone over the DAG: core "
                    f"{node.core_id} (pass {assignment[node.core_id]}) reads core "
                    f"{source} (pass {assignment[source]}), which runs later"
                )
    return assignment


@dataclass(frozen=True)
class PassCut:
    """A segment's pass assignment plus everything that must cross its boundaries."""

    assignment: Mapping[int, int]
    carried: Tuple[CarriedWire, ...]
    pass_count: int

    @classmethod
    def over(
        cls, nodes: Sequence[CutNode], passes: Sequence[Sequence[int]]
    ) -> "PassCut":
        """The cut induced by assigning ``nodes`` to ``passes``, in pass order."""
        assignment = _validate(nodes, passes)
        by_id = {node.core_id: node for node in nodes}
        last_read: Dict[int, int] = {}
        consumers: Dict[int, List[int]] = {}
        for node in nodes:
            consumer_pass = assignment[node.core_id]
            for source in node.sources:
                if source not in assignment:
                    continue
                if assignment[source] < consumer_pass:
                    last_read[source] = max(
                        last_read.get(source, consumer_pass), consumer_pass
                    )
                    consumers.setdefault(source, []).append(node.latency)
        carried = tuple(
            CarriedWire(
                producer=producer,
                width=by_id[producer].out_width,
                producer_latency=by_id[producer].latency,
                produced_in=assignment[producer],
                last_consumed_in=consumed_in,
                consumer_latencies=tuple(sorted(consumers.get(producer, ()))),
            )
            for producer, consumed_in in sorted(last_read.items())
        )
        return cls(
            assignment=dict(assignment),
            carried=carried,
            pass_count=len(passes),
        )

    @property
    def skewed_carries(self) -> Tuple[CarriedWire, ...]:
        """Carried wires this cut cannot reproduce exactly (see ``is_skew_free``)."""
        return tuple(w for w in self.carried if not w.is_skew_free)

    def require_exact(self, segment: str) -> None:
        """Refuse a cut that would change the computation, naming the remedy."""
        skewed = self.skewed_carries
        if not skewed:
            return
        detail = "; ".join(
            f"core {w.producer} (latency {w.producer_latency}) read by cores at "
            f"latencies {list(w.consumer_latencies)}" for w in skewed
        )
        raise ValueError(
            f"segment {segment!r} cannot be cut here without changing the "
            f"computation: {detail}. A carried wire re-enters its consumer pass as "
            f"a segment input, which is aligned per consuming core, while inside a "
            f"pass it is a live handoff with a fixed one-cycle delay — the two agree "
            f"only for consumers at producer_latency + 1. Exact remedy: "
            f"depth-balancing relay insertion (lif_depth_balancing_relays), which "
            f"equalizes intra-segment fan-in depth."
        )

    def live_at(self, boundary: int) -> Tuple[CarriedWire, ...]:
        """The wires that must still be held across boundary ``boundary``."""
        return tuple(w for w in self.carried if boundary in w.live_boundaries)

    def cut_width(self, boundary: int) -> int:
        return sum(w.width for w in self.live_at(boundary))

    def carried_bytes(self, timesteps: int) -> int:
        """Total raster payload written across every boundary of this segment."""
        return sum(raster_bytes(w.width, timesteps) for w in self.carried)

    def peak_live_bytes(self, timesteps: int) -> int:
        """The buffer a run actually needs: wires whose ranges do not overlap
        share storage, so the peak is the worst boundary, never the total."""
        if not self.carried:
            return 0
        return max(
            raster_bytes(self.cut_width(boundary), timesteps)
            for boundary in range(max(self.pass_count - 1, 1))
        )


def cut_nodes_from_cores(cores: Iterable, segment_ids: Iterable[int]) -> List[CutNode]:
    """Adapt IR ``NeuralCore``s to the graph core, keeping only intra-segment edges.

    A source outside the segment enters through the segment's own input map and is
    never carried — it is the host boundary, which collapses to counts by design.
    """
    inside = set(segment_ids)
    nodes: List[CutNode] = []
    for core in cores:
        sources = tuple(sorted({
            int(source.node_id)
            for source in core.input_sources.flatten()
            if getattr(source, "node_id", None) is not None
            and int(source.node_id) in inside
        }))
        nodes.append(CutNode(
            core_id=int(core.id),
            latency=int(core.latency or 0),
            sources=sources,
            out_width=int(core.core_matrix.shape[1]) if core.core_matrix is not None
            else 0,
        ))
    return nodes
