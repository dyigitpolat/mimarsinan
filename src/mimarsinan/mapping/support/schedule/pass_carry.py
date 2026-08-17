"""The stage-level carry census: what crosses a program's pass boundaries.

:mod:`pass_cut` answers the question over a segment's CORE DAG, before packing. This
module answers it over the packed program's stages, which is what a record can seal and
a cost model can price. Both read the same rule — a wire read by a later PASS of the
same segment crosses verbatim or collapses; a wire read by a later SEGMENT is a host
boundary and always collapses.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from mimarsinan.mapping.support.schedule.pass_cut import (
    COLLAPSE,
    TRANSFER_DISCIPLINES,
    VERBATIM,
    raster_bytes,
)


def carried_wire_bytes(width: int, timesteps: int, transfer: str) -> int:
    """Bytes one carried wire occupies under ``transfer`` — the cost of the choice.

    VERBATIM buffers one bit per (neuron, cycle); COLLAPSE buffers a window count,
    which fits in ``log2(T + 1)`` bits. That ratio IS the trade between the two
    disciplines, and it is why both exist rather than one being an error.
    """
    if transfer == VERBATIM:
        return raster_bytes(width, timesteps)
    if transfer != COLLAPSE:
        raise ValueError(f"unknown pass transfer {transfer!r}; "
                         f"expected one of {TRANSFER_DISCIPLINES}")
    return int(width) * ((max(int(timesteps), 1).bit_length() + 7) // 8)


def boundary_transfer_bytes(*, width: int, timesteps: int) -> int:
    """Bytes ONE carried wire moves across the chip boundary, per direction.

    Dense under BOTH disciplines, and that is a fact of the runner rather than
    a modelling choice: what leaves the chip is the window's emissions (both
    disciplines read the same spike trace; COLLAPSE's reduction to window
    counts happens host-side afterwards), and what re-enters is a spike train,
    because the chip's input interface takes trains and nothing else.

    So the discipline chooses what the HOST BUFFERS between the two passes
    (:func:`carried_wire_bytes`), not what crossed — under COLLAPSE the
    transfer is strictly larger than the buffer. Dense is also an upper bound
    on an address-event link, where the cost would follow the emitted spikes;
    pricing that needs a declared per-event width nobody declares yet.
    """
    return raster_bytes(int(width), int(timesteps))


def carried_wire_spans(stages: Sequence) -> Tuple[Tuple[int, int, int, int], ...]:
    """``(node_id, width, produced_in, last_consumed_in)`` per carried wire.

    The census twin of :func:`carried_outputs_by_stage`, over the same reader map:
    widths come from the producing stage's own output slice, so nothing re-derives a
    size the program already states.
    """
    reads = _stage_reads(stages)
    spans = []
    for index, stage in enumerate(stages):
        segment = _segment_of(stage)
        if segment is None:
            continue
        for slice_ in getattr(stage, "output_map", ()) or ():
            node_id = int(slice_.node_id)
            later = [
                reader for reader in reads.get(node_id, ())
                if reader > index and _segment_of(stages[reader]) == segment
            ]
            if later:
                spans.append((node_id, int(slice_.size), index, max(later)))
    return tuple(spans)


def carry_census_from_spans(
    spans: Sequence[Tuple[int, int, int, int]],
    *,
    boundary_count: int,
    timesteps: int,
    transfer: str,
) -> Dict[str, int]:
    """The carry census of ANY pass structure that can state its spans.

    ONE census for both completenesses: the record produces spans from its
    sealed hybrid stages (:func:`carried_wire_spans`), a search candidate from
    its planned pass placements (:func:`carried_softcore_spans`) — the same
    numbers because this is the same function. Wires whose live ranges do not
    overlap share the buffer, so the peak is the worst boundary rather than
    the total — the register-allocation shape, priced under whichever
    discipline the run actually executes.
    """
    total = sum(carried_wire_bytes(w, timesteps, transfer) for _, w, _, _ in spans)
    peak = 0
    for boundary in range(int(boundary_count)):
        live = sum(
            carried_wire_bytes(w, timesteps, transfer)
            for _, w, start, end in spans if start <= boundary < end
        )
        peak = max(peak, live)
    crossing = sum(
        boundary_transfer_bytes(width=w, timesteps=timesteps)
        for _, w, _, _ in spans
    )
    return {
        "carried_wires": len(spans),
        "carried_bytes": total,
        "peak_live_bytes": peak,
        # [E3] Charged per DIRECTION: the two crossings ride different
        # channels and are priced by different constants.
        "boundary_out_bytes": crossing,
        "boundary_in_bytes": crossing,
    }


def pass_carry_census(
    stages: Sequence, timesteps: int, transfer: str
) -> Dict[str, int]:
    """The sealed program's census — spans from its own hybrid stages."""
    return carry_census_from_spans(
        carried_wire_spans(stages),
        boundary_count=len(stages), timesteps=timesteps, transfer=transfer,
    )


def carried_softcore_spans(
    softcores: Sequence,
    pass_placements: Sequence[Sequence[Tuple[int, int]]],
    pair_wires,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """[H2] A candidate's carried-wire spans, from its planned pass structure.

    The record's rule restated over layout facts: a producer read by a LATER
    pass of the SAME segment is carried, at the width of its whole published
    slice (the record sizes by the producing stage's output slice), live until
    its last consuming pass. Adjacency comes from the wire-census walk
    (``pair_wires``); membership from the planner's own placements — the same
    membership the NoC estimator prices carried re-entry traffic with [E5].
    """
    membership: Dict[int, int] = {}
    for pass_index, placements in enumerate(pass_placements):
        for softcore_index, _hardcore in placements:
            membership.setdefault(int(softcore_index), pass_index)
    last_read: Dict[int, int] = {}
    for (producer, consumer) in pair_wires:
        produced = membership.get(int(producer))
        consumed = membership.get(int(consumer))
        if produced is None or consumed is None or consumed <= produced:
            continue
        if _softcore_segment(softcores[int(producer)]) != _softcore_segment(
                softcores[int(consumer)]):
            continue  # a later SEGMENT is a host boundary, never a carry
        key = int(producer)
        last_read[key] = max(last_read.get(key, consumed), consumed)
    return tuple(
        (producer, int(softcores[producer].output_count),
         membership[producer], last)
        for producer, last in sorted(last_read.items())
    )


def _softcore_segment(softcore) -> int:
    return int(getattr(softcore, "segment_id", 0) or 0)


def _segment_of(stage):
    if getattr(stage, "kind", None) != "neural":
        return None
    return getattr(stage, "schedule_segment_index", None)


def _stage_reads(stages: Sequence) -> Dict[int, List[int]]:
    reads: Dict[int, List[int]] = {}
    for index, stage in enumerate(stages):
        if _segment_of(stage) is None:
            continue
        for slice_ in getattr(stage, "input_map", ()) or ():
            reads.setdefault(int(slice_.node_id), []).append(index)
    return reads


def carried_outputs_by_stage(stages: Sequence) -> Dict[int, Tuple[int, ...]]:
    """``{stage_index: node_ids whose raster a LATER pass of the same segment reads}``.

    Duck-typed on the hybrid stages: a wire read by a later pass of the SAME segment
    is an intra-segment pass boundary and must cross verbatim; a wire read by a later
    SEGMENT crosses a host boundary and collapses to counts by design. Distinguishing
    them is the whole of the transfer rule, and it is one comparison.
    """
    reads: Dict[int, List[int]] = {}
    for index, stage in enumerate(stages):
        if getattr(stage, "kind", None) != "neural":
            continue
        if getattr(stage, "schedule_segment_index", None) is None:
            continue
        for slice_ in getattr(stage, "input_map", ()) or ():
            reads.setdefault(int(slice_.node_id), []).append(index)
    carried: Dict[int, Tuple[int, ...]] = {}
    for index, stage in enumerate(stages):
        if getattr(stage, "kind", None) != "neural":
            continue
        segment = getattr(stage, "schedule_segment_index", None)
        if segment is None:
            continue
        ids = []
        for slice_ in getattr(stage, "output_map", ()) or ():
            node_id = int(slice_.node_id)
            for reader in reads.get(node_id, ()):
                if reader <= index:
                    continue
                if getattr(stages[reader], "schedule_segment_index", None) == segment:
                    ids.append(node_id)
                    break
        if ids:
            carried[index] = tuple(sorted(set(ids)))
    return carried
