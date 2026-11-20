"""The stage-level carry census: what crosses a program's pass boundaries.

:mod:`pass_cut` answers the question over a segment's CORE DAG, before packing. This
module answers it over the packed program's stages, which is what a record can seal and
a cost model can price. Both read the same rule — a wire read by a later PASS of the
same segment crosses verbatim or collapses; a wire read by a later SEGMENT is a host
boundary and always collapses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.support.schedule.pass_cut import (
    COLLAPSE,
    TRANSFER_DISCIPLINES,
    VERBATIM,
    raster_bytes,
)
from mimarsinan.mapping.support.wavefront_stages import input_slot


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
    size the program already states. [thru:w3] Under the batch pipeline a wire is one
    input's: a stage of one input slot reads only its slot's values, so a producer's
    readers are the later stages of its own input (``same_input``), one span per input
    slot and node.
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
                and same_input(stage, stages[reader])
            ]
            if later:
                spans.append((node_id, int(slice_.size), index, max(later)))
    return tuple(spans)


def delayed_input_slices(stages: Sequence) -> Tuple[Tuple[int, int, int], ...]:
    """[U1 W10] ``(node_id, width, reading stage)`` per carried input slice a pass reads beyond its first slice of
    that producer: the builder replays a carried producer once per delivery delay (``pass_timing``), and each slice
    beyond the first crosses the chip boundary once more, at the producer's width (G5 report 23.6), the width the
    dense crossing counts too, also where the pass reads only part of the producer. The record twin of
    ``noc.fragments.delayed_carry_wires``. [thru:w3] Under the batch pipeline per input slot and node."""
    produced: Dict[Tuple[Optional[int], Optional[int]], Dict[int, int]] = {}
    out = []
    for index, stage in enumerate(stages):
        segment = _segment_of(stage)
        if segment is None:
            continue
        carried = produced.setdefault((segment, _slot_of(stage)), {})  # [thru:w3] one input's producers
        seen: set = set()
        for slice_ in getattr(stage, "input_map", ()) or ():
            node_id = int(slice_.node_id)
            if node_id in carried and node_id in seen:
                out.append((node_id, carried[node_id], index))
            seen.add(node_id)
        for s in getattr(stage, "output_map", ()) or ():
            carried[int(s.node_id)] = carried.get(int(s.node_id), 0) + int(s.size)
    return tuple(out)


def carry_census_from_spans(
    spans: Sequence[Tuple[int, int, int, int]],
    *,
    boundary_count: int,
    timesteps: int,
    transfer: str,
    extra_in: Sequence[Tuple[int, int, int]] = (),
    groups: Optional[Sequence[Optional[int]]] = None,
) -> Dict[str, int]:
    """The carry census of ANY pass structure that can state its spans.

    ONE census for both completenesses: the record produces spans from its
    sealed hybrid stages (:func:`carried_wire_spans`), a search candidate from
    its planned pass placements (:func:`carried_softcore_spans`) — the same
    numbers because this is the same function. Wires whose live ranges do not
    overlap share the buffer, so the peak is the worst boundary rather than
    the total — the register-allocation shape, priced under whichever
    discipline the run actually executes. ``extra_in``: ``(node, width, pass)``
    per delayed slice beyond one per producer and pass (:func:`delayed_input_slices`,
    ``delayed_carry_wires``), each one more boundary-in crossing of the buffered raster. [U-F4] ``groups`` (per
    boundary position, its wavefront group or None) adds the share of the census held inside one group
    (``onchip_*``), which a declared on-chip carry buffer keeps off the boundary; its live peak is stated in carried
    values too (``onchip_peak_live_elements``), so each domain sizes it at its own grid.
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
    replayed = sum(boundary_transfer_bytes(width=w, timesteps=timesteps) for _, w, _ in extra_in)
    census = {
        "carried_wires": len(spans),
        "carried_bytes": total,
        "peak_live_bytes": peak,
        # [E3] Charged per DIRECTION: the two crossings ride different
        # channels and are priced by different constants.
        "boundary_out_bytes": crossing,
        "boundary_in_bytes": crossing + replayed,
    }
    if groups is not None and any(g is not None for g in groups):
        census.update(_onchip_share(spans, extra_in, groups, timesteps=timesteps, transfer=transfer))
    return census


def group_held(spans, extra_in, groups) -> Tuple[list, list]:
    """[U-F4] The spans and delayed slices one wavefront group produces and reads itself (``groups``: per pass
    position, its group or None): the share a declared on-chip carry buffer keeps off the boundary. [U-J targeted M2]
    One rule for the census's bytes (``_onchip_share``) and the per-boundary split (``ProgramPasses``)."""
    inside = [(n, w, a, b) for n, w, a, b in spans if groups[a] is not None and groups[a] == groups[b]]
    held = {n: groups[a] for n, _w, a, _b in inside}
    return inside, [(n, w, reader) for n, w, reader in extra_in if n in held and groups[reader] == held[n]]


def _onchip_share(spans, extra_in, groups, *, timesteps: int, transfer: str) -> Dict[str, int]:
    """[U-F4] The census of the wires one wavefront group produces and reads: its crossings (out, and in with the
    group's own replays) and the peak live carry over its pass boundaries, in the transfer's buffered bytes and in
    carried values (every wire's bytes are its width times one per-value size, so one boundary holds both peaks)."""
    inside, replays = group_held(spans, extra_in, groups)
    out = sum(boundary_transfer_bytes(width=w, timesteps=timesteps) for _, w, _, _ in inside)
    again = sum(boundary_transfer_bytes(width=w, timesteps=timesteps) for _n, w, _reader in replays)
    live = [[w for _, w, a, b in inside if a <= boundary < b] for boundary in range(len(groups))]
    peak = max((sum(carried_wire_bytes(w, timesteps, transfer) for w in wires) for wires in live), default=0)
    return {"onchip_out_bytes": out, "onchip_in_bytes": out + again, "onchip_peak_live_bytes": peak,
            "onchip_peak_live_elements": max((sum(wires) for wires in live), default=0)}


def pass_carry_census(
    stages: Sequence, timesteps: int, transfer: str, groups: Optional[Sequence[Optional[int]]] = None,
) -> Dict[str, int]:
    """The sealed program's census — spans and delayed slices from its own hybrid stages (``groups``: per stage, its
    wavefront group, [U-F4]). [thru:w3] A program of input slots is read at its pass positions (``pass_positions``):
    the stages of one pass, one per input slot, run as one pass, so a span lives over passes, as the candidate plane
    counts it."""
    spans, extra = carried_wire_spans(stages), delayed_input_slices(stages)
    positions = pass_positions(stages)
    if positions is None:
        return carry_census_from_spans(spans, boundary_count=len(stages), timesteps=timesteps, transfer=transfer,
                                       extra_in=extra, groups=groups)
    at = list(positions)
    placed: List[Optional[int]] = [None] * (max(at, default=-1) + 1)
    for index, group in enumerate(groups or ()):
        if group is not None:
            placed[at[index]] = group
    return carry_census_from_spans(
        [(n, w, at[a], at[b]) for n, w, a, b in spans], boundary_count=len(placed), timesteps=timesteps,
        transfer=transfer, extra_in=[(n, w, at[r]) for n, w, r in extra],
        groups=None if groups is None else placed,
    )


def _slot_of(stage) -> Optional[int]:
    slot = input_slot(stage)
    return None if slot is None else int(slot.slot)


def same_input(producer, reader) -> bool:
    """[thru:w3] Whether ``reader`` reads ``producer``'s values: always without the batch pipeline; under it only within
    one input slot (a stage that names no slot serves every input)."""
    a, b = _slot_of(producer), _slot_of(reader)
    return a is None or b is None or a == b


def pass_positions(stages: Sequence) -> Optional[Tuple[int, ...]]:
    """[thru:w3] Per stage of a program of input slots, its pass position: the stages of one (segment, pass index),
    one per input slot, share one, every other stage holds its own, in program order. None for a program without input
    slots (each stage its own position)."""
    if not any(input_slot(stage) is not None for stage in stages):
        return None
    out: List[int] = []
    seen: Dict[Tuple[int, int], int] = {}
    count = 0
    for stage in stages:
        key = None
        if getattr(stage, "kind", None) == "neural" and getattr(stage, "schedule_segment_index", None) is not None:
            key = (int(stage.schedule_segment_index), int(getattr(stage, "schedule_pass_index", 0) or 0))
        if key is not None and key in seen:
            out.append(seen[key])
            continue
        if key is not None:
            seen[key] = count
        out.append(count)
        count += 1
    return tuple(out)


def carried_softcore_spans(
    softcores: Sequence,
    pass_placements: Sequence[Sequence[Tuple[int, int]]],
    pair_wires,
    segment_of: Optional[Sequence[int]] = None,
    pass_slots: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """[H2] A candidate's carried-wire spans, from its planned pass structure.

    The record's rule restated over layout facts: a producer read by a LATER
    pass of the SAME segment is carried, at the width of its whole published
    slice (the record sizes by the producing stage's output slice), live until
    its last consuming pass. Adjacency comes from the wire-census walk
    (``pair_wires``); membership from the planner's own placements — the same
    membership the NoC estimator prices carried re-entry traffic with [E5].
    ``segment_of``: each spec's segment the passes were planned in (the
    fragments' builder segments, ``noc.fragments.spec_segments``; [fix_b] the
    walk's runs), else its depth segment. [thru:w3] ``pass_slots`` (per pass, each
    placement's input slot, the fragments' ``pass_slots``): under the batch pipeline
    membership and spans are per input slot and spec, one span per input, as the
    record's stages carry them (``carried_wire_spans``).
    """
    segment_of = ([int(s) for s in segment_of] if segment_of is not None
                  else [_softcore_segment(softcore) for softcore in softcores])
    membership: Dict[Tuple[Optional[int], int], int] = {}
    for pass_index, placements in enumerate(pass_placements):
        slots = None if pass_slots is None else pass_slots[pass_index]
        for k, (softcore_index, _hardcore) in enumerate(placements):
            membership.setdefault((None if slots is None else int(slots[k]), int(softcore_index)), pass_index)
    last_read: Dict[Tuple[Optional[int], int], int] = {}
    for slot in sorted({b for b, _i in membership}, key=lambda b: -1 if b is None else b):
        for (producer, consumer) in pair_wires:
            produced = membership.get((slot, int(producer)))
            consumed = membership.get((slot, int(consumer)))
            if produced is None or consumed is None or consumed <= produced:
                continue
            if segment_of[int(producer)] != segment_of[int(consumer)]:
                continue  # a later SEGMENT is a host boundary, never a carry
            key = (slot, int(producer))
            last_read[key] = max(last_read.get(key, consumed), consumed)
    return tuple(
        (producer, int(softcores[producer].output_count),
         membership[(slot, producer)], last)
        for (slot, producer), last in sorted(last_read.items(), key=lambda kv: (-1 if kv[0][0] is None else kv[0][0],
                                                                                kv[0][1]))
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
    them is the whole of the transfer rule, and it is one comparison. [thru:w3] Under
    the batch pipeline a later pass of another input slot never reads it (``same_input``).
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
                if reader <= index or not same_input(stage, stages[reader]):  # [thru:w3] one input's readers
                    continue
                if getattr(stages[reader], "schedule_segment_index", None) == segment:
                    ids.append(node_id)
                    break
        if ids:
            carried[index] = tuple(sorted(set(ids)))
    return carried
