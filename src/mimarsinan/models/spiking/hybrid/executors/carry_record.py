"""The carry recording of the cycle executors: a producing pass's output raster in producer-local time.

Moved from ``models/spiking/hybrid/carry.py`` (module budget); ``carry`` re-exports every name."""

from __future__ import annotations


def record_reference_carry(
    carry, output_spans, cores, *, cycle: int, buffers, input_spikes, T: int
) -> None:
    """One cycle of the REFERENCE loop's output raster, producer-local time.

    The per-core twin of the packed executor's ``record_carry``: same span walk,
    same window discipline (a cycle outside a producer's ``[lat, lat+T)`` window
    is SKIPPED, never clamped), reading each core's live fire vector instead of
    the packed ``fires`` slab.
    """
    for sp in output_spans:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            if cycle < T:
                carry[cycle, :, d0:d1] = 1.0
            continue
        if sp.kind == "input":
            if cycle < T:
                carry[cycle, :, d0:d1] = (
                    input_spikes[:, int(sp.src_start):int(sp.src_end)])
            continue
        latency = cores[int(sp.src_core)].latency
        if latency is None:
            continue
        local = cycle - int(latency)
        if not (0 <= local < T):
            continue
        carry[local, :, d0:d1] = (
            buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)])


def require_carry_capable(stage, *, packed: bool) -> None:
    """Refuse an execution path that cannot record the raster it owes."""
    if packed:
        return
    raise NotImplementedError(
        f"stage {stage.name!r} must carry a spike raster to a later pass of its "
        f"segment, but this execution path (synchronized / recording / single-spike) "
        f"records none. Carrying is implemented on the packed cycle executor; a path "
        f"that cannot carry must refuse rather than hand the next pass a re-encoded "
        f"count."
    )


def carry_plan_for(output_spans, packed, cores, T: int):
    """(kind, dst slice, source slice, producer latency) per output span.

    The producer latency is per SPAN, not per segment: each output source starts
    emitting at its own core's latency, and producer-local time is what the
    consumer pass replays.
    """
    plan = []
    for sp in output_spans:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind in ("on", "input"):
            plan.append((sp.kind, d0, d1, int(sp.src_start), int(sp.src_end), 0))
            continue
        offset = packed.neuron_offset.get(int(sp.src_core))
        if offset is None:
            continue
        latency = int(cores[int(sp.src_core)].latency or 0)
        plan.append((
            "core", d0, d1,
            offset + int(sp.src_start), offset + int(sp.src_end), int(latency),
        ))
    return plan


def record_carry(carry, plan, *, cycle: int, fires, train, T: int) -> None:
    """Write this cycle's emissions into producer-local time."""
    for kind, d0, d1, s0, s1, latency in plan:
        local = cycle - latency
        if not (0 <= local < T):
            continue
        if kind == "on":
            carry[local, :, d0:d1] = 1.0
        elif kind == "input":
            carry[local, :, d0:d1] = train[local][:, s0:s1]
        else:
            carry[local, :, d0:d1] = fires[:, s0:s1]
