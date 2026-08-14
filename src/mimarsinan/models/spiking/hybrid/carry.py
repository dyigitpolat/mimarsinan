"""The raster carry: what a pass boundary INSIDE a neural segment replays.

A pass is a physical unit (one chip program); a segment is a semantic one. Under
streamed semantics an intra-segment pass boundary must hand the next pass the spike
RASTER, not a count — collapsing there would be the windowed transcode wearing a
scheduling hat, and the atol=0 parity gate forbids it. A wire read by a later SEGMENT
is a host boundary and still collapses, by design.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from mimarsinan.mapping.support.schedule.pass_cut import (
    COLLAPSE,
    VERBATIM,
    carried_outputs_by_stage,
    raster_bytes,
)


def carried_output_ids(hybrid_mapping) -> Dict[int, Tuple[int, ...]]:
    """``{stage_index: node_ids a later pass of the same segment replays}``, memoized."""
    cached = getattr(hybrid_mapping, "_carried_output_cache", None)
    if cached is None:
        cached = carried_outputs_by_stage(hybrid_mapping.stages)
        setattr(hybrid_mapping, "_carried_output_cache", cached)
    return cached


def publish_carried_trains(
    stage,
    output_train: torch.Tensor,
    carried_ids: Tuple[int, ...],
    state_buffer_spikes: Dict[int, torch.Tensor],
) -> None:
    """Slice the segment's output raster into the per-wire trains a later pass replays.

    Only CARRIED wires are published: publishing a wire that crosses to a later
    SEGMENT would silently upgrade a host boundary that is defined to collapse.
    """
    wanted = set(carried_ids)
    for s in stage.output_map:
        node_id = int(s.node_id)
        if node_id in wanted:
            state_buffer_spikes[node_id] = (
                output_train[:, :, s.offset : s.offset + s.size]
            )


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


#: Backends that replay a carried raster verbatim. A backend outside this set uses
#: the COLLAPSE discipline instead — never a refusal.
VERBATIM_BACKENDS = frozenset({"hcm"})


def pass_transfer_for_backend(backend: str) -> str:
    """Which pass-boundary discipline ``backend`` will actually execute.

    Both are legitimate deployments of a scheduled segment, because the boundary is
    one this program INTRODUCES: a chip that reprograms between passes has to buffer
    the intermediate signal either way, and the choice is only WHAT it buffers.

    - ``VERBATIM`` buffers the spike raster (``ceil(T/8)`` bytes per wire) and
      reproduces the fused execution bit-for-bit.
    - ``COLLAPSE`` buffers window counts (``ceil(log2(T+1)/8)`` bytes per wire) and
      re-emits an even train, which normalizes rhythm — the same decode/re-encode a
      host boundary performs, applied at a boundary that now genuinely exists.

    A backend therefore never refuses a scheduled deployment; it reports which
    discipline it ran, and the record carries that so a reader knows which of the two
    computations produced the numbers.
    """
    return VERBATIM if backend in VERBATIM_BACKENDS else COLLAPSE


def carried_wire_bytes(width: int, timesteps: int, transfer: str) -> int:
    """Bytes one carried wire occupies under ``transfer`` — the cost of the choice."""
    if transfer == VERBATIM:
        return raster_bytes(width, timesteps)
    return int(width) * ((max(int(timesteps), 1).bit_length() + 7) // 8)
