"""The raster carry: what a pass boundary INSIDE a neural segment replays.

A pass is a physical unit (one chip program); a segment is a semantic one. Under
streamed semantics an intra-segment pass boundary must hand the next pass the spike
RASTER, not a count — collapsing there would be the windowed transcode wearing a
scheduling hat, and the atol=0 parity gate forbids it. A wire read by a later SEGMENT
is a host boundary and still collapses, by design.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from mimarsinan.mapping.support.schedule.pass_carry import (
    carried_outputs_by_stage,
    carried_wire_bytes,
)
from mimarsinan.chip_simulation.activation_semantics import is_streamed_lif
from mimarsinan.mapping.support.schedule.pass_cut import (
    COLLAPSE,
    VERBATIM,
    transfer_for,
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
    output_train,
    carried_ids: Tuple[int, ...],
    state_buffer_spikes: Dict[int, Any],
) -> None:
    """Slice the segment's output raster into the per-wire trains a later pass replays.

    Only CARRIED wires are published: publishing a wire that crosses to a later
    SEGMENT would silently upgrade a host boundary that is defined to collapse.
    """
    wanted = set(carried_ids)
    for s in stage.output_map:
        node_id = int(s.node_id)
        if node_id in wanted:
            # Ellipsis, not a fixed rank: the canonical carried layout is
            # TIME-FIRST with the feature axis LAST, so the torch flow's
            # (T, B, size) and a per-sample numpy (T, size) slice identically.
            state_buffer_spikes[node_id] = (
                output_train[..., s.offset : s.offset + s.size]
            )


def apply_carried_input(encoded, stage, state_buffer_spikes) -> bool:
    """Overwrite ``encoded`` slices whose producer published a raster.

    ``encoded`` is the backend's per-axon train in ``(N, size, T)`` — SANA-FE
    passes ``N == 1``, lava the whole batch — while a carried raster is stored
    time-first: ``(T, size)`` per-sample, or ``(N, T, size)`` batched. Returns
    whether anything was replayed, so a caller can report the discipline it
    actually ran rather than the one it hoped for.
    """
    replayed = False
    for s in stage.input_map:
        train = state_buffer_spikes.get(int(s.node_id))
        if train is None:
            continue
        width = min(int(s.size), train.shape[-1], encoded.shape[1] - int(s.offset))
        if width <= 0:
            continue
        window = min(int(train.shape[-2]), int(encoded.shape[2]))
        if train.ndim == 2:
            encoded[0, s.offset : s.offset + width, :window] = (
                train[:window, :width].T
            )
        else:
            for i in range(min(int(train.shape[0]), int(encoded.shape[0]))):
                encoded[i, s.offset : s.offset + width, :window] = (
                    train[i, :window, :width].T
                )
        replayed = True
    return replayed


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


#: Backends that replay a carried raster verbatim. A backend outside this set uses
#: the COLLAPSE discipline instead — never a refusal.
VERBATIM_BACKENDS = frozenset({"hcm", "sanafe", "nevresim", "lava"})


#: config key that enables each backend, so the run's discipline is read from the
#: same declaration the pipeline runs on.
_BACKEND_ENABLE_KEYS = {
    # [ODIN P7a] the physical device buffers no producer raster across a pass
    # boundary, so enabling it costs the run its verbatim boundaries — the
    # documented price, never a refusal.
    "odin_fpga": "enable_odin_fpga_simulation",
    "sanafe": "enable_sanafe_simulation",
    "nevresim": "enable_nevresim_simulation",
    "lava": "enable_loihi_simulation",
}


def run_pass_transfer(config) -> str:
    """ONE discipline per run: semantics first, then the weakest enabled backend.

    Semantics first — only STREAMED execution has a rhythm to carry. The windowed
    disciplines re-encode at every boundary by definition and mvm has no spikes at
    all, so their pass boundaries collapse regardless of which backends run; a
    "verbatim" record for such a run would name a computation that never happened,
    and a backend replaying a raw raster into a windowed pass would diverge from
    every peer that re-encoded.

    Then the weakest backend — a run whose backends disagreed would report the
    numbers of two different computations, and the cross-backend exactness gates,
    which admit no tolerance on integer arithmetic, would compare unlike things.
    Measured on a 3-pass streamed MLP: HCM carrying while nevresim collapsed
    diverged on 4.4% of neuron windows by one spike each — exactly the rhythm a
    collapse normalizes away. So enabling a backend that cannot replay a raster
    costs the whole run its verbatim boundaries, and teaching that backend to
    record one upgrades the run.
    """
    if transfer_for(streamed=is_streamed_lif(config)) == COLLAPSE:
        return COLLAPSE
    enabled = [
        backend for backend, key in _BACKEND_ENABLE_KEYS.items() if config.get(key)
    ]
    if any(pass_transfer_for_backend(b) == COLLAPSE for b in enabled):
        return COLLAPSE
    return VERBATIM


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


#: Re-exported so a backend reads the discipline and its price from one import.
__all__ = [
    "VERBATIM_BACKENDS",
    "carried_output_ids",
    "apply_carried_input",
    "carried_wire_bytes",
    "carry_plan_for",
    "pass_transfer_for_backend",
    "run_pass_transfer",
    "run_pass_transfer",
    "publish_carried_trains",
    "record_carry",
    "record_reference_carry",
    "require_carry_capable",
]
