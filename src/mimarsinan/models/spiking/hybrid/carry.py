"""The raster carry: what a pass boundary INSIDE a neural segment replays.

A pass is a physical unit (one chip program); a segment is a semantic one. Under
streamed semantics an intra-segment pass boundary must hand the next pass the spike
RASTER, not a count — collapsing there would be the windowed transcode wearing a
scheduling hat, and the atol=0 parity gate forbids it. A wire read by a later SEGMENT
is a host boundary and still collapses, by design.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.spiking.hybrid.executors.carry_record import (  # noqa: F401 (re-exported)
    carry_plan_for, record_carry, record_reference_carry, require_carry_capable)
from mimarsinan.mapping.packing.executed_reads import COUNT, REPLAYED, slice_deliveries
from mimarsinan.mapping.packing.pass_timing import late_stage_input_readers, replay_delays
from mimarsinan.mapping.support.schedule.pass_carry import (
    carried_outputs_by_stage,
    carried_wire_bytes,
)
from mimarsinan.chip_simulation.activation_semantics import effective_legacy_spiking_mode, is_streamed_lif
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import execution_neural_stages
from mimarsinan.chip_simulation.spiking_semantics import is_cycle_based
from mimarsinan.chip_simulation.stage_input_clock import (  # noqa: F401 (re-exported)
    IN_TREE_CORE_LOCAL_BACKENDS, core_local_replay_backends, stage_input_read)
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


class PassReplayRefusal(ToolchainRefusal, ValueError):
    """A multi-pass run that cannot replay its carried rasters exactly, refused by name."""


def carry_scope(store: Dict[Any, Any], stage, *, publish: bool = False) -> Dict[int, Any]:
    """[U1 R3] The carried rasters of ``stage``'s own schedule segment: a later segment reads a carried producer
    through its count, as the fused program does."""
    segment = getattr(stage, "schedule_segment_index", None)
    return store.setdefault(segment, {}) if publish else store.get(segment, {})


def replay_raster(raster, delay: int, *, time_axis: int):
    """[U1 R2] ``raster`` read ``delay`` (<= 0) cycles late, as the fused segment delivers it: |delay| zero cycles
    first, its last |delay| dropped (numpy or torch, any layout)."""
    if not delay:
        return raster
    if delay > 0:
        raise PassReplayRefusal(f"replay delay {delay} > 0: hold-last replay is unexercised and refused")
    n = int(raster.shape[time_axis])
    shift = min(-int(delay), n)
    lead = (slice(None),) * (time_axis % raster.ndim)
    out = raster * 0
    out[lead + (slice(shift, n),)] = raster[lead + (slice(0, n - shift),)]
    return out


def replay_carried_slices(train, stage, carried: Dict[int, Any], *, reads_counts: bool = False) -> None:
    """Every carried slice of a ``(T, B, size)`` input train replayed in place from its producer's raster, every
    delayed slice shifted (a count's single-spike re-encode under COLLAPSE is the lossless codec), each as
    ``slice_deliveries`` (the static check's own reading) delivers it. A reader of window counts (a retimed level
    stage, or the synchronized count executor: ``reads_counts``) reads each slice's count as the fused segment hands
    it, so no raster and no delivery delay, a shift in time, applies to it."""
    for d in slice_deliveries(stage, carried, count_executor=reads_counts):
        if d.kind == COUNT or (d.kind != REPLAYED and not d.delay):
            continue
        s = stage.input_map[d.index]
        raster = carried[d.node] if d.kind == REPLAYED else None
        source = train[:, :, s.offset:s.offset + s.size] if raster is None else raster[:, :, :s.size].to(train.dtype)
        train[:, :, s.offset:s.offset + s.size] = replay_raster(source, d.delay, time_axis=0)


def shift_delayed_slices(encoded, stage, T: int, skip=()) -> None:
    """Every delayed slice of a backend's ``(N, size, T')`` input train (but the nodes in ``skip``) shifted in place
    within its first ``T`` cycles: a count re-encode read late (the single-spike codec under cascaded TTFS)."""
    for k, delay in replay_delays(stage).items():
        s = stage.input_map[k]
        if int(s.node_id) in skip:
            continue
        window = encoded[:, s.offset:s.offset + s.size, :T]
        encoded[:, s.offset:s.offset + s.size, :T] = replay_raster(window, delay, time_axis=2)


def replay_input_slices(encoded, stage, scope, T: int) -> None:
    """[U1 R2] A backend's ``(N, size, T')`` input train as the HCM's ``replay_carried_slices`` builds it: each slice
    whose producer published a raster in ``scope`` replayed, every other delayed slice's re-encode read late."""
    apply_carried_input(encoded, stage, scope)
    shift_delayed_slices(encoded, stage, T, skip=set(scope))


def shift_rows_to_core_time(window, rows, latency: int, T: int):
    """``window`` ``(size, N, T')`` with each given row's first ``T`` cycles moved ``latency`` late, so a core that
    starts at ``latency`` and reads at the global cycle sees them in its local time (a copy; ``window`` when idle)."""
    if latency <= 0 or not rows:
        return window
    out = window.copy()
    for lo, hi in rows:
        out[lo:hi, :, :] = 0
        out[lo:hi, :, latency:latency + T] = window[lo:hi, :, :T]
    return out


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
    replayed, delays = False, replay_delays(stage)
    for k, s in enumerate(stage.input_map):
        train = state_buffer_spikes.get(int(s.node_id))
        if train is None:
            continue
        train = replay_raster(train, delays.get(k, 0), time_axis=-2)
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


#: Backends that replay a carried raster verbatim; a streamed run refuses any other by name.
VERBATIM_BACKENDS = frozenset({"hcm", "sanafe", "nevresim", "lava"})


#: config key that enables each backend, so the run's discipline is read from the
#: same declaration the pipeline runs on.
_BACKEND_ENABLE_KEYS = {
    "sanafe": "enable_sanafe_simulation",
    "nevresim": "enable_nevresim_simulation",
    "lava": "enable_loihi_simulation",
}


def run_pass_transfer(config, hybrid_mapping=None) -> str:
    """ONE discipline per run: semantics first, then every enabled backend.

    Only STREAMED execution has a rhythm to carry: the windowed disciplines re-encode at every boundary by
    definition and mvm has no spikes, so their pass boundaries collapse whatever the backends. A streamed run carries
    rasters on every backend, or the cross-backend exactness gates would compare two computations (measured on a
    3-pass streamed MLP: HCM carrying while nevresim collapsed diverged on 4.4% of neuron windows); collapsing it
    changes the function, so an enabled backend that cannot replay is refused by name [U1 W6]. With the built
    ``hybrid_mapping``, the program's replay refusals run here, at plan time (``require_replayable_program``).
    """
    transfer = COLLAPSE if transfer_for(streamed=is_streamed_lif(config)) == COLLAPSE else VERBATIM
    weak = [backend for backend, key in _BACKEND_ENABLE_KEYS.items()
            if config.get(key) and pass_transfer_for_backend(backend) == COLLAPSE]
    if weak and transfer == VERBATIM:
        raise PassReplayRefusal(
            f"a streamed run carries spike rasters across pass boundaries, and enabled backend(s) {weak} cannot "
            f"replay them; collapsing to counts would change the function (disable them or run unscheduled)")
    if hybrid_mapping is not None:
        require_replayable_program(hybrid_mapping, config, transfer)
    return transfer


def pass_transfer_for_backend(backend: str) -> str:
    """Which pass-boundary discipline ``backend`` can execute.

    ``VERBATIM`` buffers the spike raster (``ceil(T/8)`` bytes per wire) and reproduces the fused execution
    bit-for-bit; ``COLLAPSE`` buffers window counts (``ceil(log2(T+1)/8)`` bytes per wire) and re-emits an even
    train, the decode/re-encode a host boundary performs. The record carries the discipline a run executed.
    """
    return VERBATIM if backend in VERBATIM_BACKENDS else COLLAPSE


def stochastic_replay_conflicts(hybrid_mapping, stage) -> list:
    """The input nodes of ``stage`` a verbatim replay under stochastic spike generation cannot serve: when it reads a
    carried raster, every slice it does not replay (the chip's RNG cannot be reproduced host-side)."""
    carried = {n for i, ids in carried_output_ids(hybrid_mapping).items()
               if hybrid_mapping.stages[i].schedule_segment_index == getattr(stage, "schedule_segment_index", None)
               for n in ids}
    nodes = [int(s.node_id) for s in stage.input_map]
    return [n for n in nodes if n not in carried] if carried & set(nodes) else []


def refuse_stochastic_replay(hybrid_mapping, *, spike_mode: str, pass_transfer: str) -> None:
    """[U1] Plan-time refusal by name of stochastic spike generation on a verbatim multi-pass run."""
    if spike_mode != "Stochastic" or pass_transfer != VERBATIM:
        return
    for stage in hybrid_mapping.stages:
        missing = stochastic_replay_conflicts(hybrid_mapping, stage) if stage.kind == "neural" else []
        if missing:
            raise PassReplayRefusal(
                f"segment {stage.name!r}: verbatim replay under Stochastic spike generation needs every input slice "
                f"carried, but nodes {missing} are host boundaries; the chip's RNG cannot be reproduced host-side")


class ChipClockReplayRefusal(PassReplayRefusal):
    """A program a backend cannot run in its readers' local time, refused by name."""


def require_backend_replay(hybrid_mapping, backend: str, *, transfer: str, spike_mode: str, spiking_mode: str,
                           stages=None) -> None:
    """[U1, W33] Refusals by name of a program ``backend`` cannot run exactly (every neural stage, or the given
    ``stages``): Stochastic generation under VERBATIM, and under a cycle-timed mode (an analytical TTFS value is read
    whenever its core runs), on a backend that does not read in core-local time on this checkout (``stage_input_read``:
    an in-tree read, or a probe of the source tree the backend runs, asked only here), any stage-input row, carried or
    host-encoded, read above latency 0 in a unit the backend executes, and any delayed slice."""
    refuse_stochastic_replay(hybrid_mapping, spike_mode=spike_mode, pass_transfer=transfer)
    if backend in IN_TREE_CORE_LOCAL_BACKENDS or not is_cycle_based(spiking_mode):
        return
    for stage in hybrid_mapping.stages if stages is None else stages:
        if stage.kind != "neural":
            continue
        late = executed_late_input_readers(stage)
        delayed = [k for unit in execution_neural_stages(stage) if not getattr(unit, "is_retimed_level", False)
                   for k in replay_delays(unit)]
        if not (late or delayed):
            continue
        read = stage_input_read(backend)
        if read.core_local:
            return
        if late:
            raise ChipClockReplayRefusal(
                f"{backend} reads a stage input at the global cycle ({read.evidence}), and segment {stage.name!r} has "
                f"stage-input rows read by cores that start above latency 0 ((core, latency) {late[:4]}); each would "
                f"read its row that many cycles ahead of its local time. Run without {backend}, or on a backend that "
                f"reads in core-local time")
        raise ChipClockReplayRefusal(
            f"{backend} cannot read segment {stage.name!r}'s delayed slices {sorted(delayed)} late under "
            f"{spiking_mode!r} ({read.evidence})")


def executed_late_input_readers(stage) -> list:
    """``(core, latency)`` per core that reads a stage-input row above latency 0 in the units a backend EXECUTES for
    ``stage`` (``execution_neural_stages``): the pass itself, or under per-hop re-timing its level units, whose cores
    all start at latency 0."""
    return [late for unit in execution_neural_stages(stage) for late in late_stage_input_readers(unit)]


def require_replayable_program(hybrid_mapping, config, transfer: str) -> None:
    """[U1] The plan-time refusals of ``require_backend_replay`` for the HCM and every backend ``config`` enables."""
    for backend in ["hcm"] + [b for b, key in _BACKEND_ENABLE_KEYS.items() if config.get(key)]:
        require_backend_replay(hybrid_mapping, backend, transfer=transfer,
                               spike_mode=str(config.get("spike_generation_mode", "")),
                               spiking_mode=effective_legacy_spiking_mode(config))


#: Re-exported so a backend reads the discipline and its price from one import.
__all__ = [
    "IN_TREE_CORE_LOCAL_BACKENDS", "ChipClockReplayRefusal", "VERBATIM_BACKENDS", "core_local_replay_backends",
    "PassReplayRefusal", "apply_carried_input", "carried_output_ids", "carried_wire_bytes", "carry_plan_for",
    "carry_scope", "pass_transfer_for_backend", "publish_carried_trains", "record_carry", "record_reference_carry",
    "refuse_stochastic_replay", "replay_carried_slices", "replay_input_slices", "replay_raster",
    "executed_late_input_readers", "require_backend_replay", "require_carry_capable", "require_replayable_program",
    "run_pass_transfer",
    "shift_delayed_slices", "shift_rows_to_core_time", "stage_input_read", "stochastic_replay_conflicts",
]
