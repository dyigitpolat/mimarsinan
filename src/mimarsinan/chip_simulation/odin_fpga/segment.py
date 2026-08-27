"""One neural segment on the device: export, inject, read the counts back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_numpy,
    apply_input_shifts_numpy,
)
from mimarsinan.chip_simulation.odin_fpga.records import OdinSegmentTiming
from mimarsinan.chip_simulation.odin_rtl.reference import (
    per_slot_counts_by_cycle,
    simulate_cycles,
)
from mimarsinan.chip_simulation.recording.records import (
    CoreSpikeCounts,
    SegmentSpikeRecord,
)
from mimarsinan.mapping.export.odin.exporter import export_odin
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.spiking.segment_boundary import normalize_boundary_slices_numpy

#: float64 matches HCM; float32 drifts +-1 spike at rate-encoding boundaries.
COMPUTE_DTYPE = np.dtype(np.float64)


@dataclass
class SegmentOutcome:
    """Everything one device-executed segment produced."""

    record: SegmentSpikeRecord
    timing: OdinSegmentTiming
    seg_output_counts: np.ndarray
    counts: Dict[Tuple[int, int, int, int], int]
    window_counts: Tuple[Tuple[int, ...], ...]
    latencies: Tuple[int, ...]
    cycles: int


@dataclass(frozen=True)
class SegmentPlan:
    """One segment resolved up to the device seam: entry raster, twin, export.

    Everything here is derived from the MAPPING alone, so a device run and a
    frozen deployment bundle start from the same object rather than from two
    copies of the same twelve lines.
    """

    hcm: Any
    chip_latency: int
    entry_rates: np.ndarray
    encoded: np.ndarray
    raster: List[List[int]]
    trace: Any
    export: Any
    timesteps: int

    @property
    def neurons(self) -> List[int]:
        return [int(core.neurons_per_core) for core in self.hcm.cores]

    @property
    def used_neurons(self) -> List[int]:
        return [
            max(int(core.neurons_per_core) - int(core.available_neurons or 0), 1)
            for core in self.hcm.cores
        ]

    def injection_plan(self) -> List[List[Dict[int, Tuple[int, ...]]]]:
        """The one-sample per-cycle injection plan a transport is handed."""
        return [list(per_slot_counts_by_cycle(self.trace))]

    def twin_counts(self) -> Dict[Tuple[int, int, int, int], int]:
        """The cycle-accurate twin's own answers, in the transport's key shape."""
        return {
            (0, cycle, core, neuron): int(value)
            for cycle, per_core in enumerate(self.trace.outputs)
            for core, counts in enumerate(per_core)
            for neuron, value in enumerate(counts) if value
        }


def plan_odin_segment(
    *,
    stage: Any,
    state_buffer,
    soma_law,
    behavior,
    timesteps: int,
    weight_bits: int,
    effective_max_axons: int,
    membrane_init: int,
    weight_sign_granularity: str,
    wire_divisors,
    node_shifts,
) -> SegmentPlan:
    """Encode the entry raster, run the cycle twin, and export the segment."""
    hcm = stage.hard_core_mapping
    if hcm is None:
        raise ValueError(
            f"stage {stage.name!r} is neural but carries no hard-core mapping; "
            f"there is nothing to program onto the device")
    chip_latency = int(ChipLatency(hcm).calculate())
    entry_rates = assemble_entry_rates(stage, state_buffer, wire_divisors, node_shifts)
    encoded = behavior.encode_segment_input(entry_rates, int(timesteps))
    raster = [
        [int(value) for value in encoded[0, :, cycle]]
        for cycle in range(int(timesteps))
    ]
    trace = simulate_cycles(
        hcm, soma_law=soma_law, input_counts=raster,
        simulation_length=int(timesteps), membrane_init=int(membrane_init),
        chip_latency=chip_latency)
    export = export_odin(
        hcm, soma_law=soma_law, weight_bits=int(weight_bits),
        weight_sign_granularity=str(weight_sign_granularity),
        effective_max_axons=int(effective_max_axons),
        membrane_init=int(membrane_init))
    return SegmentPlan(
        hcm=hcm, chip_latency=chip_latency, entry_rates=entry_rates,
        encoded=encoded, raster=raster, trace=trace, export=export,
        timesteps=int(timesteps))


def segment_record(
    plan: SegmentPlan, *, stage: Any, stage_index: int,
    windows: Sequence[Sequence[int]], counts: Dict[Tuple[int, int, int, int], int],
) -> Tuple[SegmentSpikeRecord, Tuple[Tuple[int, ...], ...], np.ndarray]:
    """The HCM-comparable record of one segment, from whatever answered it."""
    used = plan.used_neurons
    per_core = tuple(
        tuple(int(v) for v in row[:used[index]])
        for index, row in enumerate(windows))
    seg_output = gather_output_counts(
        getattr(plan.hcm, "output_sources", None), stage.output_map, per_core,
        timesteps=int(plan.timesteps))
    record = SegmentSpikeRecord(
        stage_index=int(stage_index),
        stage_name=str(stage.name),
        schedule_segment_index=stage.schedule_segment_index,
        schedule_pass_index=stage.schedule_pass_index,
        seg_input_rates=np.asarray(
            plan.entry_rates, dtype=np.float32).reshape(1, -1),
        seg_input_spike_count=np.asarray(
            plan.encoded[0].sum(axis=1), dtype=np.int64),
        seg_output_spike_count=seg_output,
        cores=_core_records(plan.hcm, plan.trace, per_core, counts),
    )
    return record, per_core, seg_output


def gather_output_counts(
    output_sources: Any,
    output_map: Sequence[Any],
    per_core_counts: Sequence[Sequence[int]],
    *,
    timesteps: int,
) -> np.ndarray:
    """A segment's output-count vector, gathered by its own source spans.

    The same span walk every backend's segment gather runs: an always-on wire
    contributes ``T``, an entry wire is not a segment output, and a core wire
    takes the producing core's per-neuron counts.
    """
    flat = (
        list(output_sources.flatten())
        if output_sources is not None and hasattr(output_sources, "flatten")
        else list(output_sources or [])
    )
    if not flat:
        total = max((int(s.offset) + int(s.size) for s in output_map), default=0)
        return np.zeros(total, dtype=np.int64)
    out = np.zeros(len(flat), dtype=np.int64)
    for span in compress_spike_sources(flat):
        d0, d1 = int(span.dst_start), int(span.dst_end)
        if span.kind == "off" or span.kind == "input":
            continue
        if span.kind == "on":
            out[d0:d1] = int(timesteps)
            continue
        source = per_core_counts[int(span.src_core)]
        s0 = int(span.src_start)
        take = min(int(span.length), max(len(source) - s0, 0))
        if take > 0:
            out[d0:d0 + take] = np.asarray(source[s0:s0 + take], dtype=np.int64)
    return out


def assemble_entry_rates(stage: Any, state_buffer, wire_divisors, node_shifts):
    """The segment's entry rates in the WIRE domain, clamped to ``[0, 1]``."""
    rates = assemble_segment_input_numpy(
        stage.input_map, state_buffer, num_samples=1, dtype=COMPUTE_DTYPE)
    rates = normalize_boundary_slices_numpy(stage.input_map, rates, wire_divisors)
    rates = apply_input_shifts_numpy(stage.input_map, rates, node_shifts)
    return np.clip(rates, 0.0, 1.0)


def run_odin_segment(
    *,
    stage: Any,
    stage_index: int,
    state_buffer,
    transport,
    soma_law,
    behavior,
    timesteps: int,
    weight_bits: int,
    effective_max_axons: int,
    membrane_init: int,
    weight_sign_granularity: str,
    wire_divisors,
    node_shifts,
) -> SegmentOutcome:
    """Program one segment onto the device, run it, and record what it emitted."""
    plan = plan_odin_segment(
        stage=stage, state_buffer=state_buffer, soma_law=soma_law,
        behavior=behavior, timesteps=timesteps, weight_bits=weight_bits,
        effective_max_axons=effective_max_axons, membrane_init=membrane_init,
        weight_sign_granularity=weight_sign_granularity,
        wire_divisors=wire_divisors, node_shifts=node_shifts)
    trace = plan.trace

    receipt = transport.program(plan.export)
    run = transport.run_samples(plan.injection_plan(), latencies=trace.latencies)

    # Each core contributes only inside its OWN window [latency, latency + T):
    # nevresim's SpikeCountRecorder convention, which is also the set of cycles
    # the HCM reference steps a core over, so the two records are comparable
    # term by term. Cycles outside it are drain, not deployment.
    windows = run.window_counts(
        latencies=trace.latencies, simulation_length=int(timesteps),
        neurons=plan.neurons)[0]
    record, per_core, seg_output = segment_record(
        plan, stage=stage, stage_index=stage_index, windows=windows,
        counts=run.counts)
    timing = OdinSegmentTiming(
        stage_index=int(stage_index), stage_name=str(stage.name),
        cores=int(receipt.cores), program_ops=int(receipt.ops),
        program_bytes=len(receipt.payload),
        program_wall_s=float(receipt.wall_s), program_basis=str(receipt.basis),
        run_wall_s=float(run.wall_s), device_cycles=int(run.device_cycles),
        samples=int(run.samples), cycles_per_sample=int(run.cycles_per_sample),
        detail=dict(run.detail),
    )
    return SegmentOutcome(
        record=record, timing=timing, seg_output_counts=seg_output,
        counts=dict(run.counts), window_counts=tuple(per_core),
        latencies=tuple(int(v) for v in trace.latencies),
        cycles=int(run.cycles_per_sample),
    )


def _core_records(hcm: Any, trace, per_core, counts) -> List[CoreSpikeCounts]:
    """Per-core input/output counts in the shape the HCM reference records."""
    records: List[CoreSpikeCounts] = []
    for index, core in enumerate(hcm.cores):
        n_in = max(int(core.axons_per_core) - int(core.available_axons or 0), 1)
        latency = int(trace.latencies[index])
        arrivals = np.zeros(n_in, dtype=np.int64)
        for cycle in range(latency, min(latency + trace.simulation_length,
                                        len(trace.inputs))):
            arrivals += np.asarray(
                trace.inputs[cycle][index][:n_in], dtype=np.int64)
        width = len(per_core[index])
        raster = np.zeros((trace.simulation_length, width), dtype=np.int64)
        for cycle in range(len(trace.outputs)):
            local = cycle - latency
            if 0 <= local < trace.simulation_length:
                raster[local] = [
                    counts.get((0, cycle, index, neuron), 0)
                    for neuron in range(width)
                ]
        records.append(CoreSpikeCounts(
            core_index=index,
            n_in_used=n_in,
            n_out_used=len(per_core[index]),
            core_latency=latency,
            has_hardware_bias=getattr(core, "hardware_bias", None) is not None,
            n_always_on_axons=sum(
                1 for source in core.axon_sources
                if getattr(source, "is_always_on_", False)),
            input_spike_count=arrivals,
            output_spike_count=np.asarray(per_core[index], dtype=np.int64),
            output_spike_raster=raster,
        ))
    return records
