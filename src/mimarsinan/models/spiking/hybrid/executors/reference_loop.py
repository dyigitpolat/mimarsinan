"""Per-cycle glue and recording helpers for the per-core reference segment loop."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.chip_simulation.recording.spike_recorder import CoreSpikeCounts


def allocate_record_tensors(
    cores, device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-core input/output spike-count accumulators of a recorded segment."""
    record_in_t = [
        torch.zeros(max(int(c.axons_per_core - c.available_axons), 1),
                    device=device, dtype=torch.int64) for c in cores]
    record_out_t = [
        torch.zeros(max(int(c.neurons_per_core - c.available_neurons), 1),
                    device=device, dtype=torch.int64) for c in cores]
    return record_in_t, record_out_t


def build_cycle_activity_plan(
    seg, *, cores, stepable, cycles, T, latency_gated
) -> tuple[list, list]:
    """The per-cycle (active, fill) core-index sets, memoized on the segment."""
    # A gated core only consumes its axon fill inside [latency, latency+T);
    # filling outside that window is dead work (input_signals feed nothing),
    # so both loops walk precomputed per-cycle active sets. Static per stage.
    if latency_gated:
        active_by_cycle = seg.get("active_by_cycle")
        if active_by_cycle is None or len(active_by_cycle) < cycles:
            active_by_cycle = [
                [i for i in stepable if cores[i].latency <= cycle < T + cores[i].latency]
                for cycle in range(cycles)
            ]
            seg["active_by_cycle"] = active_by_cycle
        fill_by_cycle = active_by_cycle
    else:
        # An ungated policy steps every core every cycle; fill everything.
        active_by_cycle = [stepable] * cycles
        fill_by_cycle = [list(range(len(cores)))] * cycles
    return active_by_cycle, fill_by_cycle


def fill_core_inputs(
    flow, core_indices, *, input_signals, input_spike_train, zeros_in, buffers,
    plans, core_latencies, cycle, T, single_spike,
) -> None:
    """Fill each listed core's axon signal tensor from its span-fill plan."""
    for core_idx in core_indices:
        local_cycle = cycle - core_latencies[core_idx]
        flow._fill_signal_tensor_from_spans(
            input_signals[core_idx],
            input_spikes=(
                input_spike_train[local_cycle]
                if 0 <= local_cycle < T
                else zeros_in
            ),
            buffers=buffers,
            plan=plans[core_idx],
            cycle=cycle,
            single_spike=single_spike,
            latency=core_latencies[core_idx],
        )


def accumulate_output_spans(
    output_counts, output_spans, cores, *, cycle, T, buffers, input_spikes,
) -> None:
    """Add this cycle's in-window producer fires to the segment output counts."""
    for sp in output_spans:
        d0 = int(sp.dst_start)
        d1 = int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            if cycle < T:
                output_counts[:, d0:d1] += 1.0
            continue
        if sp.kind == "input":
            if cycle < T:
                output_counts[:, d0:d1] += input_spikes[:, int(sp.src_start):int(sp.src_end)]
            continue
        src_lat = cores[int(sp.src_core)].latency
        if src_lat is None:
            continue
        if cycle < int(src_lat) or cycle >= int(src_lat) + T:
            continue
        output_counts[:, d0:d1] += buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)]


def append_core_spike_counts(
    recorder_seg, cores, *, axon_spans, record_in_t, record_out_t,
) -> None:
    """Append every core's recorded input/output spike counts to the record."""
    for core_idx, core in enumerate(cores):
        axon_span_list = axon_spans[core_idx]
        n_always_on = sum(
            int(sp.length) for sp in axon_span_list if sp.kind == "on"
        )
        recorder_seg.cores.append(
            CoreSpikeCounts(
                core_index=core_idx,
                n_in_used=max(int(core.axons_per_core - core.available_axons), 1),
                n_out_used=max(int(core.neurons_per_core - core.available_neurons), 1),
                core_latency=int(core.latency) if core.latency is not None else -1,
                has_hardware_bias=getattr(core, "hardware_bias", None) is not None,
                n_always_on_axons=n_always_on,
                input_spike_count=record_in_t[core_idx].cpu().numpy().astype(np.int64),
                output_spike_count=record_out_t[core_idx].cpu().numpy().astype(np.int64),
            )
        )
