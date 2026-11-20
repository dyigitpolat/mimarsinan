"""[U-LIF2 W9] The gated reference loop of integrate then emit (per core, the recording path), and the one dispatch of an
execution stage that integrates then emits.

The reference loop runs the gate of ``gated_window`` core by core on the per-core charge layout (``integrate_step``,
the per-cycle loop's ``step`` with the fire gate closed): every integrating core's axons are filled from the previous
cycle's fires, then each integrates (its bias on its own cycles only) or emits. It records what it executes: each
core's arrivals over its integration (its window's input counts) and its emitted count, so a recorded run (the event
census's, the parity gates') reads the executor of record.

``run_integrate_then_emit`` is the only caller of the closed form (``run_neural_segment_counts``), reached by its name
(``window_count_closed_form``: the certification's analytic gauge) and never for a record; otherwise the packed gated
executor runs, and the reference loop where a record is asked for or the packed executor is switched off."""

from __future__ import annotations

from typing import Optional

import torch

from mimarsinan.models.spiking.cycle_policy import precharge_lif_states
from mimarsinan.models.spiking.hybrid.executors.gated_window import (
    gated_window_plan,
    require_gated_window_admissible,
    run_neural_segment_gated,
)
from mimarsinan.models.spiking.hybrid.executors.reference_loop import (
    allocate_record_tensors,
    append_core_spike_counts,
)
from mimarsinan.models.spiking.hybrid.executors.sync_counts import run_neural_segment_counts, runs_closed_form
from mimarsinan.models.spiking.hybrid.membrane_readout import stash_membrane_readout_correction
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


def run_integrate_then_emit(flow, input_spike_train, *, seg, stage, T, batch_size, device, policy,
                            recorder_seg=None, readout_corrections=None) -> torch.Tensor:
    """One execution stage of a flow that integrates then emits, on its executor of record."""
    common = dict(seg=seg, T=T, batch_size=batch_size, device=device)
    if recorder_seg is None and runs_closed_form(flow, recording=False):
        return run_neural_segment_counts(flow, input_spike_train, soma_law=flow.soma_law, **common)
    if recorder_seg is None and getattr(flow, "use_packed_cycle_executor", True):
        return run_neural_segment_gated(flow, input_spike_train, stage=stage, policy=policy,
                                        readout_corrections=readout_corrections, **common)
    return run_neural_segment_gated_reference(flow, input_spike_train, stage=stage, policy=policy,
                                              recorder_seg=recorder_seg, readout_corrections=readout_corrections,
                                              **common)


def run_neural_segment_gated_reference(
    flow, input_spike_train, *, seg, stage, T, batch_size, device, policy, recorder_seg=None,
    readout_corrections=None, neuron_count_recorder: Optional[dict] = None,
) -> torch.Tensor:
    """The gate, core by core; returns the segment's output counts ``(B, out)`` and, given ``recorder_seg`` (batch 1),
    appends every core's recorded arrivals and emitted counts."""
    require_gated_window_admissible(getattr(flow, "soma_law", None), getattr(flow, "firing_mode", None))
    T = int(T)
    plan = gated_window_plan(seg, T)
    cores, plans = seg["cores"], seg["axon_fill_plans"]
    weights, thresholds, biases = seg["core_params"], seg["thresholds"], seg["hw_biases"]
    width_out = [max(int(c.neurons_per_core - c.available_neurons), 1) for c in cores]
    width_in = [max(int(c.axons_per_core - c.available_axons), 1) for c in cores]
    states = [policy.make_state(batch_size, n, device, COMPUTE_DTYPE) for n in width_out]
    precharge_lif_states(states, thresholds, float(getattr(flow, "lif_membrane_init", 0.0)))
    buffers = [torch.zeros(batch_size, n, device=device, dtype=COMPUTE_DTYPE) for n in width_out]
    signals = [torch.zeros(batch_size, n, device=device, dtype=COMPUTE_DTYPE) for n in width_in]
    counts = [torch.zeros(batch_size, n, device=device, dtype=COMPUTE_DTYPE) for n in width_out]
    record_in, record_out = allocate_record_tensors(cores, device) if recorder_seg is not None else (None, None)
    train = input_spike_train.to(COMPUTE_DTYPE)
    zeros_in = torch.zeros(batch_size, train.shape[2], device=device, dtype=COMPUTE_DTYPE)
    stepped = [i for i, s in enumerate(plan.starts) if s is not None]
    starts = {i: int(plan.starts[i]) for i in stepped}  # type: ignore[arg-type]

    for cycle in range(plan.cycles):
        integrating = [i for i in stepped if cycle < starts[i] + T]
        for i in integrating:  # every fill reads the previous cycle's fires, before any core steps
            own = cycle >= starts[i]
            plans[i].apply(signals[i], input_spikes=train[cycle - starts[i]] if own else zeros_in, buffers=buffers,
                           on_value=1.0 if own else 0.0)
        for i in integrating:
            policy.integrate_step(states[i], weights[i], signals[i],
                                  hw_bias=biases[i] if cycle >= starts[i] else None)
            if record_in is not None:
                record_in[i] += signals[i][0].to(torch.int64)
        for i in stepped:
            if starts[i] + T <= cycle < starts[i] + 2 * T:
                fired = policy.emit(states[i], thresholds[i], thresholding_mode=flow.thresholding_mode,
                                    output_dtype=COMPUTE_DTYPE)
                buffers[i] = fired
                counts[i] += fired
                if record_out is not None:
                    record_out[i] += fired[0].to(torch.int64).detach()
            elif cycle == starts[i] + 2 * T:
                buffers[i] = torch.zeros_like(buffers[i])

    if neuron_count_recorder is not None:
        for i in stepped:
            neuron_count_recorder[i] = counts[i].detach()
    output_counts = torch.zeros(batch_size, len(seg["output_sources"]), device=device, dtype=COMPUTE_DTYPE)
    input_total = train[:T].sum(dim=0)
    for sp in seg["output_spans"]:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "on":
            output_counts[:, d0:d1] = float(T)
        elif sp.kind == "input":
            output_counts[:, d0:d1] = input_total[:, int(sp.src_start):int(sp.src_end)]
        elif sp.kind != "off" and int(sp.src_core) in starts:
            output_counts[:, d0:d1] = counts[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)]
    stash_membrane_readout_correction(
        flow, seg=seg, stage=stage, output_counts=output_counts, output_spans=seg["output_spans"],
        neuron_states=states, thresholds=thresholds, single_spike=False, readout_corrections=readout_corrections)
    if recorder_seg is not None:
        assert record_in is not None and record_out is not None
        append_core_spike_counts(recorder_seg, cores, axon_spans=seg["axon_spans"], record_in_t=record_in,
                                 record_out_t=record_out)
    return output_counts
