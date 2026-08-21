"""Hybrid mapping rate-coded LIF segment execution."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.chip_simulation.recording.spike_recorder import SegmentSpikeRecord
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.models.spiking.hybrid.carry import (
    record_reference_carry, require_carry_capable)
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy, precharge_lif_states
from mimarsinan.models.spiking.hybrid.executors import (
    run_neural_segment_counts, run_neural_segment_packed,)
from mimarsinan.models.spiking.hybrid.executors.reference_loop import (
    accumulate_output_spans, allocate_record_tensors, append_core_spike_counts,
    build_cycle_activity_plan, fill_core_inputs)
from mimarsinan.models.spiking.hybrid.executors.single_spike import (
    single_spike_output_step)
from mimarsinan.models.spiking.hybrid.host import HybridFlowHost
from mimarsinan.models.spiking.hybrid.membrane_readout import stash_membrane_readout_correction
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridLifStepMixin(HybridFlowHost):
    """Rate-coded neural segments and LIF hybrid forward."""

    def _run_neural_segment_rate(
        self,
        stage: HybridStage,
        *,
        input_spike_train: torch.Tensor,
        recorder_seg: SegmentSpikeRecord | None = None,
        readout_corrections: Dict[int, torch.Tensor] | None = None,
        output_train: list | None = None,
    ) -> torch.Tensor:
        """Rate-coded segment: returns raw clamp spike counts ``(B, out_dim)`` —
        the currency parity gathers read. The [C2] membrane term goes into
        ``readout_corrections`` (never the counts) for the logits decode."""
        mapping = stage.hard_core_mapping
        assert mapping is not None

        T = self.simulation_length
        assert input_spike_train.shape[0] == T

        batch_size = input_spike_train.shape[1]
        device = input_spike_train.device
        input_size = input_spike_train.shape[2]
        recording = recorder_seg is not None
        if recording:
            assert batch_size == 1, "Spike recording requires batch_size == 1"

        seg = self._get_segment_tensors(stage, device)
        # Memoized with the segment tensors: calculate() is idempotent (fixed
        # point over core latencies) and the mapping does not change per forward.
        latency = seg.get("latency")
        if latency is None:
            latency = int(ChipLatency(mapping).calculate())
            seg["latency"] = latency
        cycles = int(latency) + T
        cores = seg["cores"]
        output_sources = seg["output_sources"]
        axon_spans = seg["axon_spans"]
        axon_fill_plans = seg["axon_fill_plans"]
        output_spans = seg["output_spans"]
        core_params = seg["core_params"]
        thresholds = seg["thresholds"]
        hw_biases = seg["hw_biases"]

        buffers = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=COMPUTE_DTYPE) for c in cores]
        policy = cycle_neuron_policy(
            self.spiking_mode, self.ttfs_cycle_schedule, self.firing_mode,
            integer_lattice=bool(
                getattr(self, "membrane_integer_lattice", False)
            ),
        )
        neuron_states = [
            policy.make_state(
                batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                device, COMPUTE_DTYPE) for c in cores]
        precharge_lif_states(neuron_states, thresholds, getattr(self, "lif_membrane_init", 0.0))

        output_counts = torch.zeros(batch_size, len(output_sources), device=device, dtype=COMPUTE_DTYPE)

        zeros_in = torch.zeros(batch_size, input_size, device=device, dtype=COMPUTE_DTYPE)
        input_signals = [
            torch.zeros(batch_size, max(int(c.axons_per_core - c.available_axons), 1),
                        device=device, dtype=COMPUTE_DTYPE) for c in cores]

        record_in_t, record_out_t = (
            allocate_record_tensors(cores, device) if recording else (None, None))

        input_spike_train = input_spike_train.to(COMPUTE_DTYPE)
        latency_gated = policy.latency_gated
        single_spike = getattr(policy, "single_spike_io", False)

        synchronized_path = (
            getattr(self, "lif_execution_synchronized", False)
            and self.spiking_mode == "lif" and not single_spike and not recording)
        if output_train is not None:
            # capable= must describe the EXECUTED path: the synchronized early-
            # return and the single-spike latch record no multi-spike raster;
            # the packed executor AND the reference loop below both do.
            require_carry_capable(
                stage, packed=(not synchronized_path and not single_spike))

        if synchronized_path:
            return run_neural_segment_counts(
                self, input_spike_train, seg=seg, T=T,
                batch_size=batch_size, device=device)

        # [cert-plan W1] stage-flat executor: same policy physics, batched
        # charge layout; recording/single-spike paths keep the per-core
        # reference loop below (byte-stable records, latch decode).
        if (not single_spike and not recording and latency_gated
                and getattr(self, "use_packed_cycle_executor", True)):
            return run_neural_segment_packed(
                self, input_spike_train, seg=seg, stage=stage, T=T,
                batch_size=batch_size, device=device, policy=policy,
                readout_corrections=readout_corrections,
                output_train=output_train)

        if single_spike:
            shifted = torch.zeros_like(input_spike_train)
            shifted[1:] = input_spike_train[:-1]
            input_spike_train = (input_spike_train - shifted).clamp_min_(0.0)

        out_arrival = (torch.zeros(batch_size, len(output_sources),
                                   device=device, dtype=COMPUTE_DTYPE)
                       if single_spike else None)

        reference_carry = None
        if output_train is not None and not single_spike:
            reference_carry = torch.zeros(
                T, batch_size, len(output_sources),
                device=device, dtype=COMPUTE_DTYPE)

        core_latencies = [int(c.latency or 0) for c in cores]
        stepable = [i for i, c in enumerate(cores) if c.latency is not None]
        active_by_cycle, fill_by_cycle = build_cycle_activity_plan(
            seg, cores=cores, stepable=stepable, cycles=cycles, T=T,
            latency_gated=latency_gated)

        for cycle in range(cycles):
            input_spikes = input_spike_train[cycle] if cycle < T else zeros_in

            fill_core_inputs(
                self, fill_by_cycle[cycle], input_signals=input_signals,
                input_spike_train=input_spike_train, zeros_in=zeros_in,
                buffers=buffers, plans=axon_fill_plans,
                core_latencies=core_latencies, cycle=cycle, T=T,
                single_spike=single_spike)

            for core_idx in active_by_cycle[cycle]:
                buffers[core_idx] = policy.step(
                    neuron_states[core_idx],
                    core_params[core_idx],
                    input_signals[core_idx],
                    thresholds[core_idx],
                    hw_bias=hw_biases[core_idx],
                    thresholding_mode=self.thresholding_mode,
                    output_dtype=COMPUTE_DTYPE,
                )

                if record_in_t is not None and record_out_t is not None:
                    record_in_t[core_idx] += input_signals[core_idx][0].to(torch.int64)
                    record_out_t[core_idx] += buffers[core_idx][0].to(torch.int64).detach()

            if single_spike:
                assert out_arrival is not None
                single_spike_output_step(
                    output_counts, out_arrival, output_spans, cores,
                    cycle=cycle, T=T, buffers=buffers, input_spikes=input_spikes)
                continue

            accumulate_output_spans(
                output_counts, output_spans, cores, cycle=cycle, T=T,
                buffers=buffers, input_spikes=input_spikes)

            if reference_carry is not None:
                record_reference_carry(
                    reference_carry, output_spans, cores, cycle=cycle,
                    buffers=buffers, input_spikes=input_spikes, T=T)

        if output_train is not None and not single_spike:
            output_train.append(reference_carry)

        stash_membrane_readout_correction(
            self,
            seg=seg,
            stage=stage,
            output_counts=output_counts,
            output_spans=output_spans,
            neuron_states=neuron_states,
            thresholds=thresholds,
            single_spike=single_spike,
            readout_corrections=readout_corrections,
        )

        if recorder_seg is not None:
            assert record_in_t is not None and record_out_t is not None
            append_core_spike_counts(
                recorder_seg, cores, axon_spans=axon_spans,
                record_in_t=record_in_t, record_out_t=record_out_t)

        return output_counts

    def _apply_input_shifts(self, input_map, seg_input_rates: torch.Tensor) -> torch.Tensor:
        """Add the Round-2a per-producer-channel positive shift before the [0,1] clamp.

        Keyed by producer ``node_id`` in ``hybrid_mapping.node_output_shifts`` (empty
        ⇒ identity); value-preserving because the consumer bias is pre-corrected (B'=B−W·s).
        """
        shifts = getattr(self.hybrid_mapping, "node_output_shifts", None)
        if not shifts:
            return seg_input_rates
        out = seg_input_rates
        cloned = False
        for s in input_map:
            shift = shifts.get(int(s.node_id))
            if shift is None:
                continue
            if not cloned:
                out = out.clone()
                cloned = True
            sh = torch.as_tensor(shift, dtype=out.dtype, device=out.device).reshape(-1)
            out[:, s.offset : s.offset + s.size] += sh[: s.size]
        return out
