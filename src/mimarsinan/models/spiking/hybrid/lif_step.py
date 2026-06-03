"""Hybrid mapping rate-coded LIF segment execution."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.chip_simulation.recording.spike_recorder import CoreSpikeCounts, SegmentSpikeRecord
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridLifStepMixin:
    """Rate-coded neural segments and LIF hybrid forward."""

    def _run_neural_segment_rate(
        self,
        stage: HybridStage,
        *,
        input_spike_train: torch.Tensor,
        recorder_seg: SegmentSpikeRecord | None = None,
    ) -> torch.Tensor:
        """Rate-coded segment: cycle loop over cores; returns spike counts ``(B, out_dim)``."""
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

        latency = ChipLatency(mapping).calculate()
        cycles = int(latency) + T

        seg = self._get_segment_tensors(stage, device)
        cores = seg["cores"]
        output_sources = seg["output_sources"]
        axon_spans = seg["axon_spans"]
        output_spans = seg["output_spans"]
        core_params = seg["core_params"]
        thresholds = seg["thresholds"]
        hw_biases = seg["hw_biases"]

        buffers = [
            torch.zeros(batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                        device=device, dtype=COMPUTE_DTYPE)
            for c in cores
        ]
        policy = cycle_neuron_policy(
            self.spiking_mode, self.ttfs_cycle_schedule, self.firing_mode,
        )
        neuron_states = [
            policy.make_state(
                batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
                device, COMPUTE_DTYPE,
            )
            for c in cores
        ]

        output_counts = torch.zeros(batch_size, len(output_sources), device=device, dtype=COMPUTE_DTYPE)

        zeros_in = torch.zeros(batch_size, input_size, device=device, dtype=COMPUTE_DTYPE)
        input_signals = [
            torch.zeros(batch_size, max(int(c.axons_per_core - c.available_axons), 1),
                        device=device, dtype=COMPUTE_DTYPE)
            for c in cores
        ]

        record_in_t: list[torch.Tensor] | None = None
        record_out_t: list[torch.Tensor] | None = None
        if recording:
            record_in_t = [
                torch.zeros(max(int(c.axons_per_core - c.available_axons), 1),
                            device=device, dtype=torch.int64)
                for c in cores
            ]
            record_out_t = [
                torch.zeros(max(int(c.neurons_per_core - c.available_neurons), 1),
                            device=device, dtype=torch.int64)
                for c in cores
            ]

        input_spike_train = input_spike_train.to(COMPUTE_DTYPE)
        latency_gated = policy.latency_gated
        single_spike = getattr(policy, "single_spike_io", False)

        if single_spike:
            # Single-spike TTFS: each input fires once (at its arrival cycle). The
            # encoded train is latched, so the per-cycle arrival is its rising edge.
            shifted = torch.zeros_like(input_spike_train)
            shifted[1:] = input_spike_train[:-1]
            input_spike_train = (input_spike_train - shifted).clamp_min_(0.0)

        # Single-spike output decode: per-output-column arrival latch whose running
        # sum over the window reconstructs the ramp value (= count for count/T).
        out_arrival = (torch.zeros(batch_size, len(output_sources),
                                   device=device, dtype=COMPUTE_DTYPE)
                       if single_spike else None)

        for cycle in range(cycles):
            input_spikes = input_spike_train[cycle] if cycle < T else zeros_in

            for core_idx, core in enumerate(cores):
                self._fill_signal_tensor_from_spans(
                    input_signals[core_idx],
                    input_spikes=input_spikes,
                    buffers=buffers,
                    spans=axon_spans[core_idx],
                    cycle=cycle,
                    single_spike=single_spike,
                )

            for core_idx, core in enumerate(cores):
                if core.latency is None:
                    continue
                if latency_gated and not (cycle >= core.latency and cycle < T + core.latency):
                    continue

                buffers[core_idx] = policy.step(
                    neuron_states[core_idx],
                    core_params[core_idx],
                    input_signals[core_idx],
                    thresholds[core_idx],
                    hw_bias=hw_biases[core_idx],
                    thresholding_mode=self.thresholding_mode,
                    output_dtype=COMPUTE_DTYPE,
                )

                if recording:
                    record_in_t[core_idx] += input_signals[core_idx][0].to(torch.int64)
                    record_out_t[core_idx] += buffers[core_idx][0].to(torch.int64).detach()

            if single_spike:
                # Single-spike decode: count each source's latched output ONLY
                # within its own window [src_lat, src_lat+T), so the value is
                # (src_lat + T - fire)/T ∈ [0,1] — the genuine TTFS value. Full-
                # window accumulation would overcount shallow sources by
                # (chip_latency - src_lat)/T, which saturates everything when
                # latency >> T.
                for sp in output_spans:
                    d0 = int(sp.dst_start)
                    d1 = int(sp.dst_end)
                    if sp.kind == "off":
                        continue
                    if sp.kind == "on":
                        # value 1.0 → fires at its window start; counts T over [0, T).
                        if cycle < T:
                            output_counts[:, d0:d1] += 1.0
                        continue
                    if sp.kind == "input":
                        if cycle < T:
                            torch.maximum(
                                out_arrival[:, d0:d1],
                                input_spikes[:, int(sp.src_start):int(sp.src_end)],
                                out=out_arrival[:, d0:d1],
                            )
                            output_counts[:, d0:d1] += out_arrival[:, d0:d1]
                        continue
                    src_lat = cores[int(sp.src_core)].latency
                    if src_lat is None:
                        continue
                    if cycle < int(src_lat) or cycle >= int(src_lat) + T:
                        continue
                    torch.maximum(
                        out_arrival[:, d0:d1],
                        buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)],
                        out=out_arrival[:, d0:d1],
                    )
                    output_counts[:, d0:d1] += out_arrival[:, d0:d1]
                continue

            for sp in output_spans:
                d0 = int(sp.dst_start)
                d1 = int(sp.dst_end)
                if sp.kind == "off":
                    continue
                if sp.kind == "on":
                    # Always-on axon: one spike per input cycle [0, T).
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
                # Accumulate only inside source core's active window [lat, lat+T).
                if cycle < int(src_lat) or cycle >= int(src_lat) + T:
                    continue
                output_counts[:, d0:d1] += buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)]

        if recording:
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

        return output_counts

    def _encode_segment_input(
        self,
        stage,
        seg_input_rates_clamped: torch.Tensor,
        state_buffer_spikes: Dict[int, torch.Tensor],
        *,
        T: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build ``(T, B, in_size)`` spike train; prefer cached LIF trains over uniform encoding."""
        from mimarsinan.spiking.segment_boundary import encode_segment_input

        return encode_segment_input(
            stage,
            seg_input_rates_clamped,
            state_buffer_spikes,
            config=self._boundary_config,
            hybrid_mapping=self.hybrid_mapping,
            T=T,
            batch_size=batch_size,
            device=device,
        )
