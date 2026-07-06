"""Golden equivalence: the active-set rate-segment loop is bit-equal to the full walk.

W3c: the per-core-per-cycle rate path filled every core's axon signal on every
cycle even though latency gating discards all work outside a core's
``[latency, latency+T)`` window. The optimized loop iterates precomputed
per-cycle active sets; this file locks bit-equality against a verbatim copy of
the pre-W3c loop across firing modes, span kinds, latencies, and recording.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.recording.spike_recorder import (
    CoreSpikeCounts,
    SegmentSpikeRecord,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


def _reference_run_neural_segment_rate(
    flow, stage, *, input_spike_train, recorder_seg=None
):
    """Verbatim copy of the pre-W3c ``_run_neural_segment_rate`` loop."""
    mapping = stage.hard_core_mapping
    assert mapping is not None

    T = flow.simulation_length
    assert input_spike_train.shape[0] == T

    batch_size = input_spike_train.shape[1]
    device = input_spike_train.device
    input_size = input_spike_train.shape[2]
    recording = recorder_seg is not None
    if recording:
        assert batch_size == 1

    seg = flow._get_segment_tensors(stage, device)
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
                    device=device, dtype=COMPUTE_DTYPE)
        for c in cores
    ]
    policy = cycle_neuron_policy(
        flow.spiking_mode, flow.ttfs_cycle_schedule, flow.firing_mode,
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

    record_in_t = record_out_t = None
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
        shifted = torch.zeros_like(input_spike_train)
        shifted[1:] = input_spike_train[:-1]
        input_spike_train = (input_spike_train - shifted).clamp_min_(0.0)

    out_arrival = (torch.zeros(batch_size, len(output_sources),
                               device=device, dtype=COMPUTE_DTYPE)
                   if single_spike else None)

    for cycle in range(cycles):
        input_spikes = input_spike_train[cycle] if cycle < T else zeros_in

        for core_idx, core in enumerate(cores):
            local_cycle = cycle - int(core.latency or 0)
            core_input_spikes = (
                input_spike_train[local_cycle]
                if 0 <= local_cycle < T
                else zeros_in
            )
            flow._fill_signal_tensor_from_spans(
                input_signals[core_idx],
                input_spikes=core_input_spikes,
                buffers=buffers,
                plan=axon_fill_plans[core_idx],
                cycle=cycle,
                single_spike=single_spike,
                latency=int(core.latency or 0),
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
                thresholding_mode=flow.thresholding_mode,
                output_dtype=COMPUTE_DTYPE,
            )

            if record_in_t is not None and record_out_t is not None:
                record_in_t[core_idx] += input_signals[core_idx][0].to(torch.int64)
                record_out_t[core_idx] += buffers[core_idx][0].to(torch.int64).detach()

        if single_spike:
            assert out_arrival is not None
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

    if recorder_seg is not None:
        assert record_in_t is not None and record_out_t is not None
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


def _random_segment(rng: random.Random, *, T: int, n_cores: int = 5):
    """Random small multi-core neural segment with mixed span kinds and latencies."""
    input_width = rng.randint(2, 4)
    cores = []
    latencies = []
    for ci in range(n_cores):
        axons = rng.randint(2, 5)
        neurons = rng.randint(1, 4)
        core = HardCore(axons + 1, neurons + 1, has_bias_capability=False)
        core.core_matrix = np.zeros((axons + 1, neurons + 1), dtype=np.float64)
        for a in range(axons):
            for n in range(neurons):
                core.core_matrix[a, n] = rng.choice([0.0, 0.25, 0.5, 1.0, -0.25])
        sources = []
        for _ in range(axons):
            pick = rng.random()
            if pick < 0.35 or ci == 0:
                sources.append(SpikeSource(-2, rng.randrange(input_width), is_input=True))
            elif pick < 0.45:
                sources.append(SpikeSource(0, 0, is_always_on=True))
            elif pick < 0.55:
                sources.append(SpikeSource(0, 0, is_off=True))
            else:
                src_core = rng.randrange(ci)
                src_neuron = rng.randrange(
                    max(cores[src_core].neurons_per_core - cores[src_core].available_neurons, 1)
                )
                sources.append(SpikeSource(src_core, src_neuron))
        core.axon_sources = sources
        core.available_axons = 1
        core.available_neurons = 1
        core.threshold = rng.choice([0.5, 1.0, 1.5])
        core.latency = 0 if ci == 0 else rng.randint(1, 3)
        latencies.append(core.latency)
        cores.append(core)

    segment = HardCoreMapping([])
    segment.cores = cores
    out_sources = []
    for _ in range(3):
        pick = rng.random()
        if pick < 0.2:
            out_sources.append(SpikeSource(-2, rng.randrange(input_width), is_input=True))
        else:
            src_core = rng.randrange(n_cores)
            src_neuron = rng.randrange(
                max(cores[src_core].neurons_per_core - cores[src_core].available_neurons, 1)
            )
            out_sources.append(SpikeSource(src_core, src_neuron))
    segment.output_sources = np.asarray(out_sources, dtype=object)

    stage = HybridStage(
        kind="neural",
        name="random_segment",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=input_width)],
        output_map=[SegmentIOSlice(node_id=1, offset=0, size=len(out_sources))],
    )
    hybrid = HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray(
            [IRSource(node_id=1, index=i) for i in range(len(out_sources))], dtype=object
        ),
    )
    return hybrid, stage, input_width


def _make_flow(hybrid, *, T, mode):
    if mode == "casc":
        kwargs = dict(
            firing_mode="TTFS", spike_mode="TTFS", thresholding_mode="<=",
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded",
        )
    else:
        kwargs = dict(
            firing_mode=mode, spike_mode="Uniform", thresholding_mode="<=",
            spiking_mode="lif",
        )
    return SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        cycle_accurate_lif_forward=True,
        **kwargs,
    ).eval()


def _input_train(rng: random.Random, *, T, batch, width, single_spike: bool):
    if single_spike:
        # Cumulative 0 -> 1 latch trains (the single-spike wire domain).
        train = torch.zeros(T, batch, width)
        for b in range(batch):
            for a in range(width):
                start = rng.randrange(T + 2)
                if start < T:
                    train[start:, b, a] = 1.0
        return train
    return (torch.rand(T, batch, width, generator=torch.Generator().manual_seed(rng.randrange(2**31))) < 0.5).float()


def _empty_record():
    return SegmentSpikeRecord(
        stage_index=0,
        stage_name="s",
        schedule_segment_index=None,
        schedule_pass_index=None,
        seg_input_rates=np.zeros(1),
        seg_input_spike_count=np.zeros(1),
        seg_output_spike_count=np.zeros(1),
    )


class TestRateSegmentLoopEquivalence:
    def _assert_equal(self, seed, mode, *, batch=3, T=6, record=False):
        rng = random.Random(seed)
        hybrid, stage, width = _random_segment(rng, T=T)
        flow = _make_flow(hybrid, T=T, mode=mode)
        train = _input_train(
            rng, T=T, batch=batch, width=width,
            single_spike=(mode == "casc"),
        )
        rec_new = _empty_record() if record else None
        got = flow._run_neural_segment_rate(
            stage, input_spike_train=train.clone(), recorder_seg=rec_new,
        )
        rec_ref = _empty_record() if record else None
        want = _reference_run_neural_segment_rate(
            flow, stage, input_spike_train=train.clone(), recorder_seg=rec_ref,
        )
        assert torch.equal(got, want), (seed, mode)
        if record:
            assert rec_new is not None and rec_ref is not None
            assert len(rec_new.cores) == len(rec_ref.cores)
            for a, b in zip(rec_new.cores, rec_ref.cores):
                assert a.core_index == b.core_index
                assert a.core_latency == b.core_latency
                assert np.array_equal(a.input_spike_count, b.input_spike_count)
                assert np.array_equal(a.output_spike_count, b.output_spike_count)

    @pytest.mark.parametrize("seed", range(10))
    def test_lif_default_bit_equal(self, seed):
        self._assert_equal(seed, "Default")

    @pytest.mark.parametrize("seed", range(10))
    def test_lif_novena_bit_equal(self, seed):
        self._assert_equal(seed, "Novena")

    @pytest.mark.parametrize("seed", range(10))
    def test_cascaded_single_spike_bit_equal(self, seed):
        self._assert_equal(seed, "casc")

    @pytest.mark.parametrize("seed", range(3))
    def test_recording_path_bit_equal(self, seed):
        self._assert_equal(seed, "Default", batch=1, record=True)
