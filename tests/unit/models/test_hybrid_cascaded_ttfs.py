"""HCM cascaded ``ttfs_cycle_based`` forward via ``_run_neural_segment_rate``.

Cascaded greedy TTFS reuses the LIF pipelined cycle executor with a swapped
per-cycle policy (``TTFSGreedyCyclePolicy``): integrate latched inputs and fire
**once**, latching the output high for the rest of the window — no reset. We
prove the routing by feeding a single-spike input to one core and checking that
the greedy schedule latches (count = T - fire_cycle) while LIF subtractive-reset
fires only on the spike cycle.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow


def _single_core_flow(spiking_mode, schedule, T):
    core = SimpleNamespace(
        latency=None,
        axons_per_core=1,
        available_axons=0,
        neurons_per_core=1,
        available_neurons=0,
        axon_sources=[SpikeSource(-2, 0, True)],
        core_matrix=np.asarray([[1.0]], dtype=np.float32),
        threshold=1.0,
        hardware_bias=None,
    )
    mapping = SimpleNamespace(
        cores=[core],
        output_sources=np.array([SpikeSource(0, 0, False, False)], dtype=object),
        weight_banks={},
        soft_core_placements_per_hard_core=[[]],
    )
    stage = SimpleNamespace(
        hard_core_mapping=mapping, kind="neural", name="t",
        schedule_segment_index=0, schedule_pass_index=0,
        input_map=[], output_map=[],
    )
    return SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=SimpleNamespace(
            stages=[stage], output_sources=np.array([], dtype=object),
        ),
        simulation_length=T,
        preprocessor=None,
        firing_mode="Default",
        spike_mode="TTFS" if spiking_mode == "ttfs_cycle_based" else "Uniform",
        thresholding_mode="<=",
        spiking_mode=spiking_mode,
        ttfs_cycle_schedule=schedule,
    ), stage


def _single_spike_train(T, fire_cycle):
    train = torch.zeros(T, 1, 1, dtype=torch.float64)
    train[fire_cycle, 0, 0] = 1.0
    return train


def test_cascaded_ttfs_fires_once_and_latches():
    """One spike at cycle 1 → core 0 fires at cycle 1 and is counted only within
    its own window ``[lat, lat+T)`` (per-source windowed decode, the genuine TTFS
    value). Core 0 is at latency 0, window ``[0, T=4)``, fire at cycle 1 →
    count ``T - fire = 3`` (value 0.75). NOT full-window ``S+latency`` accumulation,
    which would overcount to 4 and saturate when latency >> T."""
    T = 4
    flow, stage = _single_core_flow("ttfs_cycle_based", "cascaded", T)
    counts = flow._run_neural_segment_rate(
        stage, input_spike_train=_single_spike_train(T, 1), recorder_seg=None,
    )
    assert float(counts.reshape(-1)[0]) == 3.0


def test_lif_resets_and_fires_only_on_spike_cycle():
    """Same input under LIF subtractive reset → a single fire (no latch)."""
    T = 4
    flow, stage = _single_core_flow("lif", "cascaded", T)
    counts = flow._run_neural_segment_rate(
        stage, input_spike_train=_single_spike_train(T, 1), recorder_seg=None,
    )
    assert float(counts.reshape(-1)[0]) == 1.0


def test_cascaded_ttfs_windowed_decode_recovers_value():
    """A TTFS input encoding ``v`` (spike at ``spike_time = round(S·(1−v))``) is
    decoded within the source's own window → ``count = S − spike_time = round(S·v)``,
    so ``count/S`` recovers ``v`` exactly (genuine TTFS value, no ``+latency``
    overcount). Full-window accumulation would add ``latency`` and break this."""
    from mimarsinan.chip_simulation.recording import spike_modes

    T = 8
    v = 0.5
    flow, stage = _single_core_flow("ttfs_cycle_based", "cascaded", T)
    spike_time = round(T * (1.0 - v))
    rate = torch.tensor([[v]], dtype=torch.float64)
    train = torch.stack(
        [spike_modes.to_spikes(rate, c, simulation_length=T, spike_mode="TTFS")
         for c in range(T)],
        dim=0,
    )
    counts = flow._run_neural_segment_rate(
        stage, input_spike_train=train, recorder_seg=None,
    )
    count = float(counts.reshape(-1)[0])
    assert count == (T - spike_time)
    assert abs(count / T - v) < 1e-9
