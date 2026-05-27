"""Regression for ``SpikingHybridCoreFlow._run_neural_segment_rate``'s
per-source segment-output gating.

The pre-fix bug: ``output_counts`` was incremented from every output
source's buffer at every simulation cycle.  Since a source's
``buffers[core]`` entry is updated only inside its active window
``[lat, lat + T)`` and is **frozen** at the last in-window firing
afterwards, a source that sits at latency ``L < max_core_latency`` got
``(cycles - T - L)`` extra accumulations of its final-cycle firing
piled onto its true in-window total.  Segments whose output sources
were a mix of deep (max-latency) and shallow (e.g. dead-input chains
that ``ChipLatency`` couldn't bump because they only feed segment
outputs) saw the shallow sources over-report by up to ``cycles - T``
spikes per neuron, breaking the rate that downstream stages consume.

We test by patching ``_fill_signal_tensor_from_spans`` to force a
deterministic, known firing schedule on each core, then check that
the segment's ``output_counts`` reports each source's in-window total
regardless of how deep the segment is.
"""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow


def _xcore(core_idx, neuron):
    return SpikeSource(core_idx, neuron, False, False)


def _off():
    return SpikeSource(-1, 0, False, True)


def _make_core(*, used_axons, used_neurons, axon_sources, core_matrix,
               threshold=1.0, hw_bias=None):
    return SimpleNamespace(
        latency=None,
        axons_per_core=used_axons,
        available_axons=0,
        neurons_per_core=used_neurons,
        available_neurons=0,
        axon_sources=axon_sources,
        core_matrix=np.asarray(core_matrix, dtype=np.float32),
        threshold=threshold,
        hardware_bias=(np.asarray(hw_bias, dtype=np.float32)
                       if hw_bias is not None else None),
    )


def test_shallow_dead_input_source_does_not_accumulate_stale_cycles():
    """A bias-only dead-input source feeding the segment output reports
    its in-window fires only — the stale-buffer cycles after its
    window closes must not be re-accumulated.

    Topology: a four-deep real chain anchors the segment latency at
    ``max_core_latency=3`` (so ``cycles = max_delay + T = 8``).  A
    dead-input "side branch" with bias > threshold fires every cycle
    in its [0, T) window; its only consumer is the segment output.
    Pre-fix: ``output_counts`` reports ``T + (cycles - T - 0) = 8``
    spikes for that branch.  Post-fix: exactly ``T``.
    """
    T = 4

    # Real four-deep chain — saturating bias keeps every core firing
    # in its window so ChipLatency stays well-defined.
    chain = []
    prev = _xcore(-2, 0)  # segment input source for c0
    for i in range(4):
        sources = [_xcore(i - 1, 0) if i > 0 else SpikeSource(-2, 0, False, False)]
        chain.append(_make_core(
            used_axons=1, used_neurons=1,
            axon_sources=sources,
            core_matrix=[[1.0]],
            threshold=1.0,
            hw_bias=[2.0],  # bias-driven fires every cycle in window
        ))

    # Dead-input side branch — same bias pattern, but no live source.
    shallow = _make_core(
        used_axons=1, used_neurons=1,
        axon_sources=[_off()],
        core_matrix=[[0.0]],
        threshold=1.0,
        hw_bias=[2.0],
    )

    cores = chain + [shallow]
    shallow_idx = 4
    output_sources = np.array(
        [_xcore(shallow_idx, 0), _xcore(3, 0)],
        dtype=object,
    )
    mapping = SimpleNamespace(
        cores=cores,
        output_sources=output_sources,
        weight_banks={},
        soft_core_placements_per_hard_core=[[] for _ in cores],
    )
    stage = SimpleNamespace(
        hard_core_mapping=mapping, kind="neural", name="t",
        schedule_segment_index=0, schedule_pass_index=0,
        input_map=[], output_map=[],
    )
    flow = SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=SimpleNamespace(
            stages=[stage], output_sources=np.array([], dtype=object),
        ),
        simulation_length=T,
        preprocessor=None,
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<",
        spiking_mode="lif",
    )

    input_spike_train = torch.zeros(T, 1, 1, dtype=torch.float64)
    counts = flow._run_neural_segment_rate(
        stage, input_spike_train=input_spike_train, recorder_seg=None,
    )

    # Sanity: ChipLatency placed deep at lat=3, shallow at lat=0
    # (its only consumer is the segment output, so the forward-cascade
    # post-pass has no in-segment consumer to bump it against).
    assert cores[3].latency == 3
    assert cores[shallow_idx].latency == 0

    counts_np = counts.detach().cpu().numpy().reshape(-1)
    # Shallow fires every cycle of [0, 4) → exactly T fires; pre-fix
    # would have reported T + (cycles - T) = cycles = 8.
    assert counts_np[0] == T, (
        f"shallow source must report {T} in-window fires; got {counts_np[0]} "
        "(stale-buffer accumulation regression)"
    )
    # Deep core (lat=3): fires every cycle of [3, 7) → also exactly T.
    # Pre-fix would have reported T+1 because of the cycle-7 stale read.
    assert counts_np[1] == T, (
        f"deep source must report {T} in-window fires; got {counts_np[1]}"
    )
