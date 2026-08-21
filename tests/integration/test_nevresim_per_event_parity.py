"""[ODIN P3, plan §7 row 13] nevresim executes the event-serial soma law.

Three claims, all on real compiled nevresim binaries:

1. **Parity.** Per-neuron per-core WINDOW COUNTS agree with the HCM torch twin
   at atol=0 under the per-event point, on a deliberately MULTI-SPIKING fixture
   (a neuron that fires several times in one cycle, and a downstream neuron
   that consumes those multiplicities). Two implementations of the same fold
   (``models/spiking/serial/fold.py`` and ``EventSerialIntegrate``) must be the
   same arithmetic, not merely similar.
2. **Carry.** A counted raster survives the segment boundary: segment 1's
   per-cycle multiplicities are what segment 2 actually integrates. The old
   seam binarized on load, which would deliver 1 where the producer fired 3.
3. **Identity.** The documented self-check — the trains' window sum IS the
   SPKREC count — holds under counts, which is exactly where a bit raster
   would silently break it.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from integration.parity_harness import ensure_nevresim_ready, have_cxx_compiler

from mimarsinan.chip_simulation.simulation_runner.segment_run import (
    raster_from_spike_trains,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

pytestmark = pytest.mark.integration

T = 4
THETA = 1
FAN = 3          # three axon slots carry the SAME input line ...
WEIGHT = 3       # ... each one supra-threshold, so ONE cycle emits FAN spikes

PER_EVENT_8 = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "firing_granularity": "per_event",
    "membrane_bits": 8,
})


def _core(axons, neurons, matrix, sources, latency):
    """One biasless hard core — biasless because the per-event point requires a
    param-encoded bias, which is what leaves nevresim's per-cycle ``bias_`` at
    zero and the bias delivered as an always-on ROW instead."""
    core = HardCore(axons, neurons, has_bias_capability=False)
    core.core_matrix = np.asarray(matrix, dtype=np.float64)
    core.axon_sources = list(sources)
    core.available_axons = 0
    core.available_neurons = 0
    core.threshold = float(THETA)
    core.latency = latency
    return core


def _multi_spiking_mapping() -> HybridHardCoreMapping:
    """Core 0 fires ``FAN`` times per cycle; core 1 consumes that multiplicity.

    Core 0's three axon slots all source input line 0, so a single 0/1 input
    spike arrives as three ADJACENT event occurrences — three threshold
    crossings, three spikes, in one cycle. Core 1 sees that COUNT on one axon
    with a unit weight and a unit threshold, so it re-emits it exactly.
    """
    producer = _core(
        FAN, 1,
        np.full((FAN, 1), WEIGHT, dtype=np.float64),
        [SpikeSource(-2, 0, is_input=True)] * FAN,
        latency=0,
    )
    consumer = _core(
        1, 1, np.array([[1.0]]), [SpikeSource(0, 0)], latency=1,
    )
    segment = HardCoreMapping([])
    segment.cores = [producer, consumer]
    segment.output_sources = np.asarray(
        [SpikeSource(0, 0), SpikeSource(1, 0)], dtype=object)
    stage = HybridStage(
        kind="neural",
        name="per_event_fixture",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=2)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray(
            [IRSource(node_id=0, index=i) for i in range(2)], dtype=object),
    )


def _relay_segment() -> HardCoreMapping:
    """A unit relay: theta=1, w=1, one axon fed by the carried train. Under the
    per-event law it maps k arriving events to exactly k spikes, so its window
    count IS the carried multiplicity — which makes it the instrument that
    reads what the seam delivered."""
    segment = HardCoreMapping([])
    segment.cores = [
        _core(1, 1, np.array([[1.0]]),
              [SpikeSource(-2, 0, is_input=True)], latency=0)
    ]
    segment.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    return segment


def _driver(segment, tmp, *, spike_generation_mode):
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

    return NevresimDriver(
        1 if spike_generation_mode == "Uniform" else 1,
        segment,
        tmp,
        int,
        spike_generation_mode=spike_generation_mode,
        firing_mode="Novena",
        thresholding_mode="<=",
        spiking_mode="lif",
        threshold_type=int,
        connectivity_mode="runtime",
        verbose=False,
        soma_law=PER_EVENT_8,
    )


def _torch_records(hybrid):
    flow = SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="Novena",
        spike_mode="Uniform",
        thresholding_mode="<=",
        spiking_mode="lif",
        soma_law=PER_EVENT_8,
    ).eval()
    with torch.no_grad():
        _out, record = flow.forward_with_recording(
            torch.ones(1, 1, dtype=torch.float32), sample_index=0)
    return record


@pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
def test_nevresim_and_hcm_agree_per_neuron_at_atol_zero_under_per_event():
    ensure_nevresim_ready()
    hybrid = _multi_spiking_mapping()
    segment = hybrid.stages[0].hard_core_mapping
    assert segment is not None
    latency = ChipLatency(segment).calculate()
    loader = [(np.asarray([1.0], dtype=np.float64), np.zeros(1))]

    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(segment, tmp, spike_generation_mode="Uniform")
        _raw, records = driver.predict_spiking_raw_with_records(loader, T, latency)

    torch_record = _torch_records(hybrid)
    torch_cores = torch_record.segments[0].cores
    assert len(torch_cores) == 2

    for core in torch_cores:
        expected = np.asarray(core.output_spike_count, dtype=np.int64)
        used = int(core.n_out_used)
        got = np.asarray(
            records[0][core.core_index]["out"][:used], dtype=np.int64)
        np.testing.assert_array_equal(got, expected[:used])

    # The witness is non-degenerate: a per-CYCLE law could not produce these.
    producer_count = int(records[0][0]["out"][0])
    assert producer_count == FAN * T, producer_count
    assert int(records[0][1]["out"][0]) == FAN * T


@pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
def test_the_train_window_sum_is_the_spkrec_count_under_counts():
    ensure_nevresim_ready()
    hybrid = _multi_spiking_mapping()
    segment = hybrid.stages[0].hard_core_mapping
    assert segment is not None
    latency = ChipLatency(segment).calculate()
    loader = [(np.asarray([1.0], dtype=np.float64), np.zeros(1))]

    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(segment, tmp, spike_generation_mode="Uniform")
        _raw, records, trains = driver.predict_spiking_raw_with_spike_trains(
            loader, T, latency)

    for core, raster in trains[0].items():
        np.testing.assert_array_equal(
            raster.sum(axis=1).astype(np.int64),
            np.asarray(records[0][core]["out"], dtype=np.int64),
        )
    # The identity is being tested where it BREAKS under a bit raster: several
    # spikes land in a single cycle.
    assert int(trains[0][0].max()) == FAN


@pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
def test_the_counted_raster_carries_multiplicities_across_the_boundary():
    """The 2-segment round trip: segment 1's per-cycle counts arrive intact."""
    ensure_nevresim_ready()
    hybrid = _multi_spiking_mapping()
    producing = hybrid.stages[0].hard_core_mapping
    assert producing is not None
    latency = ChipLatency(producing).calculate()
    loader = [(np.asarray([1.0], dtype=np.float64), np.zeros(1))]

    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(producing, tmp, spike_generation_mode="Uniform")
        _raw, _records, trains = driver.predict_spiking_raw_with_spike_trains(
            loader, T, latency)

    raster = raster_from_spike_trains(trains[0], [("core", 1, 0)], T, None)
    assert raster[:, 0].tolist() == [FAN] * T, raster.tolist()

    consuming = _relay_segment()
    carried = [(
        raster.reshape(-1).astype(np.float64),
        np.zeros(1, dtype=np.float64),
    )]
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(consuming, tmp, spike_generation_mode="SpikeTrain")
        _raw2, records2 = driver.predict_spiking_raw_with_records(
            carried, T, ChipLatency(consuming).calculate())

    # A relay maps k events to k spikes; a seam that binarized on load would
    # report T (one per non-empty cycle) instead of FAN*T.
    assert int(records2[0][0]["out"][0]) == FAN * T
