"""[ODIN5] R11a -- semantic equivalence of the exported images against the RTL.

Plan §7 row 15, the first half of the owner's "deployable" (§3): a
``HardCoreMapping`` with integral weights and thresholds is packed by the REAL
exporter (feasibility gates included), and the resulting images and sequencer
program are executed on the byte-identical vendored ODIN core. Three
implementations of one soma law are compared at ZERO difference:

  * the torch fold  -- ``models/spiking/serial/fold.py``, the kernel both torch
    executors call, driven cycle by cycle through nevresim's own axon gather;
  * nevresim        -- the compiled C++ ``EventSerialIntegrate`` policy;
  * the RTL         -- the vendored crossbar, programmed over SPI.

The fixture carries the witnesses the plan asks for: a theta at the membrane
CEILING (255), multiplicity greater than one on the wire, and THREE consecutive
samples of which the second repeats the first's input so that an equal answer is
only possible if the per-sample CLEAR really rewrote the membrane state -- no
weight is reprogrammed between samples.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from integration.odin_rtl_harness import (
    MEMBRANE_CEILING,
    ODIN_LAW,
    compare_cycle_counts,
    export_of,
    hard_core,
    mapping_of,
    require_simulator,
    timed,
    traces_for,
)
from integration.parity_harness import ensure_nevresim_ready, have_cxx_compiler

from mimarsinan.certification.spike_certificate import certify_spike_counts
from mimarsinan.chip_simulation.odin_rtl.cosim import run_cosim
from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles
from mimarsinan.chip_simulation.odin_rtl.stimulus import OP_SPI_W, OP_TAG
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

AXONS = 16
NEURONS = 16
S = 4
PRODUCER_THETA = 15.0
SEED = 7


def _weights():
    rng = np.random.default_rng(SEED)
    producer = rng.integers(-7, 8, size=(AXONS + 1, NEURONS)).astype(np.float64)
    consumer = rng.integers(3, 8, size=(AXONS, NEURONS)).astype(np.float64)
    consumer[:, 0] = 7.0
    return producer, consumer


def _mapping():
    producer_w, consumer_w = _weights()
    producer = hard_core(
        producer_w, threshold=PRODUCER_THETA,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(AXONS)]
        + [SpikeSource(-3, 0, is_always_on=True)],
    )
    consumer = hard_core(
        consumer_w, threshold=float(MEMBRANE_CEILING),
        sources=[SpikeSource(0, i) for i in range(AXONS)],
    )
    mapping = mapping_of(
        [producer, consumer], [SpikeSource(1, i) for i in range(NEURONS)])
    chip_latency = ChipLatency(mapping).calculate()
    return mapping, chip_latency


def _rasters():
    """Three consecutive samples; sample 1 REPEATS sample 0 (the CLEAR witness)."""
    ones = [[1] * AXONS for _ in range(S)]
    alternating = [
        [1 if (cycle + axon) % 2 == 0 else 0 for axon in range(AXONS)]
        for cycle in range(S)
    ]
    return [ones, [list(row) for row in ones], alternating]


@pytest.fixture(scope="module")
def r11a():
    require_simulator()
    mapping, chip_latency = _mapping()
    export = export_of(mapping)
    rasters = _rasters()
    samples = traces_for(
        mapping, rasters, simulation_length=S, chip_latency=chip_latency)
    with timed("R11a cosim") as clock:
        result = run_cosim(
            export, [list(sample.per_cycle) for sample in samples],
            latencies=samples[0].trace.latencies)
    windows = sum(
        len(sample.trace.outputs) * len(mapping.cores) * NEURONS
        for sample in samples)
    print(f"[odin-rtl] R11a engine={result.build.engine} "
          f"build={result.build.build_seconds:.1f}s cached={result.build.cached} "
          f"sim={result.run.seconds:.1f}s cycles={result.capture.cycles} "
          f"tokens={result.token_count} comparisons={windows} "
          f"wall={clock.seconds:.1f}s")
    return mapping, chip_latency, export, rasters, samples, result


class TestTheRtlAgreesWithTheTorchFoldAtEveryCycle:
    def test_no_cycle_of_any_sample_differs(self, r11a):
        _mapping, _lat, _export, _rasters, samples, result = r11a
        assert compare_cycle_counts(result, samples) == []

    def test_the_window_counts_agree(self, r11a):
        mapping, _lat, _export, _rasters, samples, result = r11a
        rtl = result.window_counts(
            latencies=samples[0].trace.latencies, simulation_length=S,
            neurons=[int(core.neurons_per_core) for core in mapping.cores])
        for index, sample in enumerate(samples):
            assert rtl[index] == sample.trace.window_counts(), index

    def test_the_comparison_is_not_vacuous(self, r11a):
        _mapping, _lat, _export, _rasters, samples, result = r11a
        assert result.capture.events, "the RTL produced no output events at all"
        emitted = [
            max(counts) for sample in samples
            for per_core in sample.trace.outputs for counts in per_core
        ]
        assert max(emitted) >= 2, "no multiplicity on the wire: a per-cycle law "\
            "would reproduce this fixture and the gate would prove nothing"


class TestNevresimIsTheThirdArm:
    @pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
    def test_nevresim_matches_the_fold_and_therefore_the_rtl(self, r11a):
        mapping, chip_latency, _export, rasters, samples, result = r11a
        ensure_nevresim_ready()
        from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

        loader = [
            (np.asarray(raster, dtype=np.float64).reshape(-1), np.zeros(NEURONS))
            for raster in rasters
        ]
        with timed("R11a nevresim"), tempfile.TemporaryDirectory() as tmp:
            driver = NevresimDriver(
                AXONS, mapping, tmp, int,
                spike_generation_mode="SpikeTrain", firing_mode="Novena",
                thresholding_mode="<=", spiking_mode="lif", threshold_type=int,
                connectivity_mode="runtime", verbose=False, soma_law=ODIN_LAW)
            _raw, records = driver.predict_spiking_raw_with_records(
                loader, S, chip_latency)

        rtl = result.window_counts(
            latencies=samples[0].trace.latencies, simulation_length=S,
            neurons=[int(core.neurons_per_core) for core in mapping.cores])
        for sample, trace in enumerate(samples):
            fold = trace.trace.window_counts()
            for core in range(len(mapping.cores)):
                nevresim = tuple(
                    int(v) for v in np.asarray(records[sample][core]["out"])[:NEURONS])
                assert nevresim == fold[core], (sample, core)
                assert nevresim == rtl[sample][core], (sample, core)


class TestTheVerdictIsRecordedAsACertificate:
    def test_the_cosim_is_classified_exact_and_certifies_at_zero_delta(self, r11a):
        mapping, _lat, _export, _rasters, samples, result = r11a
        neurons = [int(core.neurons_per_core) for core in mapping.cores]
        rtl = result.window_counts(
            latencies=samples[0].trace.latencies, simulation_length=S,
            neurons=neurons)

        def _counts(source):
            def read(batch):
                sample = int(batch[0].item())
                return {
                    f"core{core}": torch.tensor(
                        [source(sample, core)], dtype=torch.float64)
                    for core in range(len(neurons))
                }
            return read

        certificate = certify_spike_counts(
            _counts(lambda s, c: samples[s].trace.window_counts()[c]),
            _counts(lambda s, c: rtl[s][c]),
            [torch.tensor([index]) for index in range(len(samples))],
            backend="odin_rtl",
        )
        print(f"[odin-rtl] {certificate.summary()}")
        assert certificate.backend_class == "exact"
        assert certificate.max_abs_delta == 0.0
        assert certificate.exact_match_fraction == 1.0
        assert certificate.neuron_windows_compared == len(samples) * sum(neurons)
        assert certificate.passed


class TestTheSecondSampleRunsOnAClearAlone:
    def test_the_repeated_sample_reproduces_the_first_exactly(self, r11a):
        mapping, _lat, _export, _rasters, _samples, result = r11a
        neurons = [int(core.neurons_per_core) for core in mapping.cores]
        for cycle in range(result.plan.cycles_per_sample):
            for core, count in enumerate(neurons):
                assert result.cycle_counts(0, cycle, core, count) == \
                    result.cycle_counts(1, cycle, core, count), (cycle, core)

    def test_that_equality_has_teeth_because_carried_state_would_change_it(self, r11a):
        mapping, _lat, _export, rasters, samples, _result = r11a
        span = len(samples[0].trace.outputs)
        continuous = simulate_cycles(
            mapping, soma_law=ODIN_LAW,
            input_counts=[row for _ in range(2) for row in rasters[0]]
            + [[0] * AXONS] * (2 * span),
            simulation_length=2 * span, chip_latency=0)
        first = continuous.outputs[:span]
        second = continuous.outputs[span:2 * span]
        assert first != second, (
            "the fixture's membranes come back to their starting state on their "
            "own, so an equal second sample would prove nothing about CLEAR")

    def test_no_weight_or_parameter_byte_is_rewritten_between_samples(self, r11a):
        _mapping, _lat, _export, _rasters, _samples, result = r11a
        ops = result.plan.ops
        first = next(i for i, op in enumerate(ops)
                     if op.code == OP_TAG and op.args[0] == result.plan.tag_of(0, 0))
        second = next(i for i, op in enumerate(ops)
                      if op.code == OP_TAG and op.args[0] == result.plan.tag_of(1, 0))
        for op in ops[first:second]:
            if op.code != OP_SPI_W:
                continue
            command = (op.args[1] >> 16) & 0b11
            if command == 0b00:
                continue                       # the two SPI_GATE_ACTIVITY toggles
            assert command == 0b01, "a synapse word was rewritten between samples"
            assert (op.args[1] >> 8) & 0xF in (8, 9, 10), \
                "a neuron PARAMETER byte was rewritten between samples"


class TestTheFixtureIsTheOneThePlanAsksFor:
    def test_the_consumer_theta_sits_at_the_membrane_ceiling(self, r11a):
        mapping, _lat, export, _rasters, _samples, _result = r11a
        assert mapping.cores[1].threshold == MEMBRANE_CEILING
        assert export.manifest["feasibility"]["theta_per_core"]["1"] == MEMBRANE_CEILING

    def test_there_are_at_least_two_consecutive_samples(self, r11a):
        _mapping, _lat, _export, _rasters, samples, result = r11a
        assert result.plan.samples == len(samples) >= 2

    def test_the_export_went_through_the_real_feasibility_gates(self, r11a):
        _mapping, _lat, export, _rasters, _samples, _result = r11a
        assert all(export.manifest["feasibility"]["gates"].values())
        assert export.manifest["emission_bounds"]["max"] <= \
            export.manifest["emission_bounds"]["ceiling"]
