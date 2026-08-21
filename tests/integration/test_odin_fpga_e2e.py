"""[ODIN P7a] the END-TO-END physical-backend gate: the cosim IS the device.

The REAL ``OdinFpgaDeploymentStep`` runs, through the real pipeline-step
machinery, against ``RtlCosimTransport`` — so the counts this gate certifies
were produced by the byte-identical vendored ODIN core executing the exporter's
own sequencer program, not by any software twin. Three implementations must
agree at ZERO difference:

  * the HCM torch reference (the step's own parity gate + certificate),
  * the RTL (the deployed counts the step recorded),
  * nevresim (the compiled C++ event-serial policy, an independent third arm).

The record fragment is checked here too: an accuracy read marked ``measured``
and named for the backend, and a timing fragment whose programming wall is
reported SEPARATELY from execution — the per-pass reprogramming cost is the
physics this backend exists to measure.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from integration.odin_fpga_harness import (
    INPUT_LINES,
    NEURONS,
    ODIN_LAW,
    TIMESTEPS,
    entry_sample,
    hybrid_program,
    prepare_step,
    two_core_mapping,
)
from integration.odin_rtl_harness import require_simulator, timed
from integration.parity_harness import ensure_nevresim_ready, have_cxx_compiler

from mimarsinan.chip_simulation.odin_fpga.cosim_transport import RtlCosimTransport
from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.pipelining.pipeline_steps.verification.odin_fpga_simulation_step import (
    OdinFpgaDeploymentStep,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def deployed(request):
    """One deployment of the fixture on the RTL device, through the real step."""
    require_simulator()
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    transport = RtlCosimTransport()
    with timed("P7a end-to-end deployment") as clock:
        pipeline, step = prepare_step(
            monkeypatch, OdinFpgaDeploymentStep, transport=transport)
        step.process()
    report = pipeline.cache["OdinFpgaDeploymentStep.odin_fpga_deployment_results"]
    row = report["timing_fragment"]["per_segment"][0]
    print(
        f"[odin-fpga] transport={report['transport']} "
        f"program={row['programming_s']:.2f}s bytes={row['program_bytes']} "
        f"execute={row['execution_s']:.2f}s device_cycles={row['device_cycles']} "
        f"wall={clock.seconds:.1f}s")
    return pipeline, step, report


def _twin_windows():
    mapping = two_core_mapping()
    trace = simulate_cycles(
        mapping, soma_law=ODIN_LAW,
        input_counts=[[1] * INPUT_LINES for _ in range(TIMESTEPS)],
        simulation_length=TIMESTEPS,
        chip_latency=int(ChipLatency(mapping).calculate()))
    return mapping, trace


class TestTheDeviceReproducesTheReferenceExactly:
    def test_the_step_passed_its_own_parity_gate_and_certificate(self, deployed):
        _pipeline, _step, report = deployed
        assert len(report["certificates"]) == 1
        summary = report["certificates"][0]
        assert "odin_fpga/exact" in summary, summary
        assert "PASS" in summary and "max|dcount|=0" in summary, summary
        assert "exact=1.000000" in summary, summary

    def test_the_deployed_counts_are_the_rtl_and_match_the_cycle_twin(self, deployed):
        _pipeline, step, _report = deployed
        _mapping, trace = _twin_windows()
        deployed_record = _last_run_record(step)
        for core in deployed_record.record.segments[0].cores:
            expected = trace.window_counts()[core.core_index]
            got = tuple(int(v) for v in core.output_spike_count)
            assert got == expected[:len(got)], core.core_index

    def test_the_comparison_is_not_vacuous(self, deployed):
        _pipeline, step, _report = deployed
        record = _last_run_record(step)
        counts = record.per_cycle_counts[0]
        assert counts, "the RTL produced no output events at all"
        assert max(counts.values()) >= 2, (
            "no multiplicity on the wire: a per-cycle law would reproduce this "
            "fixture and the gate would prove nothing")


class TestNevresimIsTheThirdArm:
    @pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
    def test_nevresim_matches_the_deployed_counts(self, deployed):
        _pipeline, step, _report = deployed
        ensure_nevresim_ready()
        from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

        mapping = two_core_mapping()
        chip_latency = ChipLatency(mapping).calculate()
        raster = np.asarray(
            [[1] * INPUT_LINES for _ in range(TIMESTEPS)], dtype=np.float64)
        loader = [(raster.reshape(-1), np.zeros(NEURONS))]
        with timed("P7a nevresim"), tempfile.TemporaryDirectory() as tmp:
            driver = NevresimDriver(
                INPUT_LINES, mapping, tmp, int,
                spike_generation_mode="SpikeTrain", firing_mode="Novena",
                thresholding_mode="<=", spiking_mode="lif", threshold_type=int,
                connectivity_mode="runtime", verbose=False, soma_law=ODIN_LAW)
            _raw, records = driver.predict_spiking_raw_with_records(
                loader, TIMESTEPS, chip_latency)

        deployed_record = _last_run_record(step)
        for core in deployed_record.record.segments[0].cores:
            nevresim = tuple(
                int(v) for v in np.asarray(
                    records[0][core.core_index]["out"])[:NEURONS])
            got = tuple(int(v) for v in core.output_spike_count)
            assert nevresim == got, core.core_index


class TestTheRecordFragmentCarriesMeasuredWalls:
    def test_the_accuracy_read_is_measured_and_named(self, deployed):
        _pipeline, _step, report = deployed
        read = report["accuracy_read"]
        assert read["kind"] == "measured" and read["backend"] == "odin_fpga"

    def test_the_programming_wall_is_measured_and_kept_out_of_execution(
        self, deployed,
    ):
        _pipeline, _step, report = deployed
        timing = report["timing_fragment"]
        assert timing["programming_s"] > 0.0
        assert timing["execution_s"] > 0.0
        row = timing["per_segment"][0]
        assert row["program_bytes"] > 0 and row["cores"] == 2
        assert "rtl_cosim" in row["programming_basis"]
        assert row["device_cycles"] > 0

    def test_the_transport_was_the_rtl_cosimulation(self, deployed):
        _pipeline, _step, report = deployed
        assert report["transport"] == "rtl_cosim"
        assert report["walls"]["total_s"] >= (
            report["walls"]["programming_s"] + report["walls"]["execution_s"] - 1e-9)


def _last_run_record(step):
    """The step's deployed record for the last sample (the runner's own object)."""
    assert step.deployed_records, (
        "the step retained no deployed records; the gate needs the per-cycle "
        "counts to prove the RTL produced them")
    return step.deployed_records[-1]


def test_the_fixture_is_the_one_the_plan_asks_for():
    """A per-EVENT witness: some neuron emits more than one spike in a cycle."""
    _mapping, trace = _twin_windows()
    assert max(
        max(counts) for per_core in trace.outputs for counts in per_core) > 1
    assert len(hybrid_program().stages) == 1
    assert entry_sample().shape == (1, INPUT_LINES)
