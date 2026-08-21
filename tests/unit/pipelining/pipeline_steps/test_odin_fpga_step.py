"""[ODIN P7a] the physical backend STEP, driven through the real step machinery.

The transport here is a stand-in device (the cycle-accurate twin behind the same
seam), so this gate is about the STEP: does it export every neural segment,
certify the device's counts against the HCM reference at the exact class, and
emit a record fragment whose walls are MEASURED — including the programming
wall, which is the reprogramming physics and never folded into execution?

The device behind the seam being real RTL is the [slow] gate
``tests/integration/test_odin_fpga_e2e.py``.
"""

from __future__ import annotations

import time

import pytest

from integration.odin_fpga_harness import (
    NEURONS,
    ODIN_LAW,
    TIMESTEPS,
    prepare_step,
)

from mimarsinan.chip_simulation.odin_fpga.payload import (
    counts_from_events,
    payload_bytes,
    program_plan,
    run_plan,
)
from mimarsinan.chip_simulation.odin_fpga.transport import (
    DeviceTransportError,
    ProgramReceipt,
    TransportRun,
)
from mimarsinan.chip_simulation.odin_rtl.capture import CaptureEvent
from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.pipelining.pipeline_steps.verification.odin_fpga_simulation_step import (
    OdinFpgaDeploymentStep,
)


class TwinTransport:
    """A device that executes the cycle-accurate twin behind the SAME seam.

    It replays the injected plan through ``simulate_cycles`` and emits the
    resulting AER events, so the step sees a device-shaped answer without a
    Verilog simulator. What it CANNOT prove is that real RTL agrees — that is
    exactly what the [slow] cosimulation gate is for.
    """

    name = "twin"

    def __init__(self, mapping, *, program_wall_s: float = 0.0):
        self.mapping = mapping
        self.program_wall_s = float(program_wall_s)
        self.programmed: list = []
        self.runs = 0
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        self.opened += 1

    def close(self) -> None:
        self.closed += 1

    def program(self, export) -> ProgramReceipt:
        plan = program_plan(export)
        self.programmed.append(export)
        time.sleep(self.program_wall_s)
        return ProgramReceipt(
            payload=payload_bytes(plan.ops), ops=len(plan.ops),
            cores=plan.n_cores, wall_s=self.program_wall_s,
            basis="twin: the injected plan replayed by the cycle-accurate twin",
        )

    def run_samples(self, per_cycle_inputs, *, latencies) -> TransportRun:
        self.runs += 1
        export = self.programmed[-1]
        plan = run_plan(export, per_cycle_inputs, latencies=latencies)
        raster = [[1] * len(self.mapping.cores[0].axon_sources[:-1])
                  for _ in range(TIMESTEPS)]
        trace = simulate_cycles(
            self.mapping, soma_law=ODIN_LAW, input_counts=raster,
            simulation_length=TIMESTEPS,
            chip_latency=int(ChipLatency(self.mapping).calculate()))
        events = [
            CaptureEvent(core=core, neuron=neuron, cycle=cycle,
                         tag=plan.tag_of(0, cycle))
            for cycle, per_core in enumerate(trace.outputs)
            for core, counts in enumerate(per_core)
            for neuron, count in enumerate(counts)
            for _ in range(int(count))
        ]
        return TransportRun(
            counts=counts_from_events(plan, events), samples=plan.samples,
            cycles_per_sample=plan.cycles_per_sample, wall_s=0.001,
            program_wall_s=self.program_wall_s, device_cycles=len(events) * 7,
            detail={"engine": "twin"},
        )


@pytest.fixture
def deployed(monkeypatch):
    from integration.odin_fpga_harness import two_core_mapping

    transport = TwinTransport(two_core_mapping(), program_wall_s=0.01)
    pipeline, step = prepare_step(
        monkeypatch, OdinFpgaDeploymentStep, transport=transport)
    step.process()
    return pipeline, step, transport


class TestTheStepDeploysEverySegmentAndCertifiesIt:
    def test_the_counts_certificate_is_exact_at_zero_delta(self, deployed):
        pipeline, _step, _transport = deployed
        report = pipeline.cache["OdinFpgaDeploymentStep.odin_fpga_deployment_results"]
        assert report["backend"] == "odin_fpga"
        assert len(report["certificates"]) == 1
        summary = report["certificates"][0]
        assert "odin_fpga/exact" in summary
        assert "PASS" in summary and "max|dcount|=0" in summary

    def test_the_device_was_programmed_before_every_run_and_the_session_closed(
        self, deployed,
    ):
        _pipeline, _step, transport = deployed
        assert transport.opened == 1 and transport.closed == 1
        assert len(transport.programmed) == transport.runs == 1

    def test_the_verdict_names_the_rule_and_the_transport(self, deployed):
        _pipeline, step, _transport = deployed
        assert step._verdict["status"] == "pass"
        assert "spike parity" in step._verdict["rule"]
        assert step._verdict["detail"]["transport"] == "twin"

    def test_the_step_is_metric_neutral(self, deployed):
        pipeline, step, _transport = deployed
        assert step.validate() == pipeline.get_target_metric()


class TestTheRecordFragmentCarriesMeasuredWalls:
    def test_the_accuracy_read_is_measured_and_names_the_backend(self, deployed):
        pipeline, _step, _transport = deployed
        report = pipeline.cache["OdinFpgaDeploymentStep.odin_fpga_deployment_results"]
        read = report["accuracy_read"]
        assert read["kind"] == "measured"
        assert read["backend"] == "odin_fpga"
        assert read["samples"] == 1

    def test_programming_is_reported_separately_from_execution(self, deployed):
        pipeline, _step, _transport = deployed
        report = pipeline.cache["OdinFpgaDeploymentStep.odin_fpga_deployment_results"]
        timing = report["timing_fragment"]
        assert timing["programming_s"] >= 0.01
        assert timing["execution_s"] > 0.0
        assert timing["programming_s"] not in (timing["execution_s"],)
        assert "never folded" in timing["note"]

    def test_every_segment_carries_its_own_programming_cost_and_basis(self, deployed):
        pipeline, _step, _transport = deployed
        report = pipeline.cache["OdinFpgaDeploymentStep.odin_fpga_deployment_results"]
        rows = report["timing_fragment"]["per_segment"]
        assert len(rows) == 1
        row = rows[0]
        assert row["cores"] == 2
        assert row["program_bytes"] > 0
        assert row["programming_s"] >= 0.01
        assert "twin" in row["programming_basis"]
        assert row["device_cycles"] > 0

    def test_the_headline_metrics_separate_the_two_walls(self, deployed):
        pipeline, _step, _transport = deployed
        names = [name for name, _value in pipeline.reporter.events]
        assert "ODIN FPGA Spike Parity" in names
        assert "ODIN FPGA Programming (s)" in names
        assert "ODIN FPGA Execution (s)" in names


class TestTheStepRefusesWhatTheDeviceCannotRun:
    def test_a_wrong_sample_count_refuses_by_key(self, monkeypatch):
        from integration.odin_fpga_harness import two_core_mapping

        _pipeline, step = prepare_step(
            monkeypatch, OdinFpgaDeploymentStep,
            transport=TwinTransport(two_core_mapping()),
            config_overrides={"odin_fpga_sample_count": 0})
        with pytest.raises(ValueError, match="odin_fpga_sample_count"):
            step.process()

    def test_a_ttfs_deployment_refuses_before_touching_the_device(self, monkeypatch):
        from integration.odin_fpga_harness import two_core_mapping

        transport = TwinTransport(two_core_mapping())
        _pipeline, step = prepare_step(
            monkeypatch, OdinFpgaDeploymentStep, transport=transport,
            config_overrides={
                "spiking_family": "ttfs", "spiking_variant": "analytical",
                "spiking_mode": "ttfs", "firing_granularity": "per_cycle",
                "membrane_bits": 0, "membrane_arithmetic": "unbounded",
            })
        with pytest.raises(ValueError):
            step.process()
        assert transport.opened == 0

    def test_a_diverging_device_fails_the_step_instead_of_reporting_a_number(
        self, monkeypatch,
    ):
        from integration.odin_fpga_harness import two_core_mapping

        class _Liar(TwinTransport):
            def run_samples(self, per_cycle_inputs, *, latencies):
                run = super().run_samples(per_cycle_inputs, latencies=latencies)
                counts = dict(run.counts)
                key = next(iter(sorted(counts)))
                counts[key] = counts[key] + 1
                return TransportRun(
                    counts=counts, samples=run.samples,
                    cycles_per_sample=run.cycles_per_sample,
                    wall_s=run.wall_s, program_wall_s=run.program_wall_s,
                    device_cycles=run.device_cycles, detail=run.detail)

        _pipeline, step = prepare_step(
            monkeypatch, OdinFpgaDeploymentStep,
            transport=_Liar(two_core_mapping()))
        with pytest.raises(AssertionError):
            step.process()


class TestTheFixtureDiscriminatesTheLaw:
    def test_the_producer_emits_more_than_one_spike_in_a_cycle(self):
        from integration.odin_fpga_harness import two_core_mapping

        mapping = two_core_mapping()
        trace = simulate_cycles(
            mapping, soma_law=ODIN_LAW,
            input_counts=[[1] * 4 for _ in range(TIMESTEPS)],
            simulation_length=TIMESTEPS,
            chip_latency=int(ChipLatency(mapping).calculate()))
        assert max(
            max(counts) for per_core in trace.outputs for counts in per_core) > 1
        assert all(len(row) == NEURONS for row in trace.window_counts())


def test_a_device_that_was_never_opened_refuses(monkeypatch):
    from integration.odin_fpga_harness import two_core_mapping
    from mimarsinan.chip_simulation.odin_fpga.cosim_transport import RtlCosimTransport

    del two_core_mapping
    with pytest.raises(DeviceTransportError):
        RtlCosimTransport(engine="iverilog").run_samples([], latencies=())
