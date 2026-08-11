"""Wiring pins for the V9 tie-seam fix: the chip probe and the certificate twin
both execute inside the measurement plane, on the pipeline device. The tie
SEMANTICS are pinned elsewhere; these tests pin the load-bearing WIRING (a
mutated-out plane wrap or device move must fail here)."""

import pytest

from mimarsinan.models.nn.lif_kernels import in_measurement_plane


class _PlaneRecordingMapping:
    """Duck mapping whose 'flat run' records the plane flag at execution time."""

    def __init__(self, sink):
        self._sink = sink


class _PlaneRecordingRunner:
    pass


def test_simulation_runner_run_executes_inside_the_measurement_plane(monkeypatch):
    from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner

    seen = {}
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.mapping = object()  # not a HybridHardCoreMapping -> flat path
    monkeypatch.setattr(
        SimulationRunner, "_run_flat_mapping",
        lambda self, mapping: seen.update(in_plane=in_measurement_plane()) or 0.0,
    )
    runner.run()
    assert seen["in_plane"] is True


class _DuckSamples:
    """Records .to(device); ducks the tensor surface the cert path touches."""

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _RecordingFlow:
    def __init__(self):
        self.stage_count_recorder = None
        self.lif_execution_synchronized = None
        self.called_in_plane = None

    def __call__(self, samples):
        self.called_in_plane = in_measurement_plane()


def test_certificate_twin_joins_plane_and_pipeline_device(monkeypatch):
    from mimarsinan.pipelining.pipeline_steps.verification import simulation_step

    flow = _RecordingFlow()
    monkeypatch.setattr(
        simulation_step, "build_spiking_hybrid_flow", lambda *a, **k: flow,
    )

    class _Cert:
        exact_match_fraction = 1.0
        max_abs_delta = 0.0
        neuron_windows_compared = 0
        passed = True

    monkeypatch.setattr(
        simulation_step, "certify_spike_counts", lambda *a, **k: _Cert(),
    )

    class _Plan:
        weight_quantization = False

    monkeypatch.setattr(
        simulation_step.DeploymentPlan, "of", classmethod(lambda cls, p: _Plan()),
    )
    samples = _DuckSamples()

    class _Pipeline:
        config = {"device": "cpu"}

    simulation_step._certify_nevresim_counts(
        pipeline=_Pipeline(), mapping=object(), captured={}, samples=samples,
    )
    assert samples.moved_to == "cpu"
    assert flow.called_in_plane is True
