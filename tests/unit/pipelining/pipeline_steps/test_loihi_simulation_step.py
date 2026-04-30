from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from conftest import MockPipeline, MockDataProviderFactory, make_tiny_supermodel
from mimarsinan.pipelining.pipeline_steps.loihi_simulation_step import (
    LoihiSimulationStep,
)


class _RecordingReporter:
    def __init__(self):
        self.events = []

    def report(self, name, value):
        self.events.append((name, value))


class _FakeDataLoaderFactory:
    def __init__(self, data_provider_factory, num_workers=4):
        self.data_provider_factory = data_provider_factory
        self.num_workers = num_workers

    def create_data_provider(self):
        return self.data_provider_factory.create()

    def create_test_loader(self, batch_size, data_provider):
        dataset = data_provider._get_test_dataset()
        xs = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
        ys = torch.tensor([int(dataset[i][1]) for i in range(len(dataset))])
        return [(xs, ys)]


def _fake_record(sample_index=0):
    return SimpleNamespace(
        sample_index=sample_index,
        T=4,
        segments={
            0: SimpleNamespace(
                cores=[SimpleNamespace(), SimpleNamespace()],
            ),
        },
    )


def _prepare_step(monkeypatch, *, diffs=None):
    import mimarsinan.pipelining.pipeline_steps.loihi_simulation_step as loihi_step

    calls = {
        "hcm_samples": [],
        "runner_inits": [],
        "runner_run_called": False,
        "segments_from_reference": 0,
    }

    class FakeHCM:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def forward_with_recording(self, sample, sample_index=0):
            calls["hcm_samples"].append(sample.detach().cpu().clone())
            return torch.zeros((1, 1)), _fake_record(sample_index=sample_index)

    class FakeRunner:
        def __init__(self, pipeline, mapping, simulation_length, *, thresholding_mode="<"):
            calls["runner_inits"].append(
                {
                    "pipeline": pipeline,
                    "mapping": mapping,
                    "simulation_length": simulation_length,
                    "thresholding_mode": thresholding_mode,
                }
            )

        def run(self):
            calls["runner_run_called"] = True
            raise AssertionError("LoihiSimulationStep must not call run()")

        def run_segments_from_reference(self, ref):
            calls["segments_from_reference"] += 1
            return ref

    monkeypatch.setattr(loihi_step, "DataLoaderFactory", _FakeDataLoaderFactory)
    monkeypatch.setattr(loihi_step, "SpikingHybridCoreFlow", FakeHCM)
    monkeypatch.setattr(loihi_step, "LavaLoihiRunner", FakeRunner)
    monkeypatch.setattr(loihi_step, "compare_records", lambda _ref, _actual: diffs or [])
    monkeypatch.setattr(loihi_step, "format_first_diff", lambda _diffs: "formatted spike diff")

    pipeline = MockPipeline(
        data_provider_factory=MockDataProviderFactory(input_shape=(1, 8, 8), size=3),
    )
    pipeline.config["spiking_mode"] = "lif"
    pipeline.config["simulation_steps"] = 4
    pipeline.config["thresholding_mode"] = "<"
    pipeline.config["loihi_parity_sample_index"] = 2
    pipeline.reporter = _RecordingReporter()
    pipeline.set_target_metric(0.625)
    pipeline.seed("model", make_tiny_supermodel(), step_name="Model Configuration")
    pipeline.seed("hard_core_mapping", object(), step_name="Hard Core Mapping")

    step = LoihiSimulationStep(pipeline)
    step.name = "Loihi Simulation"
    pipeline.prepare_step(step)
    return step, pipeline, calls


def test_loihi_simulation_step_runs_one_sample_spike_parity(monkeypatch):
    step, pipeline, calls = _prepare_step(monkeypatch)

    step.run()

    assert calls["runner_run_called"] is False
    assert calls["segments_from_reference"] == 1
    assert calls["runner_inits"] == [
        {
            "pipeline": None,
            "mapping": pipeline.cache["Hard Core Mapping.hard_core_mapping"],
            "simulation_length": 4,
            "thresholding_mode": "<",
        }
    ]
    assert len(calls["hcm_samples"]) == 1
    assert calls["hcm_samples"][0].shape[0] == 1
    assert step.validate() == 0.625
    assert pipeline.reporter.events == [("Loihi Spike Parity", 1.0)]


def test_loihi_simulation_step_fails_with_formatted_record_diff(monkeypatch):
    step, _pipeline, _calls = _prepare_step(monkeypatch, diffs=["diff"])

    with pytest.raises(AssertionError, match="formatted spike diff"):
        step.run()
