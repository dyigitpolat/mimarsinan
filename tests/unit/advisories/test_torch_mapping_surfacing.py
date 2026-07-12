"""Graph-advisory wiring: TorchMappingStep surfaces advisories on the converted DAG."""

import torch

from conftest import MockPipeline, default_config

from mimarsinan.advisories.surfacing import ADVISORY_EVENT_KIND
from mimarsinan.models.deep_mlp import DeepMLP
from mimarsinan.pipelining.pipeline_steps.config.torch_mapping_step import (
    TorchMappingStep,
)


class RecordingReporter:
    prefix = ""

    def __init__(self):
        self.events = []

    def report(self, metric_name, metric_value, step=None):
        pass

    def console_log(self, metric_name, metric_value):
        pass

    def event(self, kind, payload):
        self.events.append((kind, payload))

    def finish(self):
        pass


def _run_step(depth: int, simulation_steps: int):
    cfg = default_config()
    cfg.update({
        "spiking_mode": "lif",
        "simulation_steps": simulation_steps,
        "activation_quantization": True,
    })
    pipeline = MockPipeline(config=cfg)
    reporter = RecordingReporter()
    pipeline.reporter = reporter

    torch.manual_seed(0)
    model = DeepMLP(
        input_shape=cfg["input_shape"],
        num_classes=cfg["num_classes"],
        depth=depth,
        width=8,
    ).eval()
    pipeline.seed("model", model)
    step = TorchMappingStep(pipeline)
    pipeline.prepare_step(step)
    step.run()
    return reporter


def _advisory_payloads(reporter):
    return [p for k, p in reporter.events if k == ADVISORY_EVENT_KIND]


class TestTorchMappingAdvisorySurfacing:
    def test_deep_norm_free_chain_emits_graph_advisories(self, capsys):
        reporter = _run_step(depth=7, simulation_steps=4)
        payloads = _advisory_payloads(reporter)
        ids = {p["id"] for p in payloads}
        assert "ADV-STAIRCASE-DEPTH" in ids
        assert "ADV-NORMFREE-CHAIN" in ids
        assert all(p["context"] == "torch_mapping" for p in payloads)
        out = capsys.readouterr().out
        assert "[ADVISORY][RISK] ADV-STAIRCASE-DEPTH" in out
        assert "[ADVISORY][RISK] ADV-NORMFREE-CHAIN" in out

    def test_shallow_chain_at_high_s_stays_silent(self, capsys):
        reporter = _run_step(depth=2, simulation_steps=16)
        assert _advisory_payloads(reporter) == []
        assert "[ADVISORY]" not in capsys.readouterr().out
