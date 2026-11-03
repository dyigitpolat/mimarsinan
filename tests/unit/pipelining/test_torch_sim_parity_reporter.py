"""[U-LMP, U-HDA review m-4] The torch-model/deployed-sim parity announces its host arithmetic through the step's reporter.

``chip_simulation/deployed_host.py`` states that each evaluation on the deployed host prints its device and dtype and
emits a ``host_arithmetic`` reporter event. The torch-model/deployed-sim parity printed its line but called
``announce`` without a reporter, so no event reached the run's reporter. Soft Core Mapping now hands the gate its
pipeline's reporter, and the gate passes it to ``announce``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from conftest import MockPipeline, default_config


class _Recorder:
    def __init__(self) -> None:
        self.events = []

    def event(self, kind, payload) -> None:
        self.events.append((kind, dict(payload)))


def test_the_torch_sim_parity_emits_its_host_arithmetic_event_through_a_reporter():
    """Red before: the gate takes no reporter (TypeError), so its ``[HostArithmetic]`` line had no event."""
    from mimarsinan.pipelining.core.nf_scm_parity import assert_torch_vs_deployed_sim_parity_or_raise

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
    flow = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
    flow.load_state_dict(model.state_dict())
    reporter = _Recorder()
    agreement = assert_torch_vs_deployed_sim_parity_or_raise(model, flow, torch.randn(16, 5), reporter=reporter)
    assert agreement == 1.0
    kinds = [kind for kind, _ in reporter.events]
    assert kinds == ["host_arithmetic"], kinds
    payload = reporter.events[0][1]
    assert payload["device"] == "cpu" and "torch-model" in payload["evaluation"]


def test_soft_core_mapping_hands_its_reporter_to_the_torch_sim_parity(monkeypatch):
    """Red before: Soft Core Mapping called the gate without its reporter."""
    import mimarsinan.pipelining.core.nf_scm_parity as parity_mod
    import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as step_mod
    from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep

    captured = {}

    def _capture(reference, flow, samples, **kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(step_mod, "build_deployment_contract", lambda pipeline: object())
    monkeypatch.setattr(parity_mod, "torch_sim_parity_enabled", lambda contract: True)
    monkeypatch.setattr(step_mod, "build_identity_mapping_for_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(step_mod, "build_spiking_hybrid_flow", lambda *a, **k: object())
    monkeypatch.setattr(parity_mod, "torch_parity_reference", lambda model: model)
    monkeypatch.setattr(parity_mod, "assert_torch_vs_deployed_sim_parity_or_raise", _capture)
    pipeline = MockPipeline(config=default_config())
    pipeline.reporter = _Recorder()
    step = SoftCoreMappingStep(pipeline)
    step.trainer = SimpleNamespace(iter_validation_batches=lambda n: iter([(torch.rand(4, 8), None)]))
    step._run_torch_sim_parity_check(model=object(), ir_graph=object())
    assert captured.get("reporter") is pipeline.reporter
