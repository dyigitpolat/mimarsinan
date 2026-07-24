"""[C2] the membrane-readout diagnostic is a REPORT: its cost must be bounded
regardless of the validation batch size (measured: two full-batch deployed
flows with walk-sourced boundary trains ate ~an hour of the ViT SCM step)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import mimarsinan.pipelining.core.simulation_factory as sf


class _Flow(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self._seen = seen

    def forward(self, x):
        self._seen.append(int(x.shape[0]))
        return torch.zeros(x.shape[0], 3)


def test_diagnostic_samples_are_capped(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(
        sf, "build_spiking_hybrid_flow",
        lambda pipeline, mapping, model=None, membrane_readout_decode=False:
            _Flow(seen),
    )
    monkeypatch.setattr(
        sf, "build_deployment_contract",
        lambda pipeline: SimpleNamespace(spiking_mode="lif"),
    )
    monkeypatch.setattr(sf, "deployed_membrane_decode_enabled", lambda p: True)
    monkeypatch.setattr(sf, "final_only_output_nodes", lambda m: [])
    monkeypatch.setattr(sf, "emit_reporter_event", lambda *a, **k: None)

    pipeline = SimpleNamespace(
        config={"device": "cpu", "lif_membrane_readout": True},
        reporter=None,
    )
    stats = sf.run_membrane_readout_diagnostic(
        pipeline, hybrid_mapping=None, samples=torch.zeros(64, 4),
    )
    assert stats is not None
    assert stats["samples"] == sf.MEMBRANE_DIAGNOSTIC_MAX_SAMPLES
    assert seen and all(n <= sf.MEMBRANE_DIAGNOSTIC_MAX_SAMPLES for n in seen)
