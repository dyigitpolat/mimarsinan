"""[calculus §17/PR45] the nevresim spike-count certificate glue (Edge B)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.pipelining.pipeline_steps.verification.simulation_step import (
    _certify_nevresim_counts,
)


class _Flow(torch.nn.Module):
    """Stub flow: replays canned per-stage counts through the recorder seam."""

    def __init__(self, stage_counts):
        super().__init__()
        self._stage_counts = stage_counts
        self.stage_count_recorder = None
        self.lif_execution_synchronized = True

    def forward(self, x):
        for stage, counts in self._stage_counts:
            if self.stage_count_recorder is not None:
                self.stage_count_recorder(stage, counts)
        return x


def _stage(name):
    return SimpleNamespace(name=name)


def test_matching_counts_certify_exact(monkeypatch):
    import mimarsinan.pipelining.pipeline_steps.verification.simulation_step as mod

    counts = torch.tensor([[3.0, 0.0, 7.0], [1.0, 2.0, 0.0]])
    captured = [(_stage("seg0"), counts.numpy())]
    flow = _Flow([(_stage("seg0"), counts.clone())])
    monkeypatch.setattr(
        mod, "build_spiking_hybrid_flow", lambda pipeline, mapping, model: flow,
    )
    cert = _certify_nevresim_counts(
        pipeline=SimpleNamespace(config={"device": "cpu"}),
        mapping=None,
        captured=captured,
        samples=torch.zeros(2, 4),
    )
    assert cert.passed and cert.exact_match_fraction == 1.0
    assert flow.lif_execution_synchronized is False  # streaming = nevresim cell
    assert flow.stage_count_recorder is None


def test_count_mismatch_is_reported_not_fatal(monkeypatch):
    """Independent per-cycle timing (window-edge transients) is REPORTED —
    decision parity is this cell's arbiter [measured: 18% windows, parity 1.0]."""
    import mimarsinan.pipelining.pipeline_steps.verification.simulation_step as mod

    nev = torch.tensor([[3.0, 0.0, 7.0]])
    hcm = torch.tensor([[3.0, 1.0, 7.0]])
    captured = [(_stage("seg0"), nev.numpy())]
    flow = _Flow([(_stage("seg0"), hcm)])
    monkeypatch.setattr(
        mod, "build_spiking_hybrid_flow", lambda pipeline, mapping, model: flow,
    )
    cert = _certify_nevresim_counts(
        pipeline=SimpleNamespace(config={"device": "cpu"}),
        mapping=None,
        captured=captured,
        samples=torch.zeros(1, 4),
    )
    assert not cert.passed and cert.max_abs_delta == 1.0


def test_stage_count_recorder_seam_on_the_nevresim_hybrid_runner():
    from mimarsinan.chip_simulation.simulation_runner.hybrid import (
        SimulationHybridMixin,
    )

    assert hasattr(SimulationHybridMixin, "_run_hybrid")
    import inspect

    src = inspect.getsource(SimulationHybridMixin._run_hybrid)
    assert "stage_count_recorder" in src, (
        "the nevresim hybrid runner must consult the stage_count_recorder "
        "seam on raw (pre-decode) counts"
    )
    assert src.index("stage_count_recorder") < src.index("_raw_to_rates"), (
        "counts must be captured BEFORE the rate decode"
    )
