"""[calculus §17/PR44] the deployed spike-count certificate gate adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.pipelining.core.spike_count_gate import (
    run_spike_count_certificate_gate,
)


def _pipeline(config):
    return SimpleNamespace(config=config)


def test_gate_skips_non_lif_modes_without_touching_the_mapping():
    pipeline = _pipeline({"spiking_mode": "ttfs", "spike_count_parity_samples": 2})
    result = run_spike_count_certificate_gate(
        pipeline, model=None, ir_graph=None, hybrid_mapping=None,
    )
    assert result is None


def test_gate_disabled_at_zero_samples():
    pipeline = _pipeline({"spiking_mode": "lif", "spike_count_parity_samples": 0})
    result = run_spike_count_certificate_gate(
        pipeline, model=None, ir_graph=None, hybrid_mapping=None,
    )
    assert result is None


def test_gate_raises_on_failed_certificate(monkeypatch):
    import mimarsinan.pipelining.core.spike_count_gate as gate_mod

    pipeline = _pipeline({
        "spiking_mode": "lif", "spike_count_parity_samples": 1,
        "device": "cpu",
    })

    import torch

    failed = SimpleNamespace(
        passed=False,
        divergent=[],
        summary=lambda: "spike-count certificate [hcm/exact]: FAIL",
    )
    monkeypatch.setattr(
        gate_mod, "_certificate_samples", lambda pipeline, model, n: torch.zeros(1, 2),
    )
    monkeypatch.setattr(
        gate_mod, "build_identity_mapping_for_pipeline",
        lambda ir_graph, pipeline_config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        gate_mod, "build_spiking_hybrid_flow",
        lambda pipeline, mapping, model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        gate_mod, "certify_twin_flow_counts",
        lambda ir, ref_flow, backend_flow, samples, backend, discipline,
            **kw: (failed, "detail"),
    )
    with pytest.raises(RuntimeError, match="spike-count certificate"):
        run_spike_count_certificate_gate(
            pipeline,
            model=SimpleNamespace(get_mapper_repr=lambda: None),
            ir_graph=None,
            hybrid_mapping=None,
        )
