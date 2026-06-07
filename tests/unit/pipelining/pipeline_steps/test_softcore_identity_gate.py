"""Rungs 2/3 of the gate ladder: SCM measures the identity mapping, HCM the packed."""

import numpy as np
import pytest

from conftest import MockPipeline, make_tiny_ir_graph

from mimarsinan.pipelining.core import simulation_factory


class _Recorder:
    def __init__(self, value=0.5):
        self.calls = []
        self.value = value

    def __call__(self, pipeline, flow, **kwargs):
        self.calls.append(flow.hybrid_mapping)
        return self.value


@pytest.fixture
def pipeline():
    p = MockPipeline()
    p.config["spiking_mode"] = "ttfs_quantized"
    p.config["firing_mode"] = "TTFS"
    p.config["spike_generation_mode"] = "TTFS"
    p.config["thresholding_mode"] = "<="

    class _Cache(dict):
        def add(self, key, obj, strategy="basic"):
            self[key] = obj

    p.cache = _Cache()
    return p


def test_run_scm_identity_metric_builds_identity_mapping(pipeline, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(simulation_factory, "run_hcm_spiking_test", recorder)

    ir_graph = make_tiny_ir_graph()
    platform_constraints = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}]}
    acc = simulation_factory.run_scm_identity_metric(
        pipeline, ir_graph, platform_constraints,
    )
    assert acc == pytest.approx(0.5)
    (mapping,) = recorder.calls
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            assert len(placements) == 1, "SCM gate must run on the identity mapping"


def test_scm_identity_metric_does_not_cache_packed_mapping(pipeline, monkeypatch):
    monkeypatch.setattr(simulation_factory, "run_hcm_spiking_test", _Recorder())

    ir_graph = make_tiny_ir_graph()
    platform_constraints = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}]}
    simulation_factory.run_scm_identity_metric(pipeline, ir_graph, platform_constraints)
    assert "hybrid_mapping" not in pipeline.cache, (
        "rung-2 must not seed the packed mapping; HCM (rung 3) builds it itself"
    )


def test_hcm_metric_still_runs_packed_mapping(pipeline, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(simulation_factory, "run_hcm_spiking_test", recorder)

    ir_graph = make_tiny_ir_graph()
    platform_constraints = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}]}
    simulation_factory.run_hcm_mapping_metric(pipeline, ir_graph, platform_constraints)
    (mapping,) = recorder.calls
    assert "hybrid_mapping" in pipeline.cache
    packed_cores = [c for s in mapping.stages if s.kind == "neural"
                    for c in s.hard_core_mapping.cores]
    assert any(
        any(src.is_off_ for src in core.axon_sources) for core in packed_cores
    ), "rung 3 must keep measuring the packed mapping"


def test_soft_core_mapping_step_uses_identity_gate(monkeypatch):
    import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as scm_mod

    assert hasattr(scm_mod, "run_scm_identity_metric"), (
        "SoftCoreMappingStep must import the rung-2 identity gate"
    )
    assert not hasattr(scm_mod, "run_hcm_mapping_metric"), (
        "SoftCoreMappingStep must no longer run the packed-mapping metric "
        "(that is HardCoreMappingStep's rung-3 gate)"
    )
