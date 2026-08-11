"""The producing steps persist their deployment-record fragments through the cache.

SCM promises ``deployment_record_scm`` (reuse plan + relay count + IR latency
census); HCM promises ``deployment_record_hcm`` (schedule/placement/utilization
fragments, weight-programming totals, boundary traffic when the gate ran, the
accuracy read); the nevresim Simulation step promises
``deployment_record_nevresim`` (probe read, driver total-spikes figure,
StageTimer host-op walls). Entries are JSON-safe dicts that re-type via
``from_dict`` cleanly, and the steps' existing stdout/artifacts stay
byte-identical.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch.nn as nn

from conftest import MockPipeline, default_config, make_tiny_ir_graph, make_tiny_supermodel

from mimarsinan.deployment_record.schema import (
    AccuracyReadRecord,
    BoundaryTrafficRecord,
    CertificateRecord,
    PlacementRecord,
    ScheduleRecord,
    UtilizationRecord,
)
from mimarsinan.mapping.crossbar_utilization import UTILIZATION_RECORD_FILENAME
from mimarsinan.mapping.weight_reuse import (
    format_weight_reuse_summary,
    weight_reuse_plan_from_graph,
)
from mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step import (
    HardCoreMappingStep,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)

_PLATFORM = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
    "weight_bits": 8,
}


# ── Soft Core Mapping ────────────────────────────────────────────────────


def _run_soft_core_mapping(tmp_path):
    pipeline = MockPipeline(
        config=default_config(), working_directory=str(tmp_path / "run"),
    )
    pipeline.config["onchip_majority_gate"] = False
    pipeline.config["scm_torch_sim_parity_check"] = False
    model = make_tiny_supermodel()
    for perceptron in model.get_perceptrons():
        perceptron.normalization = nn.Identity()
    pipeline.seed("fused_model", model, step_name="Normalization Fusion")
    pipeline.seed(
        "platform_constraints_resolved", dict(_PLATFORM),
        step_name="Model Configuration",
    )
    step = SoftCoreMappingStep(pipeline)
    step.name = "Soft Core Mapping"
    pipeline.prepare_step(step)
    step.run()
    return pipeline


class TestSoftCoreMappingEntry:
    def test_scm_fragment_lands_json_safe_and_consistent(self, tmp_path, capsys):
        pipeline = _run_soft_core_mapping(tmp_path)
        entry = pipeline.cache["Soft Core Mapping.deployment_record_scm"]
        json.dumps(entry)  # cache 'basic' strategy = JSON; must round-trip
        assert set(entry) == {
            "reuse_plan", "relay_cores_inserted", "ir_max_latency",
        }
        ir_graph = pipeline.cache["Soft Core Mapping.ir_graph"]
        plan = weight_reuse_plan_from_graph(ir_graph)
        assert entry["reuse_plan"] == {
            "reprogram_passes": plan.reprogram_passes,
            "reuse_passes": plan.reuse_passes,
            "params_reloaded": plan.params_reloaded,
        }
        # Relays are gated off by default: the count is honestly zero.
        assert entry["relay_cores_inserted"] == 0
        assert entry["ir_max_latency"] >= 1

        out = capsys.readouterr().out
        # Byte-identity of the existing prints: the reuse-report line and the
        # IR-latency line keep their exact shapes.
        assert (
            "[SoftCoreMappingStep] Weight-reuse schedule: "
            + format_weight_reuse_summary(plan)
        ) in out
        assert (
            f"[SoftCoreMappingStep] IR Graph max latency: "
            f"{entry['ir_max_latency']}"
        ) in out


# ── Hard Core Mapping ────────────────────────────────────────────────────


class _Cache(dict):
    def add(self, key, obj, strategy="basic"):
        self[key] = obj


def _hcm_pipeline(tmp_path):
    config = default_config()
    config["core_semantics"] = "mvm"
    pipeline = MockPipeline(config=config, working_directory=str(tmp_path))
    pipeline.cache = _Cache()
    return pipeline


_SCM_ENTRY = {
    "reuse_plan": {
        "reprogram_passes": 2, "reuse_passes": 0, "params_reloaded": 17,
    },
    "relay_cores_inserted": 3,
    "ir_max_latency": 2,
}


def _run_hard_core_mapping(pipeline, monkeypatch, *, gate_result=None):
    import mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step as hcm

    monkeypatch.setattr(
        hcm, "run_spike_count_certificate_gate", lambda *a, **k: gate_result,
    )
    monkeypatch.setattr(hcm, "run_value_twin_certificate_gate", lambda *a, **k: None)
    monkeypatch.setattr(hcm, "run_value_mapping_metric", lambda *a, **k: 0.875)

    pipeline.seed("model", object())
    pipeline.seed("ir_graph", make_tiny_ir_graph())
    pipeline.seed("platform_constraints_resolved", dict(_PLATFORM))
    pipeline.seed("deployment_record_scm", dict(_SCM_ENTRY))
    step = HardCoreMappingStep(pipeline)
    pipeline.prepare_step(step)
    step.run()
    return step


def _fake_gate_result():
    cert = SimpleNamespace(
        backend="hcm", passed=True, neuron_windows_compared=16,
        exact_match_fraction=1.0, max_abs_delta=0.0,
        summary=lambda: "spike-count certificate [hcm/exact]: PASS",
    )
    boundaries = (
        BoundaryTrafficRecord(
            node_id=0, producing_stage_index=0, neurons=4, samples=2,
            total_count=12, max_neuron_count=3,
        ),
    )
    return cert, boundaries


class TestHardCoreMappingEntry:
    def test_fragments_retype_cleanly_from_the_cache(self, tmp_path, monkeypatch):
        pipeline = _hcm_pipeline(tmp_path)
        _run_hard_core_mapping(pipeline, monkeypatch)
        entry = pipeline.cache["HardCoreMappingStep.deployment_record_hcm"]
        json.dumps(entry)
        schedule = ScheduleRecord.from_dict(entry["schedule"])
        placement = PlacementRecord.from_dict(entry["placement"])
        utilization = UtilizationRecord.from_dict(entry["utilization"])
        assert schedule.params_reloaded == 17  # threaded from the SCM entry
        assert utilization.relay_cores_inserted == 3
        assert len(placement.softcores) > 0
        assert sum(
            s.params_programmed for s in schedule.segments()
        ) == entry["weight_programming"]["params_programmed"]
        (read,) = [
            AccuracyReadRecord.from_dict(r) for r in entry["accuracy_reads"]
        ]
        assert read.metric == 0.875
        assert read.backend == "value_census"  # mvm plan observes values
        assert read.kind == "measured"

    def test_gate_skip_leaves_boundary_traffic_none(self, tmp_path, monkeypatch):
        pipeline = _hcm_pipeline(tmp_path)
        _run_hard_core_mapping(pipeline, monkeypatch)
        entry = pipeline.cache["HardCoreMappingStep.deployment_record_hcm"]
        assert entry["boundary_traffic"] is None
        assert entry["certificates"] == []

    def test_gate_pass_persists_boundary_traffic_and_certificate(
        self, tmp_path, monkeypatch
    ):
        pipeline = _hcm_pipeline(tmp_path)
        _run_hard_core_mapping(
            pipeline, monkeypatch, gate_result=_fake_gate_result(),
        )
        entry = pipeline.cache["HardCoreMappingStep.deployment_record_hcm"]
        json.dumps(entry)
        (boundary,) = [
            BoundaryTrafficRecord.from_dict(b) for b in entry["boundary_traffic"]
        ]
        assert boundary.total_count == 12
        (certificate,) = [
            CertificateRecord.from_dict(c) for c in entry["certificates"]
        ]
        assert certificate.name == "spike_count_streaming_twin"
        assert certificate.passed is True
        assert certificate.neuron_windows_compared == 16

    def test_existing_artifacts_and_prints_stay_intact(
        self, tmp_path, monkeypatch, capsys
    ):
        pipeline = _hcm_pipeline(tmp_path)
        _run_hard_core_mapping(pipeline, monkeypatch)
        # The pre-existing surfaces are untouched: mapping cached, utilization
        # record written, summary lines printed exactly as before.
        assert "hybrid_mapping" in pipeline.cache
        assert (tmp_path / UTILIZATION_RECORD_FILENAME).exists()
        out = capsys.readouterr().out
        assert "[WeightProgramming]" in out
        assert "[Crossbar]" in out
        assert "[HardCoreMappingStep] Hard-core Spiking Simulation Test: 0.875" in out


# ── nevresim Simulation ──────────────────────────────────────────────────


class _FakeSimulationRunner:
    """Stands in for ``SimulationRunner``: records the opted-in stage timer,
    times one host ComputeOp through it, and surfaces the driver figures."""

    def __init__(self, pipeline, mapping, simulation_length, preprocessor=None,
                 stage_timer=None):
        self.stage_timer = stage_timer
        self.test_data = [object()] * 7
        self.nevresim_total_spikes = None

    def run(self):
        assert self.stage_timer is not None  # the step opts in
        with self.stage_timer.time_compute_stage(1, "avg_pool"):
            pass
        self.nevresim_total_spikes = 12345.0
        return 0.5


def _run_simulation_step(tmp_path, monkeypatch):
    import mimarsinan.pipelining.pipeline_steps.verification.simulation_step as sim

    monkeypatch.setattr(sim, "SimulationRunner", _FakeSimulationRunner)
    pipeline = MockPipeline(
        config=default_config(), working_directory=str(tmp_path),
    )
    pipeline.seed("hard_core_mapping", object())
    step = sim.SimulationStep(pipeline)
    pipeline.prepare_step(step)
    step.run()
    return pipeline, step


class TestSimulationStepEntry:
    def test_nevresim_fragment_lands_json_safe_with_typed_read(
        self, tmp_path, monkeypatch, capsys
    ):
        pipeline, step = _run_simulation_step(tmp_path, monkeypatch)
        entry = pipeline.cache["SimulationStep.deployment_record_nevresim"]
        json.dumps(entry)  # cache 'basic' strategy = JSON; must round-trip
        assert set(entry) == {
            "accuracy_reads", "total_spikes", "compute_stage_walls",
        }
        (read,) = [
            AccuracyReadRecord.from_dict(r) for r in entry["accuracy_reads"]
        ]
        assert read.metric == 0.5
        assert read.backend == "nevresim"
        assert read.samples == 7  # the probe's actual sample count
        assert read.kind == "measured"
        assert read.step == "SimulationStep"
        # The formerly printed-and-dropped driver figure is persisted.
        assert entry["total_spikes"] == 12345.0
        (wall,) = entry["compute_stage_walls"]
        assert wall["stage_index"] == 1
        assert wall["name"] == "avg_pool"
        assert wall["invocations"] == 1
        assert wall["wall_s_total"] >= 0.0
        # The pre-existing probe print survives byte-identically.
        assert "Simulation accuracy:  0.5" in capsys.readouterr().out

    def test_step_verdict_is_untouched_by_the_emission(
        self, tmp_path, monkeypatch
    ):
        _pipeline, step = _run_simulation_step(tmp_path, monkeypatch)
        verdict = step.step_verdict()
        assert verdict["status"] == "pass"
        assert verdict["detail"]["probe_accuracy"] == 0.5
        assert verdict["detail"]["spike_count_certificate"] is None
