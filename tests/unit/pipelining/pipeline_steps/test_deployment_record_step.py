"""The terminal Deployment Record step: fabricated cache entries → sealed artifact.

Covers: the sealed ``deployment_record.json`` on disk (loads via
``DeploymentRecord.from_dict``), plan-dependent instance requires, fail-loud
on a missing required fragment (armed gate without boundary traffic; drifted
SCM figures), and the SANA-FE-off plan (record without energy, NO
``cost_record.json``).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from conftest import MockPipeline, default_config
from unit.deployment_record.record_fixtures import (
    PARAMS_PROGRAMMED_TOTAL,
    make_placement,
    make_read,
    make_sanafe_snapshot,
    make_schedule,
    make_utilization,
)

from mimarsinan.chip_simulation.cost_extraction import (
    COST_RECORD_FILENAME,
    FT_PASS_WALLS_FILENAME,
    load_cost_record,
)
from mimarsinan.deployment_record.schema import (
    DEPLOYMENT_RECORD_FILENAME,
    BoundaryTrafficRecord,
    ComputeOpRecord,
    DeploymentRecord,
    load_deployment_record,
)
from mimarsinan.deployment_record.cost.absolute import (
    DEPLOYMENT_COST_REPORT_FILENAME,
)
from mimarsinan.deployment_record.cost.terms import DeploymentCostReport
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_step import (
    DeploymentRecordStep,
)

_PLATFORM = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
    "weight_bits": 8,
    "schedule_policy": "pool",
    "max_schedule_passes": 8,
    "allow_scheduling": False,
    "cores_per_tile": 0,
    "tile_grid_rows": 0,
    "tile_grid_cols": 0,
    "cores_per_tile_resolved": 4,
    "tile_grid_rows_resolved": 2,
    "tile_grid_cols_resolved": 2,
}

TARGET_METRIC = 0.983


def _mapping():
    """A hard-core-mapping double matching the fixture schedule's stage census."""
    return SimpleNamespace(stages=[
        SimpleNamespace(kind="compute"),
        SimpleNamespace(kind="neural"),
        SimpleNamespace(kind="neural"),
    ])


def _scm_entry(*, params_reloaded: int = PARAMS_PROGRAMMED_TOTAL) -> dict:
    return {
        "reuse_plan": {
            "reprogram_passes": 1,
            "reuse_passes": 1,
            "params_reloaded": params_reloaded,
        },
        "relay_cores_inserted": 1,
        "ir_max_latency": 2,
    }


def _hcm_entry(*, with_boundaries: bool = False) -> dict:
    boundaries = None
    if with_boundaries:
        boundaries = [
            BoundaryTrafficRecord(
                node_id=7, producing_stage_index=1, neurons=50, samples=2,
                total_count=123, max_neuron_count=9,
            ).to_dict(),
        ]
    return {
        "schedule": make_schedule().to_dict(),
        "placement": make_placement(with_floorplan=False).to_dict(),
        "utilization": make_utilization().to_dict(),
        "weight_programming": {
            "neural_stages": 2,
            "programming_events": 1,
            "params_programmed": PARAMS_PROGRAMMED_TOTAL,
            "params_unique": PARAMS_PROGRAMMED_TOTAL,
        },
        "boundary_traffic": boundaries,
        "accuracy_reads": [make_read("hcm").to_dict()],
        "certificates": [],
    }


def _nevresim_entry() -> dict:
    return {
        "accuracy_reads": [make_read("nevresim").to_dict()],
        "total_spikes": 12345.0,
        "compute_stage_walls": [
            {"stage_index": 0, "name": "maxpool_0",
             "wall_s_total": 0.5, "invocations": 1},
        ],
    }


def _build(tmp_path, *, sanafe: bool, nevresim: bool = True,
           gate_samples: int = 0, hcm_entry: dict | None = None,
           scm_entry: dict | None = None, seed_sanafe: bool = True,
           physics_profile: str | None = None):
    config = default_config()
    config["enable_sanafe_simulation"] = sanafe
    config["enable_nevresim_simulation"] = nevresim
    config["spike_count_parity_samples"] = gate_samples
    pipeline = MockPipeline(
        config=config, working_directory=str(tmp_path / "run"),
    )
    os.makedirs(pipeline.working_directory, exist_ok=True)
    pipeline.set_target_metric(TARGET_METRIC)
    pipeline.seed("hard_core_mapping", _mapping())
    pipeline.seed("deployment_record_scm", scm_entry or _scm_entry())
    pipeline.seed("deployment_record_hcm", hcm_entry or _hcm_entry())
    platform = dict(_PLATFORM)
    if physics_profile is not None:
        # Through the REAL resolver, so the record carries what a run would carry.
        platform["platform_physics_resolved"] = build_platform_constraints_resolved({
            "cores": _PLATFORM["cores"],
            "platform_physics_profile": physics_profile,
        })["platform_physics_resolved"]
    pipeline.seed("platform_constraints_resolved", platform)
    if nevresim:
        pipeline.seed("deployment_record_nevresim", _nevresim_entry())
    if sanafe and seed_sanafe:
        snapshot = make_sanafe_snapshot()
        pipeline.seed(
            "sanafe_simulation_results",
            SimpleNamespace(to_snapshot_dict=lambda: snapshot),
        )
    step = DeploymentRecordStep(pipeline)
    pipeline.prepare_step(step)
    return pipeline, step


class TestPlanDependentRequires:
    def test_static_requires_are_the_documented_lower_bound(self):
        assert DeploymentRecordStep.REQUIRES == (
            "hard_core_mapping",
            "deployment_record_scm",
            "deployment_record_hcm",
            "platform_constraints_resolved",
        )
        assert DeploymentRecordStep.PROMISES == ("deployment_record",)

    def test_instance_requires_extend_per_plan(self, tmp_path):
        _pipeline, step = _build(tmp_path, sanafe=True, nevresim=True)
        assert "sanafe_simulation_results" in step.requires
        assert "deployment_record_nevresim" in step.requires

        _pipeline, step = _build(
            tmp_path / "b", sanafe=False, nevresim=False,
        )
        assert "sanafe_simulation_results" not in step.requires
        assert "deployment_record_nevresim" not in step.requires


class TestSealedArtifact:
    def test_sanafe_run_seals_record_and_projects_cost_record(self, tmp_path):
        pipeline, step = _build(tmp_path, sanafe=True)
        step.run()

        path = os.path.join(
            pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME,
        )
        assert os.path.exists(path)
        record = load_deployment_record(path)  # from_dict = seal-valid shape
        assert record.identity.cell_key == "lif@sanafe"
        assert record.identity.run_dir == pipeline.working_directory
        assert record.identity.platform == _PLATFORM
        assert record.accuracy.deployed.metric == TARGET_METRIC
        assert record.accuracy.deployed.kind == "carried"
        # The nevresim probe read joined the reads (seal requires it).
        assert any(r.backend == "nevresim" for r in record.accuracy.reads)
        # SANA-FE fragments landed.
        assert record.energy is not None
        assert record.energy.mj_per_sample == 2.0
        assert record.energy.total_spikes == 987
        assert record.traffic is not None and record.traffic.noc is not None
        assert record.traffic.noc.total_packets == 1234
        assert len(record.timing.per_segment) == 2
        assert record.timing.latency.compute_steps == 64
        floorplan = record.placement.floorplan
        assert floorplan is not None
        assert (floorplan.mesh_width, floorplan.mesh_height) == (2, 2)
        assert floorplan.cores_per_tile == 4
        assert floorplan.derivation == "derived"
        assert record.placement.tiles[0].core_indices == (0, 1)
        # Host-op wall folded onto the ComputeOp + summed into the latency.
        (compute_op,) = [
            s for s in record.schedule.stages
            if isinstance(s, ComputeOpRecord)
        ]
        assert compute_op.wall_s_total == 0.5
        assert record.timing.latency.host_ops_s == 0.5
        assert "includes NoC hop latency" in record.timing.latency.note
        # The cache promise re-types cleanly.
        entry = pipeline.cache["DeploymentRecordStep.deployment_record"]
        json.dumps(entry)
        assert DeploymentRecord.from_dict(entry).identity == record.identity

        # Legacy continuity: the projection landed next to the record.
        cost = load_cost_record(
            os.path.join(pipeline.working_directory, COST_RECORD_FILENAME)
        )
        assert cost.cell_key == "lif@sanafe"
        assert cost.acc_deploy == TARGET_METRIC
        assert cost.mj_per_sample == 2.0
        assert cost.latency_steps == 64
        assert cost.cores == 3
        assert (cost.reprogram_passes, cost.reuse_passes) == (1, 1)
        assert cost.params_reloaded == PARAMS_PROGRAMMED_TOTAL
        assert cost.provenance == {"run_dir": pipeline.working_directory}

        verdict = step.step_verdict()
        assert verdict is not None and verdict["status"] == "pass"

    def test_sanafe_off_plan_seals_without_energy_and_no_cost_record(
        self, tmp_path
    ):
        pipeline, step = _build(tmp_path, sanafe=False)
        step.run()
        record = load_deployment_record(os.path.join(
            pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME,
        ))
        assert record.energy is None
        assert record.traffic is None  # no boundaries, no NoC
        assert record.timing.per_segment == ()
        assert record.timing.latency.compute_sim_time_s is None
        assert record.placement.floorplan is None
        assert record.identity.cell_key == "lif@nevresim"
        assert not os.path.exists(os.path.join(
            pipeline.working_directory, COST_RECORD_FILENAME,
        ))

    def test_armed_gate_boundaries_land_in_the_record(self, tmp_path):
        pipeline, step = _build(
            tmp_path, sanafe=True, gate_samples=2,
            hcm_entry=_hcm_entry(with_boundaries=True),
        )
        step.run()
        record = load_deployment_record(os.path.join(
            pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME,
        ))
        assert record.traffic is not None
        assert record.traffic.boundaries is not None
        (boundary,) = record.traffic.boundaries
        assert boundary.total_count == 123

    def test_adaptation_fragment_reads_ft_pass_walls(self, tmp_path):
        pipeline, step = _build(tmp_path, sanafe=True)
        walls_path = os.path.join(
            pipeline.working_directory, FT_PASS_WALLS_FILENAME,
        )
        with open(walls_path, "w", encoding="utf-8") as fh:
            json.dump({
                "max_ft_pass_wall_s": 9.5,
                "passes": [
                    {"label": "LIF Adaptation/pass_0", "wall_s": 9.5},
                    {"label": "LIF Adaptation/pass_1", "wall_s": 4.0},
                ],
            }, fh)
        step.run()
        record = load_deployment_record(os.path.join(
            pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME,
        ))
        adaptation = record.adaptation
        assert adaptation is not None
        assert adaptation.max_ft_pass_wall_s == 9.5
        assert [w.label for w in adaptation.ft_pass_walls] == [
            "LIF Adaptation/pass_0", "LIF Adaptation/pass_1",
        ]
        cost = load_cost_record(os.path.join(
            pipeline.working_directory, COST_RECORD_FILENAME,
        ))
        assert cost.max_ft_pass_wall_s == 9.5


class TestFailLoud:
    def test_armed_gate_without_boundary_traffic_refuses_to_seal(
        self, tmp_path
    ):
        _pipeline, step = _build(
            tmp_path, sanafe=True, gate_samples=2,
            hcm_entry=_hcm_entry(with_boundaries=False),
        )
        with pytest.raises(ValueError, match="boundaries"):
            step.run()

    def test_drifted_scm_figures_refuse_to_seal(self, tmp_path):
        _pipeline, step = _build(
            tmp_path, sanafe=True,
            scm_entry=_scm_entry(params_reloaded=999),
        )
        with pytest.raises(ValueError, match="params_reloaded"):
            step.run()

    def test_metric_is_carried_never_measured(self, tmp_path):
        pipeline, step = _build(tmp_path, sanafe=True)
        step.run()
        assert step.validate() == TARGET_METRIC
        assert step.validate_metric_kind() == "carried"
        assert pipeline.get_target_metric() == TARGET_METRIC


class TestPhysicsReport:
    """The vendor-priced plane exists exactly when the run declared physics."""

    def _report_path(self, pipeline):
        return os.path.join(pipeline.working_directory,
                            DEPLOYMENT_COST_REPORT_FILENAME)

    def test_a_run_without_physics_writes_no_report(self, tmp_path):
        pipeline, step = _build(tmp_path, sanafe=True)
        step.run()
        assert not os.path.exists(self._report_path(pipeline))

    def test_a_declared_profile_writes_the_priced_report(self, tmp_path):
        pipeline, step = _build(tmp_path, sanafe=True, physics_profile="truenorth")
        step.run()
        with open(self._report_path(pipeline), encoding="utf-8") as handle:
            report = DeploymentCostReport.from_dict(json.load(handle))
        area = [term for term in report.area if term.name == "chip_area_mm2"]
        assert area, "the declared profile prices chip area"
        # 20 declared cores x 0.0936 mm2 + 46.6 mm2 of periphery.
        assert area[0].value == pytest.approx(20 * 0.0936 + 46.6)
        assert area[0].band is not None
        assert "akopyan2015truenorth" in area[0].source

    def test_the_measured_plane_is_untouched_by_the_priced_one(self, tmp_path):
        """Byte-identity where it matters: adding physics must not move a single
        measured number, only append predicted terms."""
        bare, bare_step = _build(tmp_path / "bare", sanafe=True)
        bare_step.run()
        priced, priced_step = _build(
            tmp_path / "priced", sanafe=True, physics_profile="truenorth")
        priced_step.run()

        def _record(pipeline):
            payload = json.load(open(
                os.path.join(pipeline.working_directory,
                             DEPLOYMENT_RECORD_FILENAME), encoding="utf-8"))
            # Only the run's own coordinates and the physics declaration itself may
            # differ; every measured/derived field must be identical.
            for key in ("platform", "created_at", "run_dir"):
                payload["identity"][key] = None
            return payload

        assert _record(bare) == _record(priced)

        def _cost(pipeline):
            payload = load_cost_record(
                os.path.join(pipeline.working_directory, COST_RECORD_FILENAME)
            ).to_dict()
            payload["provenance"] = None  # carries the run dir
            return payload

        assert _cost(bare) == _cost(priced)

    def test_refusals_are_visible_in_the_report(self, tmp_path):
        pipeline, step = _build(tmp_path, sanafe=True, physics_profile="truenorth")
        step.run()
        with open(self._report_path(pipeline), encoding="utf-8") as handle:
            report = DeploymentCostReport.from_dict(json.load(handle))
        assert any(note.startswith("refused ") for note in report.notes), (
            "what the physics could NOT price must be stated, not silently missing"
        )

    def test_a_no_sanafe_run_still_prices_what_it_can(self, tmp_path):
        """No measured plane at all: the priced terms stand alone."""
        pipeline, step = _build(tmp_path, sanafe=False, physics_profile="truenorth")
        step.run()
        with open(self._report_path(pipeline), encoding="utf-8") as handle:
            report = DeploymentCostReport.from_dict(json.load(handle))
        assert report.segments == ()
        assert [t.name for t in report.area] == ["chip_area_mm2"]
