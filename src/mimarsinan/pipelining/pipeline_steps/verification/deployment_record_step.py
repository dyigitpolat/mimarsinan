"""The terminal Deployment Record step: seal the run's one deployment artifact.

Joins the fragments the producing steps already persisted through the cache
(``deployment_record_scm`` / ``deployment_record_hcm`` /
``deployment_record_nevresim`` / ``sanafe_simulation_results``) into the
sealed ``deployment_record.json`` (schema §2.7), and writes the legacy
``cost_record.json`` as a projection IFF the SANA-FE fragment is present —
the exact conditional the deleted ``SanafeSimulationStep._emit_cost_record``
had. Metric-neutral; FAIL LOUD — a record that cannot seal is a defect.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Tuple

from mimarsinan.chip_simulation.cost_extraction import save_cost_record
from mimarsinan.deployment_record.build.builder import DeploymentRecordBuilder
from mimarsinan.deployment_record.build.from_simulators import (
    depth_from_sanafe,
    energy_record_from_sanafe,
    floorplan_from_sanafe,
    noc_traffic_from_sanafe,
    s_global_from_sanafe,
    segment_timings_from_sanafe,
    tiles_from_sanafe,
)
from mimarsinan.deployment_record.cost import cost_record_from_deployment_record, emit_physics_report
from mimarsinan.deployment_record.schema import (
    AccuracyReadRecord,
    AccuracyRecord,
    BoundaryTrafficRecord,
    CertificateRecord,
    PlacementRecord,
    Provenance,
    ScheduleRecord,
    TrafficRecord,
    UtilizationRecord,
    save_deployment_record,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import (
    adaptation_from_run_dir,
    floorplan_derivation,
    fold_compute_walls,
    host_ops_wall_s,
    host_ops_wall_s_per_pass,
    identity_from_pipeline,
    nevresim_applies,
    resolved_step_names,
    seal_view_for,
    timing_record,
)

STEP_NAME = "Deployment Record"


def _prov(kind: str, producer: str, step: str, detail: str = "") -> Provenance:
    return Provenance(kind=kind, producer=producer, step=step, detail=detail)


class DeploymentRecordStep(PipelineStep):
    """Assemble, seal, and persist the run's ``deployment_record.json``."""

    # Static lower bound for the assembly-time DAG check; the plan-dependent
    # simulator fragments extend the INSTANCE contract in __init__ (the
    # documented ``declared_contract`` seam in core/steps/pipeline_step.py).
    REQUIRES = (
        "hard_core_mapping",
        "deployment_record_scm",
        "deployment_record_hcm",
        "platform_constraints_resolved",
    )
    PROMISES = ("deployment_record",)

    def __init__(self, pipeline):
        requires = list(self.REQUIRES)
        plan = DeploymentPlan.of(pipeline)
        if plan.enable_sanafe_simulation:
            requires.append("sanafe_simulation_results")
        if nevresim_applies(plan):
            requires.append("deployment_record_nevresim")
        super().__init__(requires, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.metric = None

    def validate(self):
        if self.metric is not None:
            return self.metric
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        # The record is a join of measurements, never a new metric read.
        return METRIC_CARRIED

    def process(self):
        plan = DeploymentPlan.of(self.pipeline)
        mapping = self.get_entry("hard_core_mapping")
        scm = self.get_entry("deployment_record_scm")
        hcm = self.get_entry("deployment_record_hcm")
        platform = self.get_entry("platform_constraints_resolved")

        schedule = ScheduleRecord.from_dict(hcm["schedule"])
        placement = PlacementRecord.from_dict(hcm["placement"])
        utilization = UtilizationRecord.from_dict(hcm["utilization"])
        self._cross_check_sources(schedule, scm, mapping)

        reads: List[AccuracyReadRecord] = [
            AccuracyReadRecord.from_dict(r) for r in hcm["accuracy_reads"]
        ]
        certificates = tuple(
            CertificateRecord.from_dict(c) for c in hcm["certificates"]
        )
        boundaries = self._boundary_records(hcm)

        walls: List[Any] = []
        if "deployment_record_nevresim" in self.requires:
            nevresim = self.get_entry("deployment_record_nevresim")
            reads += [
                AccuracyReadRecord.from_dict(r)
                for r in nevresim["accuracy_reads"]
            ]
            walls = list(nevresim["compute_stage_walls"])
        schedule = fold_compute_walls(schedule, walls)

        energy = None
        noc = None
        per_segment: Tuple[Any, ...] = ()
        if "sanafe_simulation_results" in self.requires:
            snapshot = self.get_entry("sanafe_simulation_results").to_snapshot_dict()
            energy = energy_record_from_sanafe(snapshot)
            per_segment = segment_timings_from_sanafe(snapshot)
            noc = noc_traffic_from_sanafe(snapshot)
            placement = replace(
                placement,
                floorplan=floorplan_from_sanafe(
                    snapshot,
                    cores_per_tile=int(platform["cores_per_tile_resolved"]),
                    derivation=floorplan_derivation(platform),
                ),
                tiles=tiles_from_sanafe(snapshot),
            )
            s_global = s_global_from_sanafe(snapshot)
            depth = depth_from_sanafe(snapshot)
        else:
            s_global = int(self.pipeline.config["simulation_steps"])
            depth = len(schedule.segments())

        timing = timing_record(
            s_global=s_global,
            depth=depth,
            per_segment=per_segment,
            host_ops_s=host_ops_wall_s(schedule),
            host_ops_s_per_pass=host_ops_wall_s_per_pass(schedule),
        )
        traffic: Optional[TrafficRecord] = None
        if boundaries is not None or noc is not None:
            traffic = TrafficRecord(boundaries=boundaries, noc=noc)

        deployed = AccuracyReadRecord(
            metric=float(self.pipeline.get_target_metric()),
            backend="pipeline",
            samples=0,
            kind="carried",
            step=STEP_NAME,
        )
        accuracy = AccuracyRecord(
            deployed=deployed, reads=tuple(reads), certificates=certificates,
        )
        view = seal_view_for(self.pipeline, plan)
        adaptation, adaptation_detail = adaptation_from_run_dir(
            self.pipeline.working_directory,
            tuner_steps_resolved=bool(
                set(resolved_step_names(self.pipeline))
                & view.tuner_hosting_step_names
            ),
        )

        builder = DeploymentRecordBuilder()
        builder.attach(
            "identity",
            identity_from_pipeline(self.pipeline, plan, platform),
            _prov("declared",
                  "DeploymentPlan.resolve + platform_constraints_resolved",
                  STEP_NAME),
        )
        builder.attach(
            "schedule", schedule,
            _prov("derived", "HybridHardCoreMapping stage census",
                  "Hard Core Mapping"),
        )
        builder.attach(
            "placement", placement,
            _prov("derived", "soft_core_placements_per_hard_core + SANA-FE floorplan",
                  "Hard Core Mapping"),
        )
        builder.attach(
            "utilization", utilization,
            _prov("derived", "CrossbarUtilizationReport + LayoutVerificationStats",
                  "Hard Core Mapping"),
        )
        builder.attach(
            "accuracy", accuracy,
            _prov("measured", "mapping metric runs + certificate gates",
                  STEP_NAME),
        )
        builder.attach(
            "timing", timing,
            _prov("measured" if per_segment else "derived",
                  "SANA-FE per-segment sim + StageTimer host walls"
                  if per_segment else "static schedule census",
                  STEP_NAME),
        )
        if traffic is not None:
            builder.attach(
                "traffic", traffic,
                _prov("measured", "spike-count gate reduction + SANA-FE NoC",
                      STEP_NAME),
            )
        if energy is not None:
            builder.attach(
                "energy", energy,
                _prov("measured", "SANA-FE energy trace", "SANA-FE Simulation"),
            )
        if adaptation is not None:
            builder.attach(
                "adaptation", adaptation,
                _prov("measured", "tuner ft_pass_wall_metrics", STEP_NAME,
                      detail=adaptation_detail),
            )
        builder.declare_weight_programming_totals(
            params_programmed=int(hcm["weight_programming"]["params_programmed"]),
        )

        record = builder.seal(view, resolved_step_names(self.pipeline))
        path = save_deployment_record(record, self.pipeline.working_directory)
        self.add_entry("deployment_record", record.to_dict(), "basic")

        cost_path = None
        if energy is not None:
            cost_path = save_cost_record(
                cost_record_from_deployment_record(record),
                self.pipeline.working_directory,
            )

        # Vendor-priced plane iff the run declared physics (else byte-identical).
        report_path = emit_physics_report(record, self.pipeline.working_directory)
        print(
            f"[DeploymentRecordStep] sealed {path} "
            f"(fragments: energy={'yes' if energy else 'no'}, "
            f"noc={'yes' if noc else 'no'}, "
            f"boundaries={'yes' if boundaries is not None else 'no'}, "
            f"adaptation={'yes' if adaptation else 'no'}; "
            f"cost_record={'written' if cost_path else 'n/a'}, "
            f"physics_report={'written' if report_path else 'none declared'})"
        )
        self.metric = self.pipeline.get_target_metric()
        self._verdict = {
            "status": "pass",
            "rule": "deployment record sealed against the plan matrix",
            "detail": {
                "path": path,
                "cost_record_path": cost_path,
                "pass_count": record.schedule.pass_count,
                "cores": sum(
                    len(seg.cores) for seg in record.schedule.segments()
                ),
                "sealed_groups": sorted(record.provenance),
            },
        }

    def _cross_check_sources(self, schedule, scm, mapping) -> None:
        """The three cached sources must still agree with the live mapping."""
        planned_reload = int(scm["reuse_plan"]["params_reloaded"])
        if schedule.params_reloaded != planned_reload:
            raise ValueError(
                f"deployment record: schedule.params_reloaded "
                f"{schedule.params_reloaded} != SCM reuse-plan figure "
                f"{planned_reload} — the cached fragments drifted"
            )
        neural_stages = sum(
            1 for stage in mapping.stages if stage.kind == "neural"
        )
        if len(schedule.segments()) != neural_stages:
            raise ValueError(
                f"deployment record: schedule carries "
                f"{len(schedule.segments())} neural segments but the mapping "
                f"has {neural_stages} neural stages — the cached fragments "
                f"drifted"
            )

    @staticmethod
    def _boundary_records(hcm) -> Optional[Tuple[BoundaryTrafficRecord, ...]]:
        boundaries = hcm["boundary_traffic"]
        if boundaries is None:
            return None
        return tuple(BoundaryTrafficRecord.from_dict(b) for b in boundaries)
