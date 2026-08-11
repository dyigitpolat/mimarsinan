"""Assembly helpers for the terminal Deployment Record step (W4 stage 4).

Pure functions that turn pipeline state + cached fragments into the typed
pieces ``DeploymentRecordStep`` seals: identity, host-op wall folding, the
timing record (with the no-double-count note), the adaptation fragment read
from the run directory, and the ``SealPlanView`` adapter over the resolved
``DeploymentPlan``. FAIL LOUD throughout — no ``best_effort`` anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import AbstractSet, Any, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.cost_extraction import FT_PASS_WALLS_FILENAME
from mimarsinan.deployment_record.schema import (
    DEPLOYMENT_RECORD_FORMAT_VERSION,
    AdaptationRecord,
    ComputeOpRecord,
    FtPassWallRecord,
    LatencyDecomposition,
    RecordIdentity,
    ScheduleRecord,
    SegmentTimingRecord,
    TimingRecord,
)
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_walls import (
    host_ops_wall_s as host_ops_wall_s,
    host_ops_wall_s_per_pass as host_ops_wall_s_per_pass,
)
from mimarsinan.pipelining.core.spike_count_gate import certificate_gate_armed
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.orchestration.run_instrumentation import (
    RETENTION_LEDGER_FILENAME,
)

# The explicit no-double-count statement (schema §2.5): stated once, verbatim.
LATENCY_NOTE = (
    "SANA-FE sim_time_s includes NoC hop latency (charged by the in-simulator "
    "C++ NoC), so compute_sim_time_s already contains NoC transport and no "
    "term double-counts it; programming_s and sync_s stay None until the "
    "stage-5 cost model prices them from the record's payload/sync census."
)


@dataclass
class PlanSealView:
    """The ``SealPlanView`` duck surface, adapted from the resolved plan.

    Deliberately not frozen: the ``SealPlanView`` protocol's attributes are
    writable, and a frozen dataclass's read-only fields would not satisfy it.
    """

    enable_sanafe_simulation: bool
    counts_observable: bool
    spike_count_gate_armed: bool
    nevresim_applies: bool
    tuner_hosting_step_names: AbstractSet[str]


def nevresim_applies(plan: Any) -> bool:
    """Whether the nevresim step joins this plan's tail (the registry SSOT)."""
    return bool(BACKEND_REGISTRY.get("nevresim").applies(plan))


def seal_view_for(pipeline: Any, plan: Any) -> PlanSealView:
    """Adapt the pipeline + plan onto the seal's required-fragment predicates."""
    observable, _reason = plan.mode_policy().certification_observable()
    return PlanSealView(
        enable_sanafe_simulation=bool(plan.enable_sanafe_simulation),
        counts_observable=observable == "counts",
        spike_count_gate_armed=bool(certificate_gate_armed(pipeline)),
        nevresim_applies=nevresim_applies(plan),
        tuner_hosting_step_names=frozenset(
            name
            for name, step in getattr(pipeline, "steps", []) or []
            if isinstance(step, TunerPipelineStep)
        ),
    )


def resolved_step_names(pipeline: Any) -> Tuple[str, ...]:
    """The run's resolved step-name sequence (assembled pipeline order)."""
    return tuple(name for name, _step in getattr(pipeline, "steps", []) or [])


def record_backend(plan: Any) -> str:
    """The identity cell's backend axis: the run's richest deployed target.

    SANA-FE when enabled (keeps ``cell_key`` value-identical to the legacy
    cost record), else nevresim when it applies, else the HCM twin itself.
    """
    if plan.enable_sanafe_simulation:
        return "sanafe"
    if nevresim_applies(plan):
        return "nevresim"
    return "hcm"


def config_digest(config: Mapping[str, Any]) -> str:
    """sha256 over the canonical (sorted-key) JSON of the resolved config.

    Runtime-injected non-JSON values (``device``, shapes) serialize via
    ``str`` — deterministic for identical resolved configs.
    """
    canonical = json.dumps(dict(config), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _workload_key(pipeline: Any) -> str:
    """The workload identity: the registered data-provider name."""
    factory = pipeline.data_provider_factory
    name = getattr(factory, "_name", None)
    if name:
        return str(name)
    return type(factory).__name__


def identity_from_pipeline(
    pipeline: Any, plan: Any, platform: Mapping[str, Any]
) -> RecordIdentity:
    """The record's key: what was deployed, where, under which options."""
    cell = CertificationCell.from_mode_policy(
        plan.mode_policy(), backend=record_backend(plan),
    )
    mode = cell.cell_key.rsplit("@", 1)[0]
    return RecordIdentity(
        format_version=DEPLOYMENT_RECORD_FORMAT_VERSION,
        run_dir=str(pipeline.working_directory),
        cell_key=cell.cell_key,
        mode=mode,
        model_type=str(plan.model_type),
        model_name=str(plan.model_name),
        workload=_workload_key(pipeline),
        config_digest=config_digest(pipeline.config),
        platform=dict(platform),
        deployment_options={
            "schedule_policy": platform.get("schedule_policy"),
            "max_schedule_passes": platform.get("max_schedule_passes"),
            "weight_bits": platform.get("weight_bits"),
            "target_tq": pipeline.config.get("target_tq"),
            "simulation_steps": int(pipeline.config["simulation_steps"]),
            "degradation_tolerance": float(plan.degradation_tolerance),
            "scm_degradation_tolerance": plan.scm_degradation_tolerance,
            "degradation_budget_total": float(plan.degradation_budget_total),
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def fold_compute_walls(
    schedule: ScheduleRecord, walls: Sequence[Mapping[str, Any]]
) -> ScheduleRecord:
    """Fold measured ``StageTimer`` walls onto the schedule's ComputeOps.

    Matched by stage NAME: the timer enumerates execution units (level stages
    included), so its indices need not equal the program's stage indices. A
    wall that matches no ComputeOp, or an ambiguous (duplicated) name, raises.
    """
    if not walls:
        return schedule
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in walls:
        name = str(row["name"])
        totals[name] = totals.get(name, 0.0) + float(row["wall_s_total"])
        counts[name] = counts.get(name, 0) + int(row.get("invocations", 0) or 0)
    compute_names = [
        stage.name
        for stage in schedule.stages
        if isinstance(stage, ComputeOpRecord)
    ]
    unknown = sorted(set(totals) - set(compute_names))
    if unknown:
        raise ValueError(
            f"timed host-op walls {unknown} match no ComputeOp stage in the "
            f"schedule (stages: {sorted(set(compute_names))})"
        )
    ambiguous = sorted(
        name for name in totals if compute_names.count(name) > 1
    )
    if ambiguous:
        raise ValueError(
            f"timed host-op walls {ambiguous} match multiple ComputeOp stages; "
            f"the name-keyed fold would be ambiguous"
        )
    stages = tuple(
        replace(
            stage,
            wall_s_total=totals[stage.name],
            invocations=counts.get(stage.name) or None,
        )
        if isinstance(stage, ComputeOpRecord) and stage.name in totals
        else stage
        for stage in schedule.stages
    )
    return replace(schedule, stages=stages)


def timing_record(
    *,
    s_global: int,
    depth: int,
    per_segment: Sequence[SegmentTimingRecord],
    host_ops_s: Optional[float],
    host_ops_s_per_pass: Optional[float] = None,
) -> TimingRecord:
    """The timing fragment: static part always, measured parts when present."""
    segments = tuple(per_segment)
    compute_sim_time_s = (
        float(sum(seg.sim_time_s for seg in segments)) if segments else None
    )
    return TimingRecord(
        s_global=int(s_global),
        depth=int(depth),
        per_segment=segments,
        latency=LatencyDecomposition(
            programming_s=None,
            compute_steps=int(
                sum(seg.timesteps_executed for seg in segments)
            ),
            compute_sim_time_s=compute_sim_time_s,
            host_ops_s=host_ops_s,
            host_ops_s_per_pass=host_ops_s_per_pass,
            sync_s=None,
            note=LATENCY_NOTE,
        ),
    )


def floorplan_derivation(platform: Mapping[str, Any]) -> str:
    """``declared`` when the platform declared any floorplan key, else ``derived``."""
    declared = any(
        int(platform.get(key, 0) or 0) > 0
        for key in ("cores_per_tile", "tile_grid_rows", "tile_grid_cols")
    )
    return "declared" if declared else "derived"


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def adaptation_from_run_dir(
    working_directory: str, *, tuner_steps_resolved: bool
) -> Tuple[Optional[AdaptationRecord], str]:
    """Read the run's adaptation artifacts into the fragment (+ ledger detail).

    ``ft_pass_walls.json`` populates the wall bundle; a resolved tuner-hosting
    step that recorded no FT passes still attaches an (empty) fragment so the
    seal's availability matrix holds honestly. Returns ``(None, "")`` when the
    run neither hosts tuners nor recorded walls.
    """
    walls_path = os.path.join(working_directory, FT_PASS_WALLS_FILENAME)
    record: Optional[AdaptationRecord] = None
    if os.path.exists(walls_path):
        data = _read_json(walls_path) or {}
        record = AdaptationRecord(
            max_ft_pass_wall_s=float(data.get("max_ft_pass_wall_s", 0.0)),
            ft_pass_walls=tuple(
                FtPassWallRecord(
                    label=str(entry["label"]), wall_s=float(entry["wall_s"])
                )
                for entry in data.get("passes") or ()
            ),
        )
    elif tuner_steps_resolved:
        record = AdaptationRecord(max_ft_pass_wall_s=0.0, ft_pass_walls=())

    detail_parts: List[str] = []
    if record is not None and not record.ft_pass_walls:
        detail_parts.append("no FT passes recorded (ft_pass_walls.json absent)")
    ledger_path = os.path.join(working_directory, RETENTION_LEDGER_FILENAME)
    if os.path.exists(ledger_path):
        entries = (_read_json(ledger_path) or {}).get("entries") or []
        detail_parts.append(
            f"retention_ledger.json: {len(entries)} tuner-step entries"
        )
    return record, "; ".join(detail_parts)
