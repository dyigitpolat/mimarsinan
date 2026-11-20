"""Host-ComputeOp wall folding: the raw run total (a diagnostic) and [U-HOST] the one per-inference host wall."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

from mimarsinan.chip_simulation.hybrid_run.stage_timing import (
    HostWallCountError, count_refusal, host_wall, timed_sample_count, wall_refusal)
from mimarsinan.deployment_record.schema import ComputeOpRecord, ScheduleRecord


def host_ops_wall_s(schedule: ScheduleRecord) -> Optional[float]:
    """Σ measured ComputeOp walls over the whole run (``stage_timing.RAW_TOTAL_LABEL``: a diagnostic, never priced);
    ``None`` when untimed."""
    return host_wall(schedule.stages).raw_total_s


def host_ops_wall_s_per_pass(schedule: ScheduleRecord) -> Optional[float]:
    """[U-HOST] The per-inference host wall (``stage_timing.host_wall``, its ``PER_INFERENCE_DEFINITION``): each op's
    wall over the inferences its calls carried. ``None`` when untimed; a timed op without its calls or samples per
    call refuses by name (``HostWallCountError``) — an unnormalizable measurement is never guessed at."""
    return host_wall(schedule.stages).required()


def fold_compute_walls(
    schedule: ScheduleRecord, walls: Sequence[Mapping[str, Any]]
) -> ScheduleRecord:
    """Fold measured ``StageTimer`` walls onto the schedule's ComputeOps.

    Matched by stage NAME: the timer enumerates execution units (level stages
    included), so its indices need not equal the program's stage indices. A
    wall that matches no ComputeOp, or an ambiguous (duplicated) name, raises.
    [U-HOST] Each row's samples per call is folded with its calls; a row without
    it, or one name timed at two counts, refuses by name (``HostWallCountError``).
    [U-HOST3] So does a wall or a call count that is not the timer's (the writers'
    rule, ``stage_timing.wall_refusal`` and ``count_refusal``).
    """
    if not walls:
        return schedule
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    per_call: dict[str, int] = {}
    for row in walls:
        name = str(row["name"])
        for field, why in (("wall_s_total", wall_refusal(row["wall_s_total"])),  # [U-HOST3] the writers' rule
                           ("invocations", count_refusal(row.get("invocations")))):
            if why is not None:
                raise HostWallCountError(f"timed host-op wall {name!r}: its {field} {why} (the timer writes it so)")
        totals[name] = totals.get(name, 0.0) + float(row["wall_s_total"])
        counts[name] = counts.get(name, 0) + row["invocations"]
        samples = timed_sample_count(row.get("samples_per_call"), f"timed host-op wall {name!r}")
        if per_call.setdefault(name, samples) != samples:
            raise HostWallCountError(f"timed host-op wall {name!r} is folded at {per_call[name]} and {samples} "
                                     "samples per call: one ComputeOp's walls carry one count")
    compute_names = [stage.name for stage in schedule.stages if isinstance(stage, ComputeOpRecord)]
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
            invocations=counts[stage.name],
            samples_per_call=per_call[stage.name],
        )
        if isinstance(stage, ComputeOpRecord) and stage.name in totals
        else stage
        for stage in schedule.stages
    )
    return replace(schedule, stages=stages)
