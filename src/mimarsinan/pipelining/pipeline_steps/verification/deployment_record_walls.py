"""Host-ComputeOp wall folding: raw run totals and their per-pass normalization."""

from __future__ import annotations

from typing import Optional

from mimarsinan.deployment_record.schema import ComputeOpRecord, ScheduleRecord


def host_ops_wall_s(schedule: ScheduleRecord) -> Optional[float]:
    """Σ measured ComputeOp walls over the whole run; ``None`` when untimed."""
    timed = [
        stage.wall_s_total
        for stage in schedule.stages
        if isinstance(stage, ComputeOpRecord) and stage.wall_s_total is not None
    ]
    if not timed:
        return None
    return float(sum(timed))


def host_ops_wall_s_per_pass(schedule: ScheduleRecord) -> Optional[float]:
    """Σ per-execution ComputeOp walls: the host time of ONE program traversal.

    Each op is normalized by its own invocation count, so ops executed at
    different rates still sum onto the per-sample axis. ``None`` when any timed
    op lacks a count — an unnormalizable measurement is never guessed at.
    """
    timed = [
        stage
        for stage in schedule.stages
        if isinstance(stage, ComputeOpRecord) and stage.wall_s_total is not None
    ]
    if not timed:
        return None
    per_pass = []
    for stage in timed:
        wall, runs = stage.wall_s_total, stage.invocations
        if wall is None or not runs:
            return None
        per_pass.append(float(wall) / int(runs))
    return float(sum(per_pass))

