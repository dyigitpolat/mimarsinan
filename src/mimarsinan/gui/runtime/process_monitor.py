"""Filesystem polling and run detail APIs for managed subprocesses."""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.viewmodel import (
    annotations_for_step,
    build_overview_chart,
    categories_for,
    persisted_step_view,
    semantic_groups_from_config_view,
    step_bar_badge,
)
from mimarsinan.gui.runtime.persistence.load import (
    load_events,
    load_live_metrics,
    load_persisted_steps,
    load_run_info,
)
from mimarsinan.gui.runtime.process_spawn import ManagedRun
from mimarsinan.gui.snapshot.rebuild import rebuild_step_snapshot_from_disk

logger = logging.getLogger("mimarsinan.gui")


def recover_orphaned_runs(runs: dict[str, ManagedRun], generated_files_root: str) -> None:
    root = Path(generated_files_root)
    if not root.is_dir():
        return
    for child in root.iterdir():
        if not child.is_dir():
            continue
        run_id = child.name
        if run_id in runs:
            continue
        info = load_run_info(str(child))
        if info is None:
            continue
        pid = info.get("pid", 0)
        status = info.get("status", "unknown")
        started_at = info.get("started_at", 0.0)
        finished_at = info.get("finished_at")

        if finished_at and (time.time() - finished_at) > 3600:
            continue

        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                pass

        if alive or status == "running" or (finished_at and (time.time() - finished_at) < 3600):
            experiment_name = (info.get("config_summary") or {}).get("experiment_name", run_id)
            managed = ManagedRun(
                run_id=run_id,
                working_dir=str(child),
                pid=pid,
                started_at=started_at,
                experiment_name=experiment_name,
            )
            runs[run_id] = managed
            logger.info("Recovered orphaned run %s (pid=%d, status=%s)", run_id, pid, status)


def cleanup_stale_runs(runs: dict[str, ManagedRun]) -> None:
    cutoff = time.time() - 3600
    to_remove = []
    for run_id, managed in runs.items():
        if not managed.is_alive():
            info = load_run_info(managed.working_dir)
            finished_at = (info or {}).get("finished_at")
            if not isinstance(finished_at, (int, float)):
                finished_at = managed.started_at
            if finished_at < cutoff:
                to_remove.append(run_id)
    for rid in to_remove:
        del runs[rid]


def list_active(runs: dict[str, ManagedRun]) -> list[dict]:
    cleanup_stale_runs(runs)
    results: list[dict] = []
    # Snapshot items before iterating so concurrent mutation can't raise RuntimeError.
    for run_id, managed in list(runs.items()):
        info = load_run_info(managed.working_dir)
        steps = load_persisted_steps(managed.working_dir)

        alive = managed.is_alive()
        status = "running" if alive else (info or {}).get("status", "unknown")

        step_names = (info or {}).get("step_names", [])
        total = len(step_names)
        completed = sum(
            1 for s in steps.values()
            if s.get("status") == "completed" or (s.get("end_time") is not None and s.get("status") != "running")
        )
        failed = sum(1 for s in steps.values() if s.get("status") == "failed")
        current_step = None
        for sn in step_names:
            sd = steps.get(sn, {})
            if sd.get("status") == "running":
                current_step = sn
                break

        target_metrics = []
        for sn in step_names:
            sd = steps.get(sn, {})
            tm = sd.get("target_metric")
            if tm is not None:
                target_metrics.append({"step": sn, "value": tm})

        progress = (completed / total) if total > 0 else 0.0

        steps_summary = {}
        for sn in step_names:
            sd = steps.get(sn, {})
            st = sd.get("status", "pending")
            if st == "pending" and sd.get("end_time") is not None:
                st = "completed"
            if st == "running" and not alive:
                st = "failed"
            steps_summary[sn] = {
                "status": st,
                "end_time": sd.get("end_time"),
            }

        results.append({
            "run_id": run_id,
            "experiment_name": managed.experiment_name,
            "is_alive": alive,
            "status": status,
            "started_at": managed.started_at,
            "total_steps": total,
            "completed_steps": completed,
            "failed_steps": failed,
            "current_step": current_step,
            "progress": progress,
            "target_metrics": target_metrics,
            "pid": managed.pid,
            "step_names": step_names,
            "steps": steps_summary,
        })
    return results


def get_run_detail(runs: dict[str, ManagedRun], run_id: str) -> dict | None:
    managed = runs.get(run_id)
    if managed is None:
        return None

    info = load_run_info(managed.working_dir)
    steps_data = load_persisted_steps(managed.working_dir)
    step_names = (info or {}).get("step_names", [])
    alive = managed.is_alive()

    steps = []
    current_step = None
    for sn in step_names:
        sd = steps_data.get(sn, {})
        status = sd.get("status", "pending")
        if status == "pending" and sd.get("end_time") is not None:
            status = "completed"
        if status == "running" and not alive:
            status = "failed"
        if status == "running":
            current_step = sn
        steps.append(persisted_step_view(sn, sd, status=status))

    config = (info or {}).get("config_summary")
    config_path = os.path.join(managed.working_dir, "_RUN_CONFIG", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    config_view = None
    if config:
        with best_effort(f"build config_view for run {run_id}", logger=logger):
            from mimarsinan.config_schema.display_view import build_config_display_view
            config_view = build_config_display_view(config, saved_config=config)
    groups = semantic_groups_from_config_view(config_view)
    for s in steps:
        s["semantic_group"] = groups.get(s["name"])

    run_status = (info or {}).get("status", "running" if alive else "unknown")
    run_error = (info or {}).get("error")
    if not alive and run_status == "running":
        run_status = "failed"

    for s in steps:
        s["badge"] = step_bar_badge(s)
    result = {
        "steps": steps,
        "current_step": current_step,
        "config": config,
        "is_alive": alive,
        "status": run_status,
        "error": run_error,
        "overview_chart": build_overview_chart(steps),
    }
    if config_view is not None:
        result["config_view"] = config_view
    return result


def get_run_step_detail(runs: dict[str, ManagedRun], run_id: str, step_name: str) -> dict | None:
    managed = runs.get(run_id)
    if managed is None:
        return None

    steps_data = load_persisted_steps(managed.working_dir)
    sd = steps_data.get(step_name)
    if sd is None:
        return None

    live = load_live_metrics(managed.working_dir, step_name=step_name)
    metrics = sd.get("metrics", [])
    if live:
        existing_seqs = {m.get("seq") for m in metrics}
        for lm in live:
            if lm.get("seq") not in existing_seqs:
                metrics.append({
                    "seq": lm.get("seq"),
                    "name": lm.get("name"),
                    "value": lm.get("value"),
                    "timestamp": lm.get("timestamp"),
                    "global_step": lm.get("global_step"),
                })

    step_status = sd.get("status", "pending")
    if step_status == "pending" and sd.get("end_time") is not None:
        step_status = "completed"

    metric_categories = categories_for({m.get("name", "") for m in metrics})
    annotations = annotations_for_step(
        load_events(managed.working_dir, step_name=step_name),
        step_name, sd.get("start_time"),
    )

    snapshot = sd.get("snapshot")
    snapshot_key_kinds = sd.get("snapshot_key_kinds")
    if snapshot is None:
        rebuilt = rebuild_step_snapshot_from_disk(managed.working_dir, step_name)
        if rebuilt is not None:
            snapshot, snapshot_key_kinds = rebuilt

    return {
        **persisted_step_view(step_name, sd, status=step_status),
        "metrics": metrics,
        "metric_categories": metric_categories,
        "annotations": annotations,
        "snapshot": snapshot,
        "snapshot_key_kinds": snapshot_key_kinds,
    }


def kill_run(runs: dict[str, ManagedRun], run_id: str) -> bool:
    managed = runs.get(run_id)
    if managed is None:
        return False
    if not managed.is_alive():
        return False
    try:
        pgid = os.getpgid(managed.pid)
    except (OSError, ProcessLookupError):
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
        logger.info("Sent SIGTERM to run %s (pid=%d)", run_id, managed.pid)
    except (OSError, ProcessLookupError) as e:
        logger.warning("Failed to SIGTERM run %s: %s", run_id, e)
        return False

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not managed.is_alive():
            return True
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
        logger.warning("Escalated to SIGKILL for run %s (pid=%d)", run_id, managed.pid)
    except (OSError, ProcessLookupError):
        pass
    return True
