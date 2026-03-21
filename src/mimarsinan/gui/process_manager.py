"""Process-based pipeline run management.

Each pipeline run is an isolated OS process spawned via ``subprocess.Popen``.
The GUI server polls the filesystem (_GUI_STATE/) to monitor progress.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimarsinan.gui.persistence import (
    load_run_info,
    load_persisted_steps,
    load_live_metrics,
    append_console_log,
)
from mimarsinan.gui.run_cache_seed import (
    copy_pipeline_cache_from_previous_run,
    copy_steps_json_from_previous_run,
)

logger = logging.getLogger("mimarsinan.gui")

_RUN_PY = str(Path(__file__).resolve().parents[3] / "run.py")


def _start_console_reader(proc: subprocess.Popen, working_dir: str) -> None:
    """Spawn daemon threads that drain stdout and stderr from *proc* into console.jsonl."""

    def _drain(pipe, stream_name: str) -> None:
        try:
            for raw in pipe:
                line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                append_console_log(working_dir, stream_name, line, time.time())
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    for pipe, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        if pipe is None:
            continue
        t = threading.Thread(
            target=_drain,
            args=(pipe, name),
            daemon=True,
            name=f"console-reader-{name}-{proc.pid}",
        )
        t.start()


@dataclass
class ManagedRun:
    run_id: str
    working_dir: str
    pid: int
    started_at: float
    experiment_name: str = ""
    _process: subprocess.Popen | None = field(default=None, repr=False)

    def is_alive(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        try:
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


class ProcessManager:
    """Spawn, track, poll and kill pipeline subprocesses."""

    def __init__(self, generated_files_root: str = "./generated") -> None:
        self._runs: dict[str, ManagedRun] = {}
        self._generated_files_root = os.path.abspath(generated_files_root)
        self._recover_orphaned_runs()

    def _recover_orphaned_runs(self) -> None:
        """Scan the generated files root for runs that are still alive (or recently finished)
        but not tracked — e.g. after a server restart."""
        root = Path(self._generated_files_root)
        if not root.is_dir():
            return
        for child in root.iterdir():
            if not child.is_dir():
                continue
            run_id = child.name
            if run_id in self._runs:
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
                self._runs[run_id] = managed
                logger.info("Recovered orphaned run %s (pid=%d, status=%s)", run_id, pid, status)

    def spawn_run(self, deployment_config: dict) -> str:
        """Write config to a unique working directory and spawn a headless subprocess.

        Returns the run_id (which is the directory basename).
        """
        experiment_name = deployment_config.get("experiment_name", "run")
        pipeline_mode = deployment_config.get("pipeline_mode", "phased")
        base = f"{experiment_name}_{pipeline_mode}_deployment_run"
        suffix = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{base}_{suffix}"

        gen_root = deployment_config.get("generated_files_path", self._generated_files_root)
        working_dir = os.path.join(os.path.abspath(gen_root), run_id)
        config_dir = os.path.join(working_dir, "_RUN_CONFIG")
        os.makedirs(config_dir, exist_ok=True)

        config_to_write = dict(deployment_config)
        config_to_write["generated_files_path"] = gen_root
        config_to_write["_working_directory"] = working_dir
        continue_from = config_to_write.pop("_continue_from_run_id", None)

        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_to_write, f, indent=2)

        os.makedirs(os.path.join(working_dir, "_GUI_STATE"), exist_ok=True)

        # Edit & continue: copy pipeline cache from the run we edited so
        # run_from(start_step=...) finds requirements in self.cache.
        if continue_from:
            copy_pipeline_cache_from_previous_run(
                os.path.abspath(gen_root),
                str(continue_from),
                working_dir,
            )
            copy_steps_json_from_previous_run(
                os.path.abspath(gen_root),
                str(continue_from),
                working_dir,
            )

        python = sys.executable
        proc = subprocess.Popen(
            [python, "-u", _RUN_PY, "--headless", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        _start_console_reader(proc, working_dir)

        managed = ManagedRun(
            run_id=run_id,
            working_dir=working_dir,
            pid=proc.pid,
            started_at=time.time(),
            experiment_name=experiment_name,
            _process=proc,
        )
        self._runs[run_id] = managed
        logger.info("Spawned run %s (pid=%d) in %s", run_id, proc.pid, working_dir)
        return run_id

    def list_active(self) -> list[dict]:
        """Return summary of all tracked runs by polling their _GUI_STATE files."""
        self._cleanup()
        results: list[dict] = []
        for run_id, managed in self._runs.items():
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

    def get_run_detail(self, run_id: str) -> dict | None:
        """Read _GUI_STATE files for detailed pipeline state (like /api/pipeline)."""
        managed = self._runs.get(run_id)
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
            start_t = sd.get("start_time")
            end_t = sd.get("end_time")
            steps.append({
                "name": sn,
                "status": status,
                "start_time": start_t,
                "end_time": end_t,
                "duration": (end_t - start_t) if start_t and end_t else None,
                "target_metric": sd.get("target_metric"),
            })

        config = (info or {}).get("config_summary")
        config_path = os.path.join(managed.working_dir, "_RUN_CONFIG", "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        # config.json stores the full outer config; pipeline specs expect the
        # flat deployment_parameters dict (same structure as pipeline.config).
        outer_config = config or {}
        flat_config = outer_config.get("deployment_parameters", outer_config)
        try:
            from mimarsinan.pipelining.pipelines.deployment_pipeline import (
                get_pipeline_semantic_group_by_step_name,
            )
            groups = get_pipeline_semantic_group_by_step_name(flat_config)
        except Exception:
            groups = {}
        for s in steps:
            s["semantic_group"] = groups.get(s["name"])

        return {
            "steps": steps,
            "current_step": current_step,
            "config": config,
            "is_alive": alive,
        }

    def get_run_step_detail(self, run_id: str, step_name: str) -> dict | None:
        """Read step detail + live metrics for a specific step of an active run."""
        managed = self._runs.get(run_id)
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

        return {
            "name": step_name,
            "status": step_status,
            "start_time": sd.get("start_time"),
            "end_time": sd.get("end_time"),
            "duration": (sd.get("end_time", 0) - sd.get("start_time", 0))
                if sd.get("start_time") and sd.get("end_time") else None,
            "target_metric": sd.get("target_metric"),
            "metrics": metrics,
            "snapshot": sd.get("snapshot"),
            "snapshot_key_kinds": sd.get("snapshot_key_kinds"),
        }

    def kill_run(self, run_id: str) -> bool:
        """Terminate a running subprocess.

        Sends SIGTERM (handled by the headless process to write status and
        os._exit), waits briefly, then escalates to SIGKILL if still alive.
        """
        managed = self._runs.get(run_id)
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

    def _cleanup(self) -> None:
        """Remove tracking for dead processes that have been finished for > 1 hour."""
        cutoff = time.time() - 3600
        to_remove = []
        for run_id, managed in self._runs.items():
            if not managed.is_alive():
                info = load_run_info(managed.working_dir)
                finished_at = (info or {}).get("finished_at", managed.started_at)
                if finished_at < cutoff:
                    to_remove.append(run_id)
        for rid in to_remove:
            del self._runs[rid]
