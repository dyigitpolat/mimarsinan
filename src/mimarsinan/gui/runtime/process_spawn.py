"""Subprocess spawn helpers for process-based pipeline runs."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.runtime.persistence.store import append_console_log
from mimarsinan.gui.runtime.run_cache_seed import (
    copy_pipeline_cache_from_previous_run,
    copy_resources_from_previous_run,
    copy_steps_json_from_previous_run,
)

logger = logging.getLogger("mimarsinan.gui")

# .../src/mimarsinan/gui/runtime/process_spawn.py -> repo root (mimarsinan/)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUN_PY = str(_REPO_ROOT / "run.py")


def start_console_reader(proc: subprocess.Popen, working_dir: str) -> None:
    """Drain stdout/stderr from *proc* into console.jsonl."""

    def _drain(pipe, stream_name: str) -> None:
        with best_effort(f"console reader drain for {stream_name}", logger=logger):
            for raw in pipe:
                line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                append_console_log(working_dir, stream_name, line, time.time())
        with best_effort(f"console reader pipe close for {stream_name}", logger=logger):
            pipe.close()

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


def spawn_run(
    runs: dict[str, ManagedRun],
    generated_files_root: str,
    deployment_config: dict,
) -> str:
    """Write config and spawn a headless subprocess; register in *runs*."""
    experiment_name = deployment_config.get("experiment_name", "run")
    pipeline_mode = deployment_config.get("pipeline_mode", "phased")
    base = f"{experiment_name}_{pipeline_mode}_deployment_run"
    suffix = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{base}_{suffix}"

    gen_root = deployment_config.get("generated_files_path", generated_files_root)
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
        copy_resources_from_previous_run(
            os.path.abspath(gen_root),
            str(continue_from),
            working_dir,
        )

    python = sys.executable
    proc = subprocess.Popen(
        [python, "-u", _RUN_PY, "--headless", config_path],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    start_console_reader(proc, working_dir)

    managed = ManagedRun(
        run_id=run_id,
        working_dir=working_dir,
        pid=proc.pid,
        started_at=time.time(),
        experiment_name=experiment_name,
        _process=proc,
    )
    runs[run_id] = managed
    logger.info("Spawned run %s (pid=%d) in %s", run_id, proc.pid, working_dir)
    return run_id
