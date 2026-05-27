"""Process-based pipeline run management."""

from __future__ import annotations

from mimarsinan.gui.runtime.process_monitor import (
    cleanup_stale_runs,
    get_run_detail,
    get_run_step_detail,
    kill_run,
    list_active,
    recover_orphaned_runs,
)
from mimarsinan.gui.runtime.process_spawn import ManagedRun, spawn_run


class ProcessManager:
    """Spawn, track, poll and kill pipeline subprocesses."""

    def __init__(self, generated_files_root: str = "./generated") -> None:
        self._runs: dict[str, ManagedRun] = {}
        self._generated_files_root = generated_files_root
        recover_orphaned_runs(self._runs, self._generated_files_root)

    def spawn_run(self, deployment_config: dict) -> str:
        return spawn_run(self._runs, self._generated_files_root, deployment_config)

    def list_active(self) -> list[dict]:
        return list_active(self._runs)

    def get_run_detail(self, run_id: str) -> dict | None:
        return get_run_detail(self._runs, run_id)

    def get_run_step_detail(self, run_id: str, step_name: str) -> dict | None:
        return get_run_step_detail(self._runs, run_id, step_name)

    def get_working_dir(self, run_id: str) -> str | None:
        managed = self._runs.get(run_id)
        if managed is None:
            return None
        return managed.working_dir

    def kill_run(self, run_id: str) -> bool:
        return kill_run(self._runs, run_id)

    def _cleanup(self) -> None:
        cleanup_stale_runs(self._runs)
