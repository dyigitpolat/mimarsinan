"""Artifacts routes: run-directory inventory and guarded file downloads."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.runs import (
    get_run_artifact_file,
    get_run_artifacts,
    list_dir_artifacts,
    resolve_artifact_file,
)

logger = logging.getLogger("mimarsinan.gui")


def register_routes(
    app: FastAPI,
    *,
    collector: Any,  # DataCollector (duck-typed: avoids the lazy-import cycle break)
    process_manager: Any,  # ProcessManager | None
) -> None:
    @app.get("/api/artifacts")
    def api_artifacts():
        working_dir = collector.get_working_directory()
        if not working_dir:
            return []
        return list_dir_artifacts(working_dir)

    @app.get("/api/artifact_file")
    def api_artifact_file(path: str):
        working_dir = collector.get_working_directory()
        resolved = resolve_artifact_file(working_dir, path) if working_dir else None
        if resolved is None:
            return JSONResponse(status_code=404, content={"error": "artifact not found"})
        return FileResponse(resolved)

    @app.get("/api/runs/{run_id}/artifacts")
    def api_run_artifacts(run_id: str):
        entries = None
        with best_effort(f"list artifacts for run {run_id}", logger=logger):
            entries = get_run_artifacts(run_id)
        if entries is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return entries

    @app.get("/api/runs/{run_id}/artifact_file")
    def api_run_artifact_file(run_id: str, path: str):
        resolved = None
        with best_effort(f"resolve artifact file for run {run_id}", logger=logger):
            resolved = get_run_artifact_file(run_id, path)
        if resolved is None:
            return JSONResponse(status_code=404, content={"error": "artifact not found"})
        return FileResponse(resolved)

    @app.get("/api/active_runs/{run_id}/artifacts")
    def api_active_artifacts(run_id: str):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        working_dir = process_manager.get_working_dir(run_id)
        if working_dir is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return list_dir_artifacts(working_dir)
