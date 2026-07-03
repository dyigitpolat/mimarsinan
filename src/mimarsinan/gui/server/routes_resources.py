"""Lazy resource serving routes for live, historical, and active runs."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from mimarsinan.gui.runs import get_runs_root, _validate_run_id
from mimarsinan.gui.runtime.persistence import load_resource_from_disk
from mimarsinan.gui.server.json_safe import SafeJSONResponse

if TYPE_CHECKING:
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.runtime.process_manager import ProcessManager

RESOURCE_MEDIA_TYPE_BY_KIND = {
    "ir_core_heatmap": "image/png",
    "ir_core_pre_pruning": "image/png",
    "ir_bank_heatmap": "image/png",
    "hard_core_heatmap": "image/png",
    "pruning_layer_heatmap": "image/png",
    "connectivity": "application/json",
}


def serve_resource_from_disk(
    working_dir: str | None,
    step_name: str,
    kind: str,
    rid: str,
) -> Response:
    """Load a persisted resource file and return it with the correct Content-Type."""
    media_type = RESOURCE_MEDIA_TYPE_BY_KIND.get(kind)
    if media_type is None:
        return JSONResponse(status_code=404, content={"error": f"unknown resource kind {kind!r}"})
    if not working_dir:
        return JSONResponse(status_code=404, content={"error": "run not found"})
    payload = load_resource_from_disk(working_dir, step_name, kind, rid, media_type=media_type)
    if payload is None:
        return JSONResponse(status_code=404, content={"error": "resource not found"})
    headers = {"Cache-Control": "public, max-age=3600, immutable"}
    return Response(content=payload, media_type=media_type, headers=headers)


def register_routes(
    app: FastAPI,
    *,
    collector: "DataCollector",
    process_manager: "ProcessManager | None",
) -> None:
    @app.get("/api/steps/{step_name}/resources/{kind}/{rid:path}")
    async def step_resource(step_name: str, kind: str, rid: str):
        store = collector.get_resource_store()
        media_type = RESOURCE_MEDIA_TYPE_BY_KIND.get(kind)
        if media_type is None:
            return JSONResponse(status_code=404, content={"error": f"unknown resource kind {kind!r}"})
        if store is not None:
            if media_type == "image/png":
                hit = await asyncio.to_thread(store.get_bytes, step_name, kind, rid)
                if hit is not None:
                    payload, mt = hit
                    return Response(
                        content=payload,
                        media_type=mt,
                        headers={"Cache-Control": "public, max-age=3600, immutable"},
                    )
            else:
                payload = await asyncio.to_thread(store.get_json, step_name, kind, rid)
                if payload is not None:
                    return SafeJSONResponse(
                        content=payload,
                        headers={"Cache-Control": "public, max-age=3600, immutable"},
                    )
        working_dir = collector.get_working_directory()
        if working_dir:
            return await asyncio.to_thread(
                serve_resource_from_disk, working_dir, step_name, kind, rid,
            )
        return JSONResponse(status_code=404, content={"error": "resource not found"})

    @app.get("/api/runs/{run_id}/steps/{step_name}/resources/{kind}/{rid:path}")
    def api_run_step_resource(run_id: str, step_name: str, kind: str, rid: str):
        try:
            _validate_run_id(run_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "invalid run_id"})
        run_dir = os.path.join(get_runs_root(), run_id)
        if not os.path.isdir(run_dir):
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return serve_resource_from_disk(run_dir, step_name, kind, rid)

    @app.get("/api/active_runs/{run_id}/steps/{step_name}/resources/{kind}/{rid:path}")
    def api_active_step_resource(run_id: str, step_name: str, kind: str, rid: str):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        working_dir = process_manager.get_working_dir(run_id)
        if working_dir is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return serve_resource_from_disk(working_dir, step_name, kind, rid)
