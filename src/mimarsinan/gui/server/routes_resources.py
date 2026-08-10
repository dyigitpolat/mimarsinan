"""Lazy resource serving routes for live, historical, and active runs.

Every 200 carries an ``ETag`` of the served bytes and honors
``If-None-Match`` with an empty-body 304 -- resource bytes are immutable per
step version, so a re-attaching browser revalidates hundreds of heatmap tiles
without re-downloading one. ``?res=full`` serves the on-demand near-native
render from the persisted source, falling back to the UI-resolution artifact.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
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
    "ir_core_bias": "image/png",
    "ir_bank_heatmap": "image/png",
    "hard_core_heatmap": "image/png",
    "pruning_layer_heatmap": "image/png",
    "pruning_mask_map": "image/png",
    "heatmap_colorbar": "image/png",
    "connectivity": "application/json",
}

_CACHE_CONTROL = "public, max-age=3600, immutable"


def conditional_bytes_response(
    payload: bytes, media_type: str, request: Request | None,
) -> Response:
    """A 200 with ``ETag``, or an empty 304 when ``If-None-Match`` matches."""
    etag = f'"{hashlib.sha256(payload).hexdigest()[:32]}"'
    headers = {"Cache-Control": _CACHE_CONTROL, "ETag": etag}
    if request is not None:
        if_none_match = request.headers.get("if-none-match")
        if if_none_match is not None:
            candidates = {tag.strip() for tag in if_none_match.split(",")}
            if "*" in candidates or etag in candidates:
                return Response(status_code=304, headers=headers)
    return Response(content=payload, media_type=media_type, headers=headers)


def _conditional_json_response(payload: object, request: Request | None) -> Response:
    body = SafeJSONResponse(content=payload).body
    return conditional_bytes_response(bytes(body), "application/json", request)


def serve_resource_from_disk(
    working_dir: str | None,
    step_name: str,
    kind: str,
    rid: str,
    request: Request | None = None,
    res: str = "ui",
) -> Response:
    """Load a persisted resource file and return it with the correct Content-Type."""
    media_type = RESOURCE_MEDIA_TYPE_BY_KIND.get(kind)
    if media_type is None:
        return JSONResponse(status_code=404, content={"error": f"unknown resource kind {kind!r}"})
    if not working_dir:
        return JSONResponse(status_code=404, content={"error": "run not found"})
    payload = None
    if res == "full" and media_type == "image/png":
        payload = load_resource_from_disk(
            working_dir, step_name, kind, rid, media_type=media_type, variant="full",
        )
    if payload is None:
        payload = load_resource_from_disk(working_dir, step_name, kind, rid, media_type=media_type)
    if payload is None:
        return JSONResponse(status_code=404, content={"error": "resource not found"})
    return conditional_bytes_response(payload, media_type, request)


def register_routes(
    app: FastAPI,
    *,
    collector: "DataCollector",
    process_manager: "ProcessManager | None",
) -> None:
    @app.get("/api/steps/{step_name}/resources/{kind}/{rid:path}")
    async def step_resource(step_name: str, kind: str, rid: str, request: Request, res: str = "ui"):
        store = collector.get_resource_store()
        media_type = RESOURCE_MEDIA_TYPE_BY_KIND.get(kind)
        if media_type is None:
            return JSONResponse(status_code=404, content={"error": f"unknown resource kind {kind!r}"})
        working_dir = collector.get_working_directory()
        if res == "full" and media_type == "image/png" and working_dir:
            payload = await asyncio.to_thread(
                load_resource_from_disk, working_dir, step_name, kind, rid,
                media_type=media_type, variant="full",
            )
            if payload is not None:
                return conditional_bytes_response(payload, media_type, request)
        if store is not None:
            if media_type == "image/png":
                hit = await asyncio.to_thread(store.get_bytes, step_name, kind, rid)
                if hit is not None:
                    payload, mt = hit
                    return conditional_bytes_response(payload, mt, request)
            else:
                payload = await asyncio.to_thread(store.get_json, step_name, kind, rid)
                if payload is not None:
                    return _conditional_json_response(payload, request)
        if working_dir:
            return await asyncio.to_thread(
                serve_resource_from_disk, working_dir, step_name, kind, rid, request, res,
            )
        return JSONResponse(status_code=404, content={"error": "resource not found"})

    @app.get("/api/runs/{run_id}/steps/{step_name}/resources/{kind}/{rid:path}")
    def api_run_step_resource(run_id: str, step_name: str, kind: str, rid: str, request: Request, res: str = "ui"):
        try:
            _validate_run_id(run_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "invalid run_id"})
        run_dir = os.path.join(get_runs_root(), run_id)
        if not os.path.isdir(run_dir):
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return serve_resource_from_disk(run_dir, step_name, kind, rid, request, res)

    @app.get("/api/active_runs/{run_id}/steps/{step_name}/resources/{kind}/{rid:path}")
    def api_active_step_resource(run_id: str, step_name: str, kind: str, rid: str, request: Request, res: str = "ui"):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        working_dir = process_manager.get_working_dir(run_id)
        if working_dir is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return serve_resource_from_disk(working_dir, step_name, kind, rid, request, res)
