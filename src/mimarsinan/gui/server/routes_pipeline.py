"""Pipeline monitoring, run history, and WebSocket routes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from mimarsinan.gui.runs import (
    list_runs,
    get_run_config,
    get_run_pipeline,
    get_run_step_detail as hist_step_detail,
    get_run_console_logs,
)
from mimarsinan.gui.runtime.active_run_hub import ActiveRunHub
from mimarsinan.gui.runtime.persistence import load_console_logs
from mimarsinan.gui.server.json_safe import SafeJSONResponse

if TYPE_CHECKING:
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.runtime.process_manager import ProcessManager

logger = logging.getLogger("mimarsinan.gui")


def register_routes(
    app: FastAPI,
    *,
    collector: "DataCollector",
    process_manager: "ProcessManager | None",
) -> None:
    @app.get("/api/pipeline")
    def pipeline_overview():
        return collector.get_pipeline_overview()

    @app.get("/api/steps/{step_name}")
    def step_detail(step_name: str, request: Request, since_seq: int = 0):
        etag = collector.get_step_snapshot_etag(step_name)
        if etag is None:
            return JSONResponse(status_code=404, content={"error": "step not found"})
        inm = request.headers.get("if-none-match")
        if inm and inm == etag:
            return Response(status_code=304, headers={"ETag": etag})
        detail = collector.get_step_detail(step_name, since_seq=since_seq)
        if detail is None:
            return JSONResponse(status_code=404, content={"error": "step not found"})
        return SafeJSONResponse(content=detail, headers={"ETag": etag})

    @app.get("/api/steps/{step_name}/metrics")
    def step_metrics(step_name: str):
        return collector.get_step_metrics(step_name)

    @app.get("/api/metrics")
    def all_metrics():
        return collector.get_all_metrics()

    @app.get("/api/config")
    def pipeline_config():
        return collector.pipeline_config or {}

    @app.get("/api/console")
    def api_console_logs(offset: int = 0):
        return collector.get_console_logs(offset=offset)

    @app.get("/api/runs")
    def api_list_runs(include_steps: bool = False):
        return list_runs(include_steps=include_steps)

    @app.get("/api/runs/{run_id}/config")
    def api_run_config(run_id: str):
        config = get_run_config(run_id)
        if config is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return config

    @app.get("/api/runs/{run_id}/pipeline")
    def api_run_pipeline(run_id: str):
        data = get_run_pipeline(run_id)
        if data is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return data

    @app.get("/api/runs/{run_id}/steps/{step_name}")
    def api_run_step(run_id: str, step_name: str):
        detail = hist_step_detail(run_id, step_name)
        if detail is None:
            return {"error": "step not found"}
        return detail

    @app.get("/api/runs/{run_id}/console")
    def api_run_console(run_id: str, offset: int = 0):
        return get_run_console_logs(run_id, offset=offset)

    @app.get("/api/active_runs")
    def api_active_runs():
        if process_manager is None:
            return []
        return process_manager.list_active()

    @app.get("/api/active_runs/{run_id}/pipeline")
    def api_active_pipeline(run_id: str):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        detail = process_manager.get_run_detail(run_id)
        if detail is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return detail

    @app.get("/api/active_runs/{run_id}/steps/{step_name}")
    def api_active_step(run_id: str, step_name: str):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        detail = process_manager.get_run_step_detail(run_id, step_name)
        if detail is None:
            return {"error": "step not found"}
        return detail

    @app.delete("/api/active_runs/{run_id}")
    def api_kill_run(run_id: str):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        ok = process_manager.kill_run(run_id)
        return {"killed": ok}

    @app.get("/api/active_runs/{run_id}/console")
    def api_active_console(run_id: str, offset: int = 0):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        working_dir = process_manager.get_working_dir(run_id)
        if working_dir is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return load_console_logs(working_dir, offset=offset)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        ws._loop = asyncio.get_event_loop()
        collector.add_ws_listener(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "resume":
                    try:
                        last_seq = int(msg.get("last_seq", 0) or 0)
                    except Exception:
                        last_seq = 0
                    collector.replay_events_since(ws, last_seq)
        except WebSocketDisconnect:
            pass
        finally:
            collector.remove_ws_listener(ws)

    if process_manager is not None:
        def _active_overview(run_id: str):
            try:
                return process_manager.get_run_detail(run_id)
            except Exception:
                return None

        active_hub = ActiveRunHub(
            get_working_dir=process_manager.get_working_dir,
            build_overview=_active_overview,
        )
        app.state.active_run_hub = active_hub

        @app.websocket("/ws/active_runs/{run_id}")
        async def active_run_ws(ws: WebSocket, run_id: str):
            await ws.accept()
            loop = asyncio.get_event_loop()

            def _send(msg: dict) -> None:
                try:
                    asyncio.run_coroutine_threadsafe(ws.send_json(msg), loop)
                except Exception:
                    logger.debug("Failed to schedule active-run WS send", exc_info=True)

            subscribed = active_hub.subscribe(run_id, ws, _send)
            if not subscribed:
                await ws.close(code=1008)
                return
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                active_hub.unsubscribe(run_id, ws)
