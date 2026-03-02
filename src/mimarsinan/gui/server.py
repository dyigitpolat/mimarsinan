"""FastAPI server for the pipeline monitoring GUI.

Runs in a daemon thread so the pipeline is not blocked.  Provides REST
endpoints for polling and a WebSocket for real-time push.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from mimarsinan.gui.data_collector import DataCollector

logger = logging.getLogger("mimarsinan.gui")

_STATIC_DIR = Path(__file__).parent / "static"


class _SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN / Inf to ``null``."""

    def default(self, o: Any) -> Any:  # noqa: D401
        return super().default(o)

    def encode(self, o: Any) -> str:
        return super().encode(_sanitize(o))


def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class _SafeJSONResponse(JSONResponse):
    """JSONResponse that silently converts NaN/Inf to null."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            cls=_SafeJSONEncoder,
        ).encode("utf-8")


def create_app(collector: DataCollector) -> FastAPI:
    app = FastAPI(
        title="Mimarsinan Pipeline Monitor",
        docs_url=None,
        redoc_url=None,
        default_response_class=_SafeJSONResponse,
    )

    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # -- REST endpoints --------------------------------------------------------

    @app.get("/api/pipeline")
    def pipeline_overview():
        return collector.get_pipeline_overview()

    @app.get("/api/steps/{step_name}")
    def step_detail(step_name: str):
        detail = collector.get_step_detail(step_name)
        if detail is None:
            return {"error": "step not found"}
        return detail

    @app.get("/api/steps/{step_name}/metrics")
    def step_metrics(step_name: str):
        return collector.get_step_metrics(step_name)

    @app.get("/api/metrics")
    def all_metrics():
        return collector.get_all_metrics()

    @app.get("/api/config")
    def pipeline_config():
        return collector.pipeline_config or {}

    # -- WebSocket -------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        ws._loop = asyncio.get_event_loop()
        collector.add_ws_listener(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            collector.remove_ws_listener(ws)

    # -- Static files ----------------------------------------------------------

    @app.get("/")
    def index():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse("<h1>Mimarsinan GUI</h1><p>Static files not found.</p>")

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


def _port_is_free(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
            return True
        except OSError:
            return False


def start_server(
    collector: DataCollector,
    host: str = "0.0.0.0",
    port: int = 8501,
    max_port_attempts: int = 20,
) -> threading.Thread:
    """Start the FastAPI server in a background daemon thread.

    If *port* is already in use, tries successive ports up to
    *port + max_port_attempts - 1* before giving up.
    """
    import uvicorn

    chosen_port = port
    for offset in range(max_port_attempts):
        candidate = port + offset
        if _port_is_free(host, candidate):
            chosen_port = candidate
            break
    else:
        logger.warning(
            "All ports %d–%d busy; falling back to %d (may fail)",
            port, port + max_port_attempts - 1, port,
        )
        chosen_port = port

    app = create_app(collector)

    config = uvicorn.Config(
        app,
        host=host,
        port=chosen_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True, name="gui-server")
    thread.start()
    logger.info("GUI server started on http://%s:%d", host, chosen_port)
    print(f"\n  Pipeline Monitor GUI: http://localhost:{chosen_port}\n")
    return thread
