"""FastAPI application factory and server startup."""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.env import gui_no_browser
from mimarsinan.gui.server import routes_layout, routes_pipeline, routes_resources, routes_wizard
from mimarsinan.gui.server.json_safe import SafeJSONResponse

if TYPE_CHECKING:
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.runtime.process_manager import ProcessManager

logger = logging.getLogger("mimarsinan.gui")

_STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app(
    collector: "DataCollector",
    run_config_fn: Callable[[dict, "DataCollector"], None] | None = None,
    process_manager: "ProcessManager | None" = None,
) -> FastAPI:
    app = FastAPI(
        title="Mimarsinan Pipeline Monitor",
        docs_url=None,
        redoc_url=None,
        default_response_class=SafeJSONResponse,
    )

    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/wizard":
            response.headers["Cache-Control"] = "no-store"
        return response

    routes_pipeline.register_routes(app, collector=collector, process_manager=process_manager)
    routes_wizard.register_routes(
        app,
        collector=collector,
        run_config_fn=run_config_fn,
        process_manager=process_manager,
    )
    routes_layout.register_routes(app)
    routes_resources.register_routes(
        app,
        collector=collector,
        process_manager=process_manager,
    )

    @app.get("/wizard")
    def wizard_page():
        wizard_path = _STATIC_DIR / "wizard.html"
        if wizard_path.exists():
            return FileResponse(wizard_path, media_type="text/html")
        return HTMLResponse("<h1>Wizard not found</h1>", status_code=404)

    @app.get("/monitor")
    def monitor_page():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse("<h1>Mimarsinan Monitor</h1><p>Static files not found.</p>")

    @app.get("/")
    def index():
        welcome_path = _STATIC_DIR / "welcome.html"
        if welcome_path.exists():
            return FileResponse(welcome_path, media_type="text/html")
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse("<h1>Mimarsinan GUI</h1><p>Static files not found.</p>")

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


def gui_entry_url(port: int) -> str:
    """HTTP URL for the welcome page (``/``) shown and opened after the GUI server starts."""
    return f"http://127.0.0.1:{port}/"


def schedule_open_browser(url: str) -> None:
    """Open *url* in the default browser after a short delay so the server is accepting."""
    if gui_no_browser():
        return

    def _open() -> None:
        with best_effort(f"open browser for {url}", logger=logger):
            webbrowser.open(url)

    threading.Timer(0.6, _open).start()


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
            return True
        except OSError:
            return False


def start_server(
    collector: "DataCollector",
    host: str = "0.0.0.0",
    port: int = 8501,
    max_port_attempts: int = 20,
    run_config_fn: Callable[[dict, "DataCollector"], None] | None = None,
) -> threading.Thread:
    """Start the FastAPI server in a background daemon thread."""
    chosen_port = port
    for offset in range(max_port_attempts):
        candidate = port + offset
        if port_is_free(host, candidate):
            chosen_port = candidate
            break
    else:
        logger.warning(
            "All ports %d–%d busy; falling back to %d (may fail)",
            port, port + max_port_attempts - 1, port,
        )
        chosen_port = port

    pm = None
    if run_config_fn is not None:
        from mimarsinan.gui.runtime.process_manager import ProcessManager
        pm = ProcessManager()

    app = create_app(collector, run_config_fn=run_config_fn, process_manager=pm)

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
    logger.info("GUI server started (bind %s:%d)", host, chosen_port)
    entry_url = gui_entry_url(chosen_port)
    print(f"\n  Mimarsinan is running at: {entry_url}\n")
    schedule_open_browser(entry_url)
    return thread
