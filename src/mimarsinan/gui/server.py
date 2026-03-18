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
from typing import TYPE_CHECKING, Any, Callable

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


# Large bound used for the layout estimation pass so we always capture natural
# softcore sizes without hitting any tiling/splitting check inside LayoutIRMapping.
# This is intentionally much larger than any realistic layer dimension.
_LAYOUT_PASS_LIMIT = 1 << 20  # ~1M axons/neurons — effectively unconstrained


def _get_softcores_from_request(body: dict):
    """Build a model repr and run layout mapping, returning LayoutSoftCoreSpec list.

    Handles both native (simple_mlp) and torch (torch_sequential_linear, etc.) builders
    by delegating to the appropriate builder class.

    The builder receives the user-specified max_axons/max_neurons so that native
    models are sized appropriately.  The layout pass itself uses _LAYOUT_PASS_LIMIT
    so that no tiling or error triggers inside LayoutIRMapping — we want the natural
    (unsplit) softcore shapes regardless of the user's hardware target.
    """
    from mimarsinan.mapping.mapping_verifier import verify_soft_core_mapping
    from mimarsinan.models.builders import BUILDERS_REGISTRY
    from mimarsinan.pipelining.model_registry import ModelRegistry
    import torch

    model_type = body.get("model_type", "simple_mlp")
    input_shape = tuple(int(x) for x in body.get("input_shape", [1, 28, 28]))
    num_classes = int(body.get("num_classes", 10))
    model_config = body.get("model_config", {})
    max_axons = int(body.get("max_axons", 1024))
    max_neurons = int(body.get("max_neurons", 1024))
    threshold_groups = int(body.get("threshold_groups", 1))
    pruning_fraction = float(body.get("pruning_fraction", 0.0))
    threshold_seed = int(body.get("threshold_seed", 0))

    builder_cls = BUILDERS_REGISTRY.get(model_type)
    if builder_cls is None:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    pipeline_config = {
        "target_tq": int(body.get("target_tq", 32)),
        "device": "cpu",
    }
    builder = builder_cls(
        device=torch.device("cpu"),
        input_shape=input_shape,
        num_classes=num_classes,
        max_axons=max_axons,
        max_neurons=max_neurons,
        pipeline_config=pipeline_config,
    )
    raw_model = builder.build(model_config)

    category = ModelRegistry.get_category(model_type)
    if category == "torch":
        from mimarsinan.torch_mapping.converter import convert_torch_model
        # Initialize lazy modules (e.g. LazyBatchNorm1d) before conversion
        raw_model.eval()
        with torch.no_grad():
            try:
                raw_model(torch.randn(1, *input_shape))
            except Exception:
                pass
        supermodel = convert_torch_model(
            raw_model,
            input_shape=input_shape,
            num_classes=num_classes,
            device="cpu",
        )
        model_repr = supermodel.get_mapper_repr()
    else:
        # Native model — also initialize lazy modules
        raw_model.eval()
        with torch.no_grad():
            try:
                raw_model(torch.randn(2, *input_shape))
            except Exception:
                pass
        model_repr = raw_model.get_mapper_repr()

    # Use the large layout-pass limit (not the user's max_axons/max_neurons) so
    # that LayoutIRMapping never tiles or errors — we need the natural softcore sizes.
    result = verify_soft_core_mapping(
        model_repr,
        max_axons=_LAYOUT_PASS_LIMIT,
        max_neurons=_LAYOUT_PASS_LIMIT,
        threshold_groups=threshold_groups,
        pruning_fraction=pruning_fraction,
        threshold_seed=threshold_seed,
    )
    if not result.feasible:
        raise ValueError(f"Soft-core mapping verification failed: {result.error}")
    return result.softcores


def create_app(
    collector: "DataCollector",
    run_config_fn: Callable[[dict, "DataCollector"], None] | None = None,
    process_manager: "ProcessManager | None" = None,
) -> FastAPI:
    from mimarsinan.gui.runs import (
        list_runs, get_run_config, get_run_pipeline, get_run_step_detail as hist_step_detail,
    )
    from mimarsinan.gui.templates import list_templates, get_template, save_template, delete_template
    app = FastAPI(
        title="Mimarsinan Pipeline Monitor",
        docs_url=None,
        redoc_url=None,
        default_response_class=_SafeJSONResponse,
    )

    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/wizard":
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

    # -- Wizard / config APIs (used when run_config_fn is set, e.g. --ui mode) ----

    @app.get("/api/data_providers")
    def api_data_providers():
        from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
        return BasicDataProviderFactory.list_registered()

    @app.get("/api/model_types")
    def api_model_types():
        from mimarsinan.pipelining.model_registry import get_model_types
        return get_model_types()

    @app.get("/api/model_config_schema/{model_type}")
    def api_model_config_schema(model_type: str):
        from mimarsinan.pipelining.model_registry import get_model_config_schema
        return get_model_config_schema(model_type)

    @app.post("/api/run")
    def api_run(body: dict):
        if process_manager is not None:
            try:
                run_id = process_manager.spawn_run(body)
                return JSONResponse(status_code=202, content={"status": "accepted", "run_id": run_id})
            except Exception as e:
                logger.exception("Spawn run failed")
                return JSONResponse(status_code=400, content={"error": str(e)})
        if run_config_fn is None:
            return JSONResponse(
                status_code=501,
                content={"error": "Run from config not available (start with --ui to enable)."},
            )
        try:
            run_config_fn(body, collector)
            return JSONResponse(status_code=202, content={"status": "accepted"})
        except Exception as e:
            logger.exception("Run from config failed")
            return JSONResponse(
                status_code=400,
                content={"error": str(e)},
            )

    # -- Runs (historical) -----------------------------------------------------

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

    # -- Templates --------------------------------------------------------------

    @app.get("/api/templates")
    def api_list_templates():
        return list_templates()

    @app.get("/api/templates/{template_id}")
    def api_get_template(template_id: str):
        t = get_template(template_id)
        if t is None:
            return JSONResponse(status_code=404, content={"error": "template not found"})
        return t

    @app.post("/api/templates")
    def api_save_template(body: dict):
        name = body.pop("_template_name", body.get("experiment_name", "template"))
        tid = save_template(name, body)
        return {"id": tid}

    @app.delete("/api/templates/{template_id}")
    def api_delete_template(template_id: str):
        ok = delete_template(template_id)
        if not ok:
            return JSONResponse(status_code=404, content={"error": "template not found"})
        return {"deleted": True}

    # -- Active runs (process-based) -------------------------------------------

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

    @app.post("/api/hw_config_verify")
    def api_hw_config_verify(body: dict):
        """Verify that a hardware core config can fit the given model's softcores.

        Request body:
            model_repr_json: {
                model_type: str,
                input_shape: list[int],
                num_classes: int,
                model_config: dict,
                max_axons: int,
                max_neurons: int,
                threshold_groups: int,
                pruning_fraction: float,
                threshold_seed: int,
            }
            core_types: [{max_axons, max_neurons, count}, ...]
        """
        try:
            from mimarsinan.mapping.mapping_verifier import (
                verify_soft_core_mapping,
                verify_hardware_config,
            )
            mr = dict(body.get("model_repr_json", {}))
            # Ensure native models are built with a reasonable minimum layer size.
            # _get_softcores_from_request uses _LAYOUT_PASS_LIMIT for the layout pass itself.
            mr["max_axons"] = max(int(mr.get("max_axons", 1024)), 4096)
            mr["max_neurons"] = max(int(mr.get("max_neurons", 1024)), 4096)
            softcores = _get_softcores_from_request(mr)
            core_types = body.get("core_types", [])
            allow_neuron_splitting = bool(body.get("allow_neuron_splitting", False))
            allow_coalescing = bool(body.get("allow_coalescing", False))
            result = verify_hardware_config(
                softcores, core_types,
                allow_neuron_splitting=allow_neuron_splitting,
                allow_axon_coalescing=allow_coalescing,
            )
            return {
                "feasible": result["feasible"],
                "errors": result["errors"],
                "field_errors": result["field_errors"],
                "packing": {
                    "cores_used": result["packing_result"].cores_used if result["packing_result"] else 0,
                    "total_capacity": result["packing_result"].total_capacity if result["packing_result"] else 0,
                    "used_area": result["packing_result"].used_area if result["packing_result"] else 0,
                } if result["packing_result"] else None,
            }
        except Exception as e:
            logger.exception("hw_config_verify failed")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/hw_config_auto")
    def api_hw_config_auto(body: dict):
        """Auto-suggest a hardware configuration for the given model.

        Request body keys (from wizard state):
            model_type: str
            input_shape: list[int]
            num_classes: int
            model_config: dict        (builder configuration)
            max_axons: int            (upper bound for layout pass)
            max_neurons: int          (upper bound for layout pass)
            threshold_groups: int     (default 1)
            pruning_fraction: float   (default 0.0)
            threshold_seed: int       (default 0)
            allow_coalescing: bool    (default false)
            hardware_bias: bool       (default true)
            axon_granularity: int     (default 1)
            neuron_granularity: int   (default 1)
        """
        try:
            from mimarsinan.mapping.hw_config_suggester import suggest_hardware_config
            # Ensure native models are built with a reasonable minimum layer size.
            # _get_softcores_from_request uses _LAYOUT_PASS_LIMIT for the layout pass itself.
            layout_body = dict(body)
            layout_body["max_axons"] = max(int(body.get("max_axons", 1024)), 4096)
            layout_body["max_neurons"] = max(int(body.get("max_neurons", 1024)), 4096)
            softcores = _get_softcores_from_request(layout_body)
            suggestion = suggest_hardware_config(
                softcores,
                allow_coalescing=bool(body.get("allow_coalescing", False)),
                hardware_bias=bool(body.get("hardware_bias", True)),
                axon_granularity=int(body.get("axon_granularity", 1)),
                neuron_granularity=int(body.get("neuron_granularity", 1)),
                safety_margin=float(body.get("safety_margin", 0.15)),
                allow_neuron_splitting=bool(body.get("allow_neuron_splitting", False)),
            )
            return {
                "core_types": suggestion.core_types,
                "total_cores": suggestion.total_cores,
                "rationale": suggestion.rationale,
            }
        except Exception as e:
            logger.exception("hw_config_auto failed")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/pipeline_steps")
    def api_pipeline_steps(body: dict):
        """Return ordered pipeline step names for the given deployment config (wizard preview)."""
        from mimarsinan.pipelining.pipelines.deployment_pipeline import (
            DeploymentPipeline,
            get_pipeline_step_specs,
        )
        try:
            deployment_parameters = dict(body.get("deployment_parameters", {}))
            pipeline_mode = body.get("pipeline_mode", "vanilla")
            DeploymentPipeline.apply_preset(pipeline_mode, deployment_parameters)
            config = dict(DeploymentPipeline.default_deployment_parameters)
            config.update(deployment_parameters)
            config.update(DeploymentPipeline.default_platform_constraints)
            platform = body.get("platform_constraints") or {}
            if isinstance(platform, dict):
                config.update(platform)
            specs = get_pipeline_step_specs(config)
            return {"steps": [name for name, _ in specs]}
        except Exception as e:
            logger.debug("pipeline_steps failed: %s", e)
            return JSONResponse(
                status_code=400,
                content={"error": str(e)},
            )

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


def _port_is_free(host: str, port: int) -> bool:
    import socket
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
    """Start the FastAPI server in a background daemon thread.

    If *port* is already in use, tries successive ports up to
    *port + max_port_attempts - 1* before giving up.

    If *run_config_fn* is provided (e.g. when started with --ui), POST /api/run
    will run the pipeline from the request body and attach to this collector.
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

    pm = None
    if run_config_fn is not None:
        from mimarsinan.gui.process_manager import ProcessManager
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
    logger.info("GUI server started on http://%s:%d", host, chosen_port)
    print(f"\n  Pipeline Monitor GUI: http://localhost:{chosen_port}\n")
    if run_config_fn is not None:
        print(f"  Wizard: http://localhost:{chosen_port}/wizard\n")
    return thread
