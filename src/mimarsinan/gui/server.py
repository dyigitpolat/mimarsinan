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


def _get_softcores_from_request(body: dict):
    """Build a model repr and run layout mapping, returning LayoutSoftCoreSpec list.

    Handles both native (simple_mlp) and torch (torch_sequential_linear, etc.) builders
    by delegating to the appropriate builder class.
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

    result = verify_soft_core_mapping(
        model_repr,
        max_axons=max_axons,
        max_neurons=max_neurons,
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
) -> FastAPI:
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
            # Use large fixed bounds so softcores reflect true model dimensions,
            # consistent with the auto-config pass.
            mr["max_axons"] = max(int(mr.get("max_axons", 1024)), 4096)
            mr["max_neurons"] = max(int(mr.get("max_neurons", 1024)), 4096)
            softcores = _get_softcores_from_request(mr)
            core_types = body.get("core_types", [])
            result = verify_hardware_config(softcores, core_types)
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
            # Use a large fixed bound for the layout pass so softcores reflect the true
            # model dimensions (not the previously-suggested, potentially-smaller bounds).
            # This breaks the circular dependency where verify re-runs layout with the
            # newly-suggested smaller dimensions and gets different softcores.
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

    app = create_app(collector, run_config_fn=run_config_fn)

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
