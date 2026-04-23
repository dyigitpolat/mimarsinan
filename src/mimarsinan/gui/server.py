"""FastAPI server for the pipeline monitoring GUI.

Runs in a daemon thread so the pipeline is not blocked.  Provides REST
endpoints for polling and a WebSocket for real-time push.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
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


def _get_layout_result_from_request(
    body: dict,
    *,
    tiling_max_axons: int | None = None,
    tiling_max_neurons: int | None = None,
):
    """Build a model repr and run layout mapping, returning the verification result.

    Handles both native (simple_mlp) and torch (torch_sequential_linear, etc.) builders
    by delegating to the appropriate builder class.

    ``tiling_max_axons`` / ``tiling_max_neurons`` override the layout pass's
    tiling bounds so the wizard can match the pipeline's ``max(core_types)``
    resolution.  When omitted, ``body["max_axons"]``/``body["max_neurons"]``
    are used (appropriate for the auto-suggester path which has no fixed
    core_types yet).
    """
    from mimarsinan.mapping.mapping_verifier import verify_soft_core_mapping
    from mimarsinan.models.builders import BUILDERS_REGISTRY
    from mimarsinan.pipelining.model_registry import ModelRegistry
    import torch

    model_type = body.get("model_type", "simple_mlp")
    input_shape = tuple(int(x) for x in body.get("input_shape", [1, 28, 28]))
    num_classes = int(body.get("num_classes", 10))
    model_config = body.get("model_config", {})
    allow_coalescing = bool(body.get("allow_coalescing", False))
    hardware_bias = bool(body.get("hardware_bias", False))
    effective_max_axons = int(
        tiling_max_axons if tiling_max_axons is not None else body.get("max_axons", 1024)
    )
    effective_max_neurons = int(
        tiling_max_neurons if tiling_max_neurons is not None else body.get("max_neurons", 1024)
    )

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

    # Populate perceptron_index on the mapper graph so threshold groups
    # are assigned consistently with SoftCoreMappingStep.
    if hasattr(model_repr, "assign_perceptron_indices"):
        model_repr.assign_perceptron_indices()

    # Use the same tiling bounds that the real pipeline uses (max across
    # the user's core types), so LayoutIRMapping produces byte-identical
    # softcore shapes to IRMapping.
    result = verify_soft_core_mapping(
        model_repr,
        max_axons=effective_max_axons,
        max_neurons=effective_max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    )
    if not result.feasible:
        raise ValueError(f"Soft-core mapping verification failed: {result.error}")
    return result


def _get_softcores_from_request(body: dict):
    """Build a model repr and run layout mapping, returning LayoutSoftCoreSpec list."""
    return _get_layout_result_from_request(body).softcores


_RESOURCE_MEDIA_TYPE_BY_KIND = {
    # Heatmap kinds → image/png
    "ir_core_heatmap": "image/png",
    "ir_core_pre_pruning": "image/png",
    "ir_bank_heatmap": "image/png",
    "hard_core_heatmap": "image/png",
    "pruning_layer_heatmap": "image/png",
    # JSON resource kinds
    "connectivity": "application/json",
}


def _serve_resource_from_disk(
    working_dir: str | None,
    step_name: str,
    kind: str,
    rid: str,
) -> Response:
    """Load a resource file written by the snapshot executor and return it
    with the correct Content-Type. Used for cross-process resource serving
    (historical runs and subprocess-spawned active runs that persist lazy
    outputs to ``_GUI_STATE/resources/``).
    """
    from mimarsinan.gui.persistence import load_resource_from_disk

    media_type = _RESOURCE_MEDIA_TYPE_BY_KIND.get(kind)
    if media_type is None:
        return JSONResponse(status_code=404, content={"error": f"unknown resource kind {kind!r}"})
    if not working_dir:
        return JSONResponse(status_code=404, content={"error": "run not found"})
    payload = load_resource_from_disk(working_dir, step_name, kind, rid, media_type=media_type)
    if payload is None:
        return JSONResponse(status_code=404, content={"error": "resource not found"})
    headers = {"Cache-Control": "public, max-age=3600, immutable"}
    return Response(content=payload, media_type=media_type, headers=headers)


def create_app(
    collector: "DataCollector",
    run_config_fn: Callable[[dict, "DataCollector"], None] | None = None,
    process_manager: "ProcessManager | None" = None,
) -> FastAPI:
    from mimarsinan.gui.runs import (
        list_runs, get_run_config, get_run_pipeline,
        get_run_step_detail as hist_step_detail,
        get_run_console_logs,
        get_runs_root, _validate_run_id,
    )
    from mimarsinan.gui.persistence import load_console_logs
    from mimarsinan.gui.templates import (
        delete_template,
        get_template,
        list_templates,
        name_and_deployment_from_post_body,
        save_template,
    )
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
    def step_detail(step_name: str, request: Request, since_seq: int = 0):
        """Step detail with HTTP ETag caching + ``since_seq`` metric pagination.

        * ``If-None-Match`` → 304 when the ``snapshot_etag`` matches. The
          frontend uses this during idle live-training polls so unchanged
          snapshots don't re-ship the whole payload.
        * ``?since_seq=N`` filters the ``metrics`` list to ``seq > N``. The
          ``latest_metric_seq`` in the payload lets the client advance its
          cursor even when the filtered list is empty.
        """
        etag = collector.get_step_snapshot_etag(step_name)
        if etag is None:
            return JSONResponse(status_code=404, content={"error": "step not found"})
        inm = request.headers.get("if-none-match")
        if inm and inm == etag:
            return Response(status_code=304, headers={"ETag": etag})
        detail = collector.get_step_detail(step_name, since_seq=since_seq)
        if detail is None:
            return JSONResponse(status_code=404, content={"error": "step not found"})
        return _SafeJSONResponse(content=detail, headers={"ETag": etag})

    @app.get("/api/steps/{step_name}/metrics")
    def step_metrics(step_name: str):
        return collector.get_step_metrics(step_name)

    # -- Lazy resources (heatmaps / connectivity for the live run) ------------

    @app.get("/api/steps/{step_name}/resources/{kind}/{rid:path}")
    def step_resource(step_name: str, kind: str, rid: str):
        """Fetch a lazy resource for a live step.

        Tries the in-memory :class:`ResourceStore` first (live run owned
        by this server process); falls back to the on-disk persisted
        copy (populated by the snapshot executor) so resources keep
        working after server restarts.
        """
        store = collector.get_resource_store()
        media_type = _RESOURCE_MEDIA_TYPE_BY_KIND.get(kind)
        if media_type is None:
            return JSONResponse(status_code=404, content={"error": f"unknown resource kind {kind!r}"})
        if store is not None:
            if media_type == "image/png":
                hit = store.get_bytes(step_name, kind, rid)
                if hit is not None:
                    payload, mt = hit
                    return Response(
                        content=payload,
                        media_type=mt,
                        headers={"Cache-Control": "public, max-age=3600, immutable"},
                    )
            else:
                payload = store.get_json(step_name, kind, rid)
                if payload is not None:
                    return _SafeJSONResponse(
                        content=payload,
                        headers={"Cache-Control": "public, max-age=3600, immutable"},
                    )
        working_dir = None
        try:
            from mimarsinan.pipelining.pipelines.deployment_pipeline import DeploymentPipeline  # noqa: F401
        except Exception:
            pass
        # No explicit working_dir is exposed here — disk fallback is only
        # available for subprocess/historical runs (handled by the mirror
        # endpoints below). For the primary collector, if the in-memory
        # store missed, the resource simply isn't available.
        return JSONResponse(status_code=404, content={"error": "resource not found"})

    @app.get("/api/metrics")
    def all_metrics():
        return collector.get_all_metrics()

    @app.get("/api/config")
    def pipeline_config():
        return collector.pipeline_config or {}

    @app.get("/api/console")
    def api_console_logs(offset: int = 0):
        return collector.get_console_logs(offset=offset)

    # -- Wizard / config APIs (used when run_config_fn is set, e.g. --ui mode) ----

    @app.get("/api/wizard/schema")
    def api_wizard_schema():
        from mimarsinan.gui.wizard.schema import get_wizard_nas_schema
        return {"nas": get_wizard_nas_schema()}

    @app.get("/api/data_providers")
    def api_data_providers():
        from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
        return BasicDataProviderFactory.list_registered()

    # In-process cache so the wizard's per-keystroke metadata fetch never
    # re-downloads datasets or re-reads ImageNet's devkit.  Key is the full
    # tuple the client can vary; entries never expire (registry is fixed).
    _metadata_cache: dict[tuple, dict] = {}
    _metadata_cache_lock = threading.Lock()

    @app.get("/api/data_providers/{provider_id}/metadata")
    async def api_data_provider_metadata(
        provider_id: str,
        resize_to: int | None = None,
        normalize: str | None = None,
        interpolation: str | None = None,
        datasets_path: str | None = None,
    ):
        """Return preprocessing-aware ``{input_shape, num_classes, ...}``.

        Runs the (potentially dataset-loading) provider instantiation on a
        worker thread so the FastAPI threadpool stays available for WS
        broadcasts and other requests.  Results are cached in-process.
        """
        from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
        preprocessing = None
        if resize_to is not None or normalize:
            preprocessing = {}
            if resize_to is not None and int(resize_to) > 0:
                preprocessing["resize_to"] = int(resize_to)
            if normalize:
                preprocessing["normalize"] = normalize
            if interpolation:
                preprocessing["interpolation"] = interpolation
            if not preprocessing:
                preprocessing = None

        cache_key = (
            provider_id,
            datasets_path or "./datasets",
            resize_to if resize_to and int(resize_to) > 0 else None,
            (normalize or None),
            (interpolation or None),
        )
        with _metadata_cache_lock:
            hit = _metadata_cache.get(cache_key)
        if hit is not None:
            return hit

        def _load():
            return BasicDataProviderFactory.get_metadata(
                provider_id,
                datasets_path or "./datasets",
                preprocessing=preprocessing,
            )

        try:
            result = await asyncio.to_thread(_load)
        except ValueError as e:
            return JSONResponse(status_code=404, content={"error": str(e)})
        except Exception as e:
            logger.exception("data_provider_metadata failed")
            return JSONResponse(status_code=503, content={"error": str(e)})

        with _metadata_cache_lock:
            _metadata_cache[cache_key] = result
        return result

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

    @app.get("/api/runs/{run_id}/console")
    def api_run_console(run_id: str, offset: int = 0):
        return get_run_console_logs(run_id, offset=offset)

    @app.get("/api/runs/{run_id}/steps/{step_name}/resources/{kind}/{rid:path}")
    def api_run_step_resource(run_id: str, step_name: str, kind: str, rid: str):
        """Serve a lazy resource for a historical run from its on-disk cache.

        Legacy runs that were persisted before the lazy-resource split
        naturally return 404 — they never wrote the standalone PNG/JSON
        files (this matches the ``newonly`` rollout decision).
        """
        try:
            _validate_run_id(run_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "invalid run_id"})
        run_dir = os.path.join(get_runs_root(), run_id)
        if not os.path.isdir(run_dir):
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return _serve_resource_from_disk(run_dir, step_name, kind, rid)

    @app.get("/api/runs/{run_id}/discovered")
    def api_run_discovered(run_id: str):
        """Return discovered architectural parameters from a previous search run.

        Used by edit-and-continue to pre-populate the wizard with values
        found by architecture search.
        """
        from mimarsinan.gui.runs import get_runs_root, _validate_run_id
        _validate_run_id(run_id)
        run_dir = os.path.join(get_runs_root(), run_id)
        pop_path = os.path.join(run_dir, "final_population.json")
        if not os.path.isfile(pop_path):
            return _SafeJSONResponse(content={"discovered": False})
        try:
            import json as _json
            with open(pop_path, encoding="utf-8") as f:
                data = _json.load(f)
            best = data.get("best", {})
            cfg = best.get("configuration", {})
            return _SafeJSONResponse(content={
                "discovered": True,
                "search_mode_used": data.get("search_mode_used"),
                "discovered_model_config": data.get("discovered_model_config") or cfg.get("model_config"),
                "discovered_platform_constraints": data.get("discovered_platform_constraints") or cfg.get("platform_constraints"),
                "active_objectives": data.get("active_objectives", []),
                "best_objectives": best.get("objectives", {}),
            })
        except Exception:
            return _SafeJSONResponse(content={"discovered": False})

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
        name, config = name_and_deployment_from_post_body(body)
        tid = save_template(name, config)
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

    @app.get("/api/active_runs/{run_id}/steps/{step_name}/resources/{kind}/{rid:path}")
    def api_active_step_resource(run_id: str, step_name: str, kind: str, rid: str):
        """Serve a lazy resource for a subprocess-spawned active run.

        The child writes resources to ``_GUI_STATE/resources/...`` via
        the snapshot executor; the parent server reads those files here.
        """
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        working_dir = process_manager.get_working_dir(run_id)
        if working_dir is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return _serve_resource_from_disk(working_dir, step_name, kind, rid)

    @app.get("/api/active_runs/{run_id}/console")
    def api_active_console(run_id: str, offset: int = 0):
        if process_manager is None:
            return JSONResponse(status_code=404, content={"error": "not found"})
        managed = process_manager._runs.get(run_id)
        if managed is None:
            return JSONResponse(status_code=404, content={"error": "run not found"})
        return load_console_logs(managed.working_dir, offset=offset)

    def _expand_preview_for_scheduling(preview, per_segment_passes, per_segment_pass_lists):
        """Expand layout_preview flow to show schedule pass boundaries.

        Uses the **actual pass lists** from the unified partitioner so the
        miniview shows the same partition that hard-core mapping will apply.

        Each neural segment that requires >1 pass gets its softcores
        distributed across passes according to the partitioner output,
        with sync barriers inserted between consecutive passes.
        """
        if not preview or not preview.get("flow"):
            return preview

        old_flow = preview["flow"]

        # --- Phase 1: identify neural segments in the flow ---
        segments = []
        current_seg_neural = []
        current_seg_id = 0

        for item in old_flow:
            if item.get("kind") == "neural":
                current_seg_neural.append(item)
            elif item.get("kind") == "host":
                if current_seg_neural:
                    segments.append((current_seg_id, current_seg_neural))
                    current_seg_neural = []
                    current_seg_id += 1

        if current_seg_neural:
            segments.append((current_seg_id, current_seg_neural))

        # --- Phase 2: build per-segment pass assignments from partitioner ---
        seg_pass_assignments = {}
        for seg_id, neural_items in segments:
            n_passes = per_segment_passes.get(seg_id, per_segment_passes.get(str(seg_id), 1))
            pass_lists = per_segment_pass_lists.get(seg_id, per_segment_pass_lists.get(str(seg_id)))

            if n_passes <= 1 or not pass_lists or len(pass_lists) <= 1:
                seg_pass_assignments[seg_id] = [(0, neural_items)]
                continue

            # Use the partitioner's authoritative pass lists directly.
            # Each pass_list entry is a list of LayoutSoftCoreSpec objects;
            # len(pl) is the exact softcore count for that pass.
            passes = []
            for pi, pass_list in enumerate(pass_lists):
                if neural_items:
                    template = dict(neural_items[0])
                else:
                    template = {"kind": "neural", "latency_group_index": 0, "latency_tag": 0, "softcore_count": 0, "segment_count": 1}
                template["softcore_count"] = len(pass_list)
                passes.append((pi, [template]))

            seg_pass_assignments[seg_id] = passes

        # --- Phase 3: rebuild flow with sync barriers ---
        new_flow = [{"kind": "input"}]
        seg_idx = 0
        saw_neural_in_segment = False
        emitted_segments = set()
        for item in old_flow:
            if item.get("kind") == "input" or item.get("kind") == "output":
                continue
            if item.get("kind") == "host":
                new_flow.append(item)
                if saw_neural_in_segment:
                    seg_idx += 1
                    saw_neural_in_segment = False
                continue
            if item.get("kind") == "neural":
                saw_neural_in_segment = True
                passes_for_seg = seg_pass_assignments.get(seg_idx)
                if passes_for_seg is None:
                    new_flow.append(item)
                    continue

                if seg_idx not in emitted_segments:
                    emitted_segments.add(seg_idx)
                    for pi, (pass_id, pass_items) in enumerate(passes_for_seg):
                        if pi > 0:
                            new_flow.append({
                                "kind": "host",
                                "slot": -1,
                                "compute_op_count": 0,
                                "schedule_sync": True,
                            })
                        for neural_item in pass_items:
                            new_flow.append(neural_item)
                continue

        new_flow.append({"kind": "output"})

        schedule_syncs = sum(
            1 for item in new_flow
            if item.get("kind") == "host" and item.get("schedule_sync")
        )

        result = dict(preview)
        result["flow"] = new_flow
        result["schedule_sync_count"] = schedule_syncs
        return result

    @app.post("/api/hw_config_verify")
    async def api_hw_config_verify(body: dict):
        """Verify that a hardware core config can fit the given model's softcores.

        Heavy (full layout + pack).  Offloaded via ``asyncio.to_thread`` so
        a rapid-fire sequence of wizard edits (debounced at 250 ms) cannot
        saturate the default FastAPI threadpool and stall every other
        request including the ones feeding the monitor UI.
        """
        def _run():
            from mimarsinan.mapping.mapping_verifier import (
                verify_soft_core_mapping,
                verify_hardware_config,
            )
            mr = dict(body.get("model_repr_json", {}))
            core_types = body.get("core_types", [])
            # Tiling must match what the pipeline's SoftCoreMappingStep does:
            # max across all user-defined core types.
            if core_types:
                tile_max_ax = max(int(ct["max_axons"]) for ct in core_types)
                tile_max_neu = max(int(ct["max_neurons"]) for ct in core_types)
                mr["allow_coalescing"] = bool(body.get("allow_coalescing", False))
                mr["hardware_bias"] = all(
                    bool(ct.get("has_bias", True)) for ct in core_types
                )
            else:
                tile_max_ax = int(mr.get("max_axons", 1024))
                tile_max_neu = int(mr.get("max_neurons", 1024))
            # Ensure native models are built with a reasonable minimum layer size.
            mr["max_axons"] = max(int(mr.get("max_axons", 1024)), 4096)
            mr["max_neurons"] = max(int(mr.get("max_neurons", 1024)), 4096)
            layout_result = _get_layout_result_from_request(
                mr,
                tiling_max_axons=tile_max_ax,
                tiling_max_neurons=tile_max_neu,
            )
            softcores = layout_result.softcores
            allow_neuron_splitting = bool(body.get("allow_neuron_splitting", False))
            allow_coalescing = bool(body.get("allow_coalescing", False))
            allow_scheduling = bool(body.get("allow_scheduling", False))
            result = verify_hardware_config(
                softcores, core_types,
                allow_neuron_splitting=allow_neuron_splitting,
                allow_coalescing=allow_coalescing,
                allow_scheduling=allow_scheduling,
            )
            stats_out = {
                **(result.get("stats") or {}),
                "host_side_segment_count": layout_result.host_side_segment_count,
                "layout_preview": layout_result.layout_preview,
            }

            # When scheduling is active, expand the miniview flow to show
            # per-segment pass boundaries as sync barriers.
            si = result.get("schedule_info")
            if si and si.get("per_segment_passes"):
                stats_out["layout_preview"] = _expand_preview_for_scheduling(
                    layout_result.layout_preview,
                    si["per_segment_passes"],
                    si.get("per_segment_pass_lists", {}),
                )

            resp = {
                "feasible": result["feasible"],
                "errors": result["errors"],
                "field_errors": result["field_errors"],
                "packing": {
                    "cores_used": result["packing_result"].cores_used if result["packing_result"] else 0,
                    "total_capacity": result["packing_result"].total_capacity if result["packing_result"] else 0,
                    "used_area": result["packing_result"].used_area if result["packing_result"] else 0,
                } if result["packing_result"] else None,
                "stats": stats_out,
            }
            if "schedule_info" in result:
                si_clean = {
                    k: v for k, v in result["schedule_info"].items()
                    if k != "per_segment_pass_lists"
                }
                resp["schedule_info"] = si_clean
            return resp

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.exception("hw_config_verify failed")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/hw_config_auto")
    async def api_hw_config_auto(body: dict):
        """Auto-suggest a hardware configuration for the given model.

        Heavy (greedy pack + layout per iteration).  Offloaded the same
        way as ``hw_config_verify`` so the FastAPI threadpool stays free
        for WS broadcasts and other REST calls under rapid wizard edits.
        """
        def _run():
            from mimarsinan.mapping.hw_config_suggester import suggest_hardware_config, suggest_hardware_config_scheduled
            layout_body = dict(body)
            layout_body["max_axons"] = max(int(body.get("max_axons", 1024)), 4096)
            layout_body["max_neurons"] = max(int(body.get("max_neurons", 1024)), 4096)
            softcores = _get_softcores_from_request(layout_body)

            common_kwargs = dict(
                allow_coalescing=bool(body.get("allow_coalescing", False)),
                hardware_bias=bool(body.get("hardware_bias", True)),
                axon_granularity=int(body.get("axon_granularity", 1)),
                neuron_granularity=int(body.get("neuron_granularity", 1)),
                safety_margin=float(body.get("safety_margin", 0.15)),
                allow_neuron_splitting=bool(body.get("allow_neuron_splitting", False)),
            )

            if bool(body.get("allow_scheduling", False)):
                suggestion = suggest_hardware_config_scheduled(
                    softcores,
                    max_passes=int(body.get("max_schedule_passes", 8)),
                    latency_weight=float(body.get("scheduling_latency_weight", 1.0)),
                    **common_kwargs,
                )
            else:
                suggestion = suggest_hardware_config(softcores, **common_kwargs)

            return {
                "core_types": suggestion.core_types,
                "total_cores": suggestion.total_cores,
                "rationale": suggestion.rationale,
                "num_passes": suggestion.num_passes,
                "estimated_latency_multiplier": suggestion.estimated_latency_multiplier,
            }

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.exception("hw_config_auto failed")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/pipeline_steps")
    def api_pipeline_steps(body: dict):
        """Return ordered pipeline step names and semantic groups for the given deployment config (wizard preview)."""
        from mimarsinan.pipelining.pipelines.deployment_pipeline import (
            DeploymentPipeline,
            get_pipeline_step_specs,
            get_pipeline_semantic_group_by_step_name,
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
            groups = get_pipeline_semantic_group_by_step_name(config)
            return {
                "steps": [name for name, _ in specs],
                "semantic_groups": [groups.get(name, "other") for name, _ in specs],
            }
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
                raw = await ws.receive_text()
                # Clients send ``{"type":"resume","last_seq":N}`` right
                # after (re)connect to get any lifecycle events that were
                # broadcast while the socket was down.  Parse is best-effort
                # — garbled messages are ignored so the listener stays up.
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

    # -- Real-time streaming for subprocess-spawned active runs ----------------
    #
    # Previously the frontend polled ``/api/active_runs/{rid}/pipeline`` and
    # ``.../steps/{name}`` every 3 s, so every metric arrived in visibly
    # batched flushes. ``ActiveRunHub`` tails the subprocess's
    # ``live_metrics.jsonl`` and ``steps.json`` files and pushes per-line
    # events over this WS so the charts update at ~20 Hz instead.
    if process_manager is not None:
        from mimarsinan.gui.active_run_stream import ActiveRunHub

        def _active_overview(run_id: str):
            try:
                return process_manager.get_run_detail(run_id)
            except Exception:
                return None

        active_hub = ActiveRunHub(
            get_working_dir=process_manager.get_working_dir,
            build_overview=_active_overview,
        )
        # Expose on the app so tests and shutdown hooks can inspect it.
        app.state.active_run_hub = active_hub

        @app.websocket("/ws/active_runs/{run_id}")
        async def active_run_ws(ws: WebSocket, run_id: str):
            await ws.accept()
            loop = asyncio.get_event_loop()

            def _send(msg: dict) -> None:
                # Tailer threads call this; schedule the actual write on
                # the event loop so slow clients never stall the tailer.
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


def _gui_entry_url(port: int) -> str:
    """HTTP URL for the welcome page (``/``) shown and opened after the GUI server starts."""
    return f"http://127.0.0.1:{port}/"


def _schedule_open_browser(url: str) -> None:
    """Open *url* in the default browser after a short delay so the server is accepting.

    Uses :func:`webbrowser.open` (``xdg-open`` on Linux, ``open`` on macOS, etc.).
    Set ``MIMARSINAN_GUI_NO_BROWSER=1`` to skip (e.g. CI or headless environments).
    """
    if os.environ.get("MIMARSINAN_GUI_NO_BROWSER", "").strip().lower() in ("1", "true", "yes"):
        return

    def _open() -> None:
        try:
            webbrowser.open(url)
        except Exception:
            logger.debug("Could not open browser for %s", url, exc_info=True)

    threading.Timer(0.6, _open).start()


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
    logger.info("GUI server started (bind %s:%d)", host, chosen_port)
    entry_url = _gui_entry_url(chosen_port)
    print(f"\n  Mimarsinan is running at: {entry_url}\n")
    _schedule_open_browser(entry_url)
    return thread
