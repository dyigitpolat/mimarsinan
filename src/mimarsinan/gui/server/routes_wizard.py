"""Wizard configuration and deployment API routes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mimarsinan.gui.runs import get_runs_root, _validate_run_id
from mimarsinan.gui.server.json_safe import SafeJSONResponse
from mimarsinan.gui.templates import (
    delete_template,
    get_template,
    list_templates,
    name_and_deployment_from_post_body,
    save_template,
)

if TYPE_CHECKING:
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.runtime.process_manager import ProcessManager

logger = logging.getLogger("mimarsinan.gui")


def register_routes(
    app: FastAPI,
    *,
    collector: "DataCollector",
    run_config_fn: Callable[[dict, "DataCollector"], None] | None,
    process_manager: "ProcessManager | None",
) -> None:
    _metadata_cache: dict[tuple, dict] = {}
    _metadata_cache_lock = threading.Lock()

    @app.get("/api/wizard/schema")
    def api_wizard_schema():
        from mimarsinan.gui.wizard.schema import get_wizard_defaults, get_wizard_nas_schema
        return {
            "nas": get_wizard_nas_schema(),
            "defaults": get_wizard_defaults(),
        }

    @app.get("/api/data_providers")
    def api_data_providers():
        from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
        return BasicDataProviderFactory.list_registered()

    @app.get("/api/data_providers/{provider_id}/metadata")
    async def api_data_provider_metadata(
        provider_id: str,
        resize_to: int | None = None,
        normalize: str | None = None,
        interpolation: str | None = None,
        datasets_path: str | None = None,
    ):
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
        from mimarsinan.pipelining.core.registry.model_registry import get_model_types
        return get_model_types()

    @app.get("/api/model_config_schema/{model_type}")
    def api_model_config_schema(model_type: str):
        from mimarsinan.pipelining.core.registry.model_registry import get_model_config_schema
        return get_model_config_schema(model_type)

    @app.post("/api/run")
    def api_run(body: dict, request: Request):
        from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state
        from mimarsinan.gui.wizard.validation import validate_wizard_state

        raw = body or {}
        if request.query_params.get("validate") == "1":
            errs = validate_wizard_state(raw)
            if errs:
                return JSONResponse(
                    status_code=400,
                    content={"error": "validation failed", "field_errors": errs},
                )

        body = build_deployment_config_from_state(raw)
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

    @app.get("/api/runs/{run_id}/discovered")
    def api_run_discovered(run_id: str):
        _validate_run_id(run_id)
        run_dir = os.path.join(get_runs_root(), run_id)
        pop_path = os.path.join(run_dir, "final_population.json")
        if not os.path.isfile(pop_path):
            return SafeJSONResponse(content={"discovered": False})
        try:
            with open(pop_path, encoding="utf-8") as f:
                data = json.load(f)
            best = data.get("best", {})
            cfg = best.get("configuration", {})
            return SafeJSONResponse(content={
                "discovered": True,
                "search_mode_used": data.get("search_mode_used"),
                "discovered_model_config": data.get("discovered_model_config") or cfg.get("model_config"),
                "discovered_platform_constraints": data.get("discovered_platform_constraints") or cfg.get("platform_constraints"),
                "active_objectives": data.get("active_objectives", []),
                "best_objectives": best.get("objectives", {}),
            })
        except Exception:
            return SafeJSONResponse(content={"discovered": False})

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

    @app.post("/api/pipeline_steps")
    def api_pipeline_steps(body: dict):
        from mimarsinan.config_schema import build_flat_pipeline_config
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
            DeploymentPipeline,
            get_pipeline_step_specs,
            get_pipeline_semantic_group_by_step_name,
        )
        try:
            deployment_parameters = dict(body.get("deployment_parameters", {}))
            pipeline_mode = body.get("pipeline_mode", "vanilla")
            DeploymentPipeline.apply_preset(pipeline_mode, deployment_parameters)
            platform = body.get("platform_constraints") or {}
            config = build_flat_pipeline_config(
                deployment_parameters,
                platform if isinstance(platform, dict) else None,
                pipeline_mode=pipeline_mode,
            )
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
