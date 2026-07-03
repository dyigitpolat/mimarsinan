"""Hardware layout verification and auto-suggest API routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger("mimarsinan.gui")


def get_layout_result_from_request(
    body: dict,
    *,
    tiling_max_axons: int | None = None,
    tiling_max_neurons: int | None = None,
):
    """Build model repr and run layout-mapping verification (cached per identical body)."""
    from mimarsinan.mapping.verification.layout_mapping_service import (
        DEFAULT_LAYOUT_MAPPING_SERVICE,
    )
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    request = LayoutMappingRequest.from_wizard_body(
        body,
        tiling_max_axons=tiling_max_axons,
        tiling_max_neurons=tiling_max_neurons,
    )
    result = DEFAULT_LAYOUT_MAPPING_SERVICE.get_verification(request)
    if not result.feasible:
        raise ValueError(f"Soft-core mapping verification failed: {result.error}")
    return result


def get_softcores_from_request(body: dict):
    """Build a model repr and run layout mapping, returning LayoutSoftCoreSpec list."""
    return get_layout_result_from_request(body).softcores


def expand_preview_for_scheduling(preview, per_segment_passes, per_segment_pass_lists):
    """Expand layout_preview flow to show schedule pass boundaries."""
    if not preview or not preview.get("flow"):
        return preview

    old_flow = preview["flow"]

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

    seg_pass_assignments = {}
    for seg_id, neural_items in segments:
        n_passes = per_segment_passes.get(seg_id, per_segment_passes.get(str(seg_id), 1))
        pass_lists = per_segment_pass_lists.get(seg_id, per_segment_pass_lists.get(str(seg_id)))

        if n_passes <= 1 or not pass_lists or len(pass_lists) <= 1:
            seg_pass_assignments[seg_id] = [(0, neural_items)]
            continue

        item_by_latency = {
            int(it.get("latency_tag", 0)): it
            for it in neural_items
            if it.get("kind") == "neural" and it.get("latency_tag") is not None
        }
        fallback_template = (
            dict(neural_items[0]) if neural_items else
            {"kind": "neural", "latency_group_index": 0, "latency_tag": 0,
             "softcore_count": 0, "segment_count": 1}
        )

        passes = []
        for pi, pass_list in enumerate(pass_lists):
            bucket: Dict[int, int] = {}
            for sc in pass_list:
                lat = int(sc.latency_tag) if getattr(sc, "latency_tag", None) is not None else 0
                bucket[lat] = bucket.get(lat, 0) + 1
            pass_items = []
            for lat in sorted(bucket.keys()):
                template = dict(item_by_latency.get(lat, fallback_template))
                template["latency_tag"] = lat
                template["softcore_count"] = bucket[lat]
                if lat not in item_by_latency:
                    template.setdefault("latency_group_index", len(item_by_latency) + lat)
                pass_items.append(template)
            if not pass_items:
                pass_items = [dict(fallback_template) | {"softcore_count": len(pass_list)}]
            passes.append((pi, pass_items))

        seg_pass_assignments[seg_id] = passes

    new_flow: list[dict] = [{"kind": "input"}]
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


def register_routes(app: FastAPI) -> None:
    @app.post("/api/hw_config_verify")
    async def api_hw_config_verify(body: dict):
        def _run():
            from mimarsinan.mapping.verification.verifier import (
                verify_hardware_config,
            )
            from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
            mr = dict(body.get("model_repr_json", {}))
            core_types = body.get("core_types", [])
            from mimarsinan.mapping.platform.platform_constraints import (
                resolve_platform_mapping_params,
            )

            if core_types:
                pmap = resolve_platform_mapping_params(
                    core_types,
                    allow_coalescing=bool(body.get("allow_coalescing", False)),
                )
                tile_max_ax = pmap.effective_max_axons
                tile_max_neu = pmap.effective_max_neurons
                mr["allow_coalescing"] = pmap.allow_coalescing
                mr["hardware_bias"] = pmap.hardware_bias
            else:
                tile_max_ax = int(mr.get("max_axons", 1024))
                tile_max_neu = int(mr.get("max_neurons", 1024))
            mr["max_axons"] = max(int(mr.get("max_axons", 1024)), 4096)
            mr["max_neurons"] = max(int(mr.get("max_neurons", 1024)), 4096)
            layout_result = get_layout_result_from_request(
                mr,
                tiling_max_axons=tile_max_ax,
                tiling_max_neurons=tile_max_neu,
            )
            softcores = layout_result.softcores
            result = verify_hardware_config(
                softcores, core_types,
                **ChipCapabilities.from_platform_constraints(body).permission_kwargs(),
            )
            stats_out = {
                **(result.get("stats") or {}),
                "host_side_segment_count": layout_result.host_side_segment_count,
                "layout_preview": layout_result.layout_preview,
            }

            si = result.get("schedule_info")
            if si and si.get("per_segment_passes"):
                stats_out["layout_preview"] = expand_preview_for_scheduling(
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
        def _run():
            from mimarsinan.mapping.verification.suggester.hw_config_suggester import suggest_hardware_config
            from mimarsinan.mapping.verification.suggester.hw_config_suggester_scheduled import (
                suggest_hardware_config_scheduled,
            )
            layout_body = dict(body)
            layout_body["max_axons"] = max(int(body.get("max_axons", 1024)), 4096)
            layout_body["max_neurons"] = max(int(body.get("max_neurons", 1024)), 4096)
            softcores = get_softcores_from_request(layout_body)

            allow_coalescing = bool(body.get("allow_coalescing", False))
            hardware_bias = bool(body.get("hardware_bias", True))
            axon_granularity = int(body.get("axon_granularity", 1))
            neuron_granularity = int(body.get("neuron_granularity", 1))
            safety_margin = float(body.get("safety_margin", 0.15))
            allow_neuron_splitting = bool(body.get("allow_neuron_splitting", False))

            if bool(body.get("allow_scheduling", False)):
                suggestion = suggest_hardware_config_scheduled(
                    softcores,
                    max_passes=int(body.get("max_schedule_passes", 8)),
                    latency_weight=float(body.get("scheduling_latency_weight", 1.0)),
                    allow_coalescing=allow_coalescing,
                    hardware_bias=hardware_bias,
                    axon_granularity=axon_granularity,
                    neuron_granularity=neuron_granularity,
                    safety_margin=safety_margin,
                    allow_neuron_splitting=allow_neuron_splitting,
                )
            else:
                suggestion = suggest_hardware_config(
                    softcores,
                    allow_coalescing=allow_coalescing,
                    hardware_bias=hardware_bias,
                    axon_granularity=axon_granularity,
                    neuron_granularity=neuron_granularity,
                    safety_margin=safety_margin,
                    allow_neuron_splitting=allow_neuron_splitting,
                )

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
