"""Layout payload collection helpers for MimarsinanLayoutBackend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.search.problems.joint.problem import json_key


def collect_layout_payload(
    problem: Any,
    configuration: Dict[str, Any],
) -> Dict[str, Any]:
    """Pull per-softcore, per-layer, and layout_stats data via the problem."""
    pcfg = configuration.get("platform_constraints", {})
    cores_cfg = pcfg.get("cores", [])

    core_types = [
        LayoutHardCoreType(
            max_axons=int(c["max_axons"]),
            max_neurons=int(c["max_neurons"]),
            count=int(c["count"]),
        )
        for c in cores_cfg
    ]

    cache = getattr(problem, "_hw_only_cache", None)
    softcores: List[LayoutSoftCoreSpec]
    host_segments: int
    total_params: float
    if cache is not None and getattr(problem, "search_mode", "joint") == "hardware":
        softcores = list(cache.softcores)
        host_segments = int(cache.host_side_segment_count)
        total_params = float(cache.total_params)
    else:
        mc = configuration.get("model_config", {})
        try:
            model, total_params = problem._build_model(mc, pcfg)
        except Exception:
            key = json_key(configuration)
            vc = getattr(problem, "_validation_cache", {}).get(key)
            if vc is None:
                raise
            hw_obj = vc.hw_objectives
            return {
                "softcore_count": 0,
                "per_softcore": [],
                "per_layer": [],
                "layout_stats": {},
                "hw_objectives": dict(hw_obj),
            }
        softcores, host_segments = problem._collect_softcores(model, pcfg)

    stats, _err = compute_mapping_stats(
        softcores=softcores,
        core_types=core_types,
        allow_scheduling=bool(pcfg.get("allow_scheduling", False)),
        allow_neuron_splitting=bool(pcfg.get("allow_neuron_splitting", False)),
        allow_coalescing=bool(pcfg.get("allow_coalescing", False)),
    )
    per_softcore = [softcore_to_dict(sc, idx) for idx, sc in enumerate(softcores)]
    per_layer = aggregate_per_layer(softcores)
    hw_objectives, _ = problem._compute_hw_objectives(
        softcores, pcfg, total_params, host_segments,
    )

    return {
        "softcore_count": len(softcores),
        "per_softcore": per_softcore,
        "per_layer": per_layer,
        "layout_stats": stats.to_dict() if stats else {},
        "hw_objectives": dict(hw_objectives or {}),
    }


def softcore_to_dict(sc: LayoutSoftCoreSpec, index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "name": sc.name,
        "input_count": int(sc.input_count),
        "output_count": int(sc.output_count),
        "area": int(sc.area),
        "threshold_group_id": int(sc.threshold_group_id),
        "latency_tag": (None if sc.latency_tag is None else int(sc.latency_tag)),
        "segment_id": (None if sc.segment_id is None else int(sc.segment_id)),
    }


def aggregate_per_layer(
    softcores: Sequence[LayoutSoftCoreSpec],
) -> List[Dict[str, Any]]:
    """Roll per-softcore facts up to per-layer rows for the agent."""
    by_layer: Dict[str, Dict[str, Any]] = {}
    for sc in softcores:
        key = layer_key(sc)
        row = by_layer.setdefault(
            key,
            {
                "layer": key,
                "softcore_count": 0,
                "total_area": 0,
                "max_input_count": 0,
                "max_output_count": 0,
                "threshold_groups": set(),
                "latency_tags": set(),
                "segments": set(),
            },
        )
        row["softcore_count"] += 1
        row["total_area"] += int(sc.area)
        row["max_input_count"] = max(row["max_input_count"], int(sc.input_count))
        row["max_output_count"] = max(row["max_output_count"], int(sc.output_count))
        row["threshold_groups"].add(int(sc.threshold_group_id))
        if sc.latency_tag is not None:
            row["latency_tags"].add(int(sc.latency_tag))
        if sc.segment_id is not None:
            row["segments"].add(int(sc.segment_id))

    rows: List[Dict[str, Any]] = []
    for row in by_layer.values():
        rows.append(
            {
                "layer": row["layer"],
                "softcore_count": row["softcore_count"],
                "total_area": row["total_area"],
                "max_input_count": row["max_input_count"],
                "max_output_count": row["max_output_count"],
                "threshold_group_count": len(row["threshold_groups"]),
                "latency_tag_count": len(row["latency_tags"]),
                "segment_count": len(row["segments"]),
            }
        )
    rows.sort(key=lambda r: r["layer"])
    return rows


def layer_key(sc: LayoutSoftCoreSpec) -> str:
    name = sc.name or f"unnamed_tg{int(sc.threshold_group_id)}"
    for sep in ("_tile_", "_psum_pos_", "_psum_neg_", "_psum_accum_", "_pos", "_col"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name
