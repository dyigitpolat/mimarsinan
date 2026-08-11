"""Layout payload collection helpers for MimarsinanLayoutBackend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List

from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def collect_layout_payload(
    problem: Any,
    configuration: Dict[str, Any],
) -> Dict[str, Any]:
    """Render the problem's own layout of this candidate as an agent-readable payload.

    Everything here is a PROJECTION of ``problem.candidate_layout``: the chip is
    the one the deployment resolver builds for the candidate, the softcores are
    the ones the search packs, and the objective values are the registry read of
    the very view the search scores. Nothing is recomputed on the side, so the
    agent cannot be shown a chip or a number the search does not use.
    """
    layout = problem.candidate_layout(configuration)
    return {
        "softcore_count": len(layout.softcores),
        "per_softcore": [
            softcore_to_dict(sc, idx) for idx, sc in enumerate(layout.softcores)
        ],
        "per_layer": aggregate_per_layer(layout.softcores),
        "layout_stats": layout.stats.to_dict(),
        # Every axis this candidate's STATIC facts can answer — the training
        # proxy is not among them, by construction of the candidate view.
        "hw_objectives": OBJECTIVES.extract(layout.view),
    }


def softcore_to_dict(sc: LayoutSoftCoreSpec, index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "name": sc.name,
        "input_count": int(sc.input_count),
        "output_count": int(sc.output_count),
        "area": int(sc.area),
        "residency_class_id": int(sc.residency_class_id),
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
                "residency_classes": set(),
                "latency_tags": set(),
                "segments": set(),
            },
        )
        row["softcore_count"] += 1
        row["total_area"] += int(sc.area)
        row["max_input_count"] = max(row["max_input_count"], int(sc.input_count))
        row["max_output_count"] = max(row["max_output_count"], int(sc.output_count))
        row["residency_classes"].add(int(sc.residency_class_id))
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
                "residency_class_count": len(row["residency_classes"]),
                "latency_tag_count": len(row["latency_tags"]),
                "segment_count": len(row["segments"]),
            }
        )
    rows.sort(key=lambda r: r["layer"])
    return rows


def layer_key(sc: LayoutSoftCoreSpec) -> str:
    name = sc.name or f"unnamed_tg{int(sc.residency_class_id)}"
    for sep in ("_tile_", "_psum_pos_", "_psum_neg_", "_psum_accum_", "_pos", "_col"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name
