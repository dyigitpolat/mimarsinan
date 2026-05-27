from __future__ import annotations
from typing import Any, Dict, Optional

def stats_dict_from_hybrid_mapping(mapping: Any) -> Optional[Dict[str, Any]]:
    """Wizard-shaped mapping performance stats from a compiled hybrid mapping."""
    if mapping is None:
        return None
    stages = getattr(mapping, "stages", None)
    if not stages:
        return None

    cores_seen: list[Any] = []
    schedule_pass_count = 0
    max_pass_by_segment: dict[int, int] = {}
    schedule_segments_present = False
    for stage in stages:
        if getattr(stage, "kind", None) != "neural":
            continue
        hcm = getattr(stage, "hard_core_mapping", None)
        if hcm is None:
            continue
        sched_idx = getattr(stage, "schedule_pass_index", None)
        seg_idx = getattr(stage, "schedule_segment_index", None) or 0
        if sched_idx is not None:
            schedule_segments_present = True
            max_pass_by_segment[seg_idx] = max(
                max_pass_by_segment.get(seg_idx, 0), sched_idx + 1
            )
        for core in getattr(hcm, "cores", []) or []:
            cores_seen.append(core)
    if not cores_seen:
        return None

    if schedule_segments_present:
        schedule_pass_count = sum(max_pass_by_segment.values())

    per_core_ax_pct: list[float] = []
    per_core_neu_pct: list[float] = []
    per_core_param_pct: list[float] = []
    total_used_area = 0
    total_available_area = 0
    total_used_axons = 0
    total_available_axons = 0
    total_used_neurons = 0
    total_available_neurons = 0

    for core in cores_seen:
        ax_total = int(getattr(core, "axons_per_core", 0))
        neu_total = int(getattr(core, "neurons_per_core", 0))
        ax_avail = int(getattr(core, "available_axons", 0))
        neu_avail = int(getattr(core, "available_neurons", 0))
        ax_used = max(0, ax_total - ax_avail)
        neu_used = max(0, neu_total - neu_avail)
        area_total = ax_total * neu_total
        area_used = ax_used * neu_used

        total_used_axons += ax_used
        total_available_axons += ax_total
        total_used_neurons += neu_used
        total_available_neurons += neu_total
        total_used_area += area_used
        total_available_area += area_total

        per_core_ax_pct.append(
            ((ax_total - ax_used) / ax_total * 100.0) if ax_total > 0 else 0.0
        )
        per_core_neu_pct.append(
            ((neu_total - neu_used) / neu_total * 100.0) if neu_total > 0 else 0.0
        )
        per_core_param_pct.append(
            (area_used / area_total * 100.0) if area_total > 0 else 0.0
        )

    placements_total = 0
    for stage in stages:
        if getattr(stage, "kind", None) != "neural":
            continue
        hcm = getattr(stage, "hard_core_mapping", None)
        placements = getattr(hcm, "soft_core_placements_per_hard_core", None) if hcm else None
        if placements:
            placements_total += sum(len(pl) for pl in placements)
    total_softcores = placements_total or len(cores_seen)

    def _pct(part: float, total: float) -> float:
        return (part / total * 100.0) if total > 0 else 0.0

    def _mmm(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        return float(min(values)), float(sum(values) / len(values)), float(max(values))

    ax_min, ax_avg, ax_max = _mmm(per_core_ax_pct)
    neu_min, neu_avg, neu_max = _mmm(per_core_neu_pct)
    param_min, param_avg, param_max = _mmm(per_core_param_pct)

    return {
        "feasible": True,
        "total_cores": len(cores_seen),
        "total_softcores": int(total_softcores),
        "total_hw_cores": len(cores_seen),
        "total_wasted_axons_pct": _pct(
            total_available_axons - total_used_axons, total_available_axons
        ),
        "total_wasted_neurons_pct": _pct(
            total_available_neurons - total_used_neurons, total_available_neurons
        ),
        "mapped_params_pct": _pct(total_used_area, total_available_area),
        "per_core_wasted_axons_pct_min": ax_min,
        "per_core_wasted_axons_pct_avg": ax_avg,
        "per_core_wasted_axons_pct_max": ax_max,
        "per_core_wasted_neurons_pct_min": neu_min,
        "per_core_wasted_neurons_pct_avg": neu_avg,
        "per_core_wasted_neurons_pct_max": neu_max,
        "per_core_mapped_params_pct_min": param_min,
        "per_core_mapped_params_pct_avg": param_avg,
        "per_core_mapped_params_pct_max": param_max,
        "neural_segment_count": len(getattr(mapping, "get_neural_segments", lambda: [])()),
        "schedule_pass_count": int(schedule_pass_count),
        "schedule_sync_count": max(0, schedule_pass_count - len(max_pass_by_segment))
        if schedule_pass_count
        else 0,
        "coalescing_group_count": 0,
        "split_softcore_count": 0,
    }
