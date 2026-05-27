from __future__ import annotations
from typing import Dict, List, Optional, Sequence
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec

def _pct(part: float, total: float) -> float:
    return (part / total * 100.0) if total > 0 else 0.0


def _safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return float(s[n // 2]) if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _latency_stats(
    softcores: Optional[Sequence[LayoutSoftCoreSpec]],
) -> tuple[int, float, float, float, int]:
    """Return segment count, latency tiers/segment min-median-max, threshold count."""
    if not softcores:
        return 0, 0.0, 0.0, 0.0, 0

    tagged_softcores = [sc for sc in softcores if sc.latency_tag is not None]
    threshold_groups = len({sc.threshold_group_id for sc in softcores})
    if not tagged_softcores:
        return 0, 0.0, 0.0, 0.0, threshold_groups

    segments_to_latencies: Dict[int, set[int]] = {}
    fallback_by_latency_tag = {int(sc.latency_tag) for sc in tagged_softcores}
    has_segment_ids = any(sc.segment_id is not None for sc in tagged_softcores)

    if has_segment_ids:
        for sc in tagged_softcores:
            seg_id = int(sc.segment_id) if sc.segment_id is not None else int(sc.latency_tag)
            segments_to_latencies.setdefault(seg_id, set()).add(int(sc.latency_tag))
    else:
        # Backward-compatible fallback for handcrafted tests / old softcore data:
        # each distinct latency tag is treated as its own single-tier segment.
        for lat in fallback_by_latency_tag:
            segments_to_latencies[lat] = {lat}

    per_segment_latencies = sorted(float(len(latencies)) for latencies in segments_to_latencies.values())
    return (
        len(per_segment_latencies),
        float(min(per_segment_latencies)),
        _safe_median(per_segment_latencies),
        float(max(per_segment_latencies)),
        threshold_groups,
    )
