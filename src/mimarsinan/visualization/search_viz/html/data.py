"""Parse search result JSON into report-ready structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from mimarsinan.common.safe_numeric import safe_float
from mimarsinan.visualization.search_viz.series import (
    PENALTY_CUTOFF,
    best_metric_series,
    goal_by_metric,
)


@dataclass
class ReportData:
    metric_names: List[str]
    goal_by_name: Dict[str, Any]
    gens: List[Any]
    bests: List[Dict[str, Any]]
    candidate_data: Dict[str, Any]
    table_rows: List[Dict[str, Any]]
    pareto: List[Any]
    history_series: Dict[str, List[Any]]


def extract_candidate_data(
    candidates: List[Any],
    metric_names: List[str],
    *,
    check_pareto: bool = False,
) -> Dict[str, Any]:
    """Extract objective data from candidates list."""
    data: Dict[str, Any] = {name: [] for name in metric_names}
    data["is_pareto"] = []
    data["generation"] = []
    data["hover_info"] = []

    for candidate in candidates:
        obj = (candidate.get("objectives", {}) if isinstance(candidate, dict) else {}) or {}
        meta = (candidate.get("metadata", {}) if isinstance(candidate, dict) else {}) or {}

        is_pareto = meta.get("is_pareto", check_pareto)
        gen = meta.get("generation", -1)

        info_parts = [f"Gen: {gen}", f"Pareto: {'✓' if is_pareto else '✗'}"]
        info_parts.extend([f"{k}: {safe_float(v):.4f}" for k, v in obj.items()])
        hover_info = "<br>".join(info_parts)

        valid = True
        for name in metric_names:
            value = safe_float(obj.get(name))
            if value is None or value >= PENALTY_CUTOFF:
                valid = False
                break
            data[name].append(value)

        if valid:
            data["is_pareto"].append(is_pareto)
            data["generation"].append(gen)
            data["hover_info"].append(hover_info)
        else:
            for name in metric_names:
                if data[name] and len(data[name]) > len(data["is_pareto"]):
                    data[name].pop()

    return data


def build_table_rows(
    all_candidates: List[Any],
    pareto: List[Any],
    metric_names: List[str],
) -> List[Dict[str, Any]]:
    table_rows: List[Dict[str, Any]] = []
    source_candidates = all_candidates if all_candidates else pareto
    for idx, candidate in enumerate(source_candidates):
        config = candidate.get("configuration", {}) if isinstance(candidate, dict) else {}
        model_cfg = config.get("model_config", {})
        platform_cfg = config.get("platform_constraints", {})
        objectives_data = candidate.get("objectives", {}) if isinstance(candidate, dict) else {}
        meta = candidate.get("metadata", {}) if isinstance(candidate, dict) else {}

        is_pareto = meta.get("is_pareto", True if not all_candidates else False)
        gen = meta.get("generation", -1)

        has_penalty = any((safe_float(v, 0.0) or 0.0) >= PENALTY_CUTOFF for v in objectives_data.values())
        if has_penalty:
            continue

        row: Dict[str, Any] = {
            "id": idx,
            "generation": gen,
            "is_pareto": is_pareto,
        }
        for key, value in model_cfg.items():
            row[f"model_{key}"] = value
        cores = platform_cfg.get("cores", [])
        if cores:
            row["hw_core_types"] = len(cores)
            row["hw_max_axons"] = max((int(ct.get("max_axons", 0)) for ct in cores), default="")
            row["hw_max_neurons"] = max((int(ct.get("max_neurons", 0)) for ct in cores), default="")
        for key, value in objectives_data.items():
            val = safe_float(value)
            if val is not None and val < PENALTY_CUTOFF:
                row[f"obj_{key}"] = val

        table_rows.append(row)
    return table_rows


def hist_series(bests: List[Dict[str, Any]], name: str) -> List[Any]:
    return list(best_metric_series(bests, name))


def parse_search_result(result_json: Dict[str, Any]) -> ReportData:
    goal_by_name = goal_by_metric(result_json)
    metric_names = list(goal_by_name.keys())

    pareto = result_json.get("pareto_front", []) or []
    all_candidates = result_json.get("all_candidates", []) or []
    hist = result_json.get("history", []) or []

    gens = [h.get("gen") for h in hist if isinstance(h, dict) and "gen" in h]
    bests = [h.get("best", {}) if isinstance(h, dict) else {} for h in hist]

    if all_candidates:
        candidate_data = extract_candidate_data(all_candidates, metric_names)
    else:
        candidate_data = extract_candidate_data(pareto, metric_names, check_pareto=True)

    table_rows = build_table_rows(all_candidates, pareto, metric_names)
    history_series = {name: hist_series(bests, name) for name in metric_names}

    return ReportData(
        metric_names=metric_names,
        goal_by_name=goal_by_name,
        gens=gens,
        bests=bests,
        candidate_data=candidate_data,
        table_rows=table_rows,
        pareto=pareto,
        history_series=history_series,
    )
