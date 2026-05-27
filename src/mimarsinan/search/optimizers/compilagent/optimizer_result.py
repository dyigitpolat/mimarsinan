"""SearchResult construction for CompilagentOptimizer."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from mimarsinan.search.optimizers.agent_evolve.codec import (
    compute_pareto_front,
    result_to_candidate,
    select_best_candidate_minimax,
    sort_pareto_results_minimax_first,
)
from mimarsinan.search.optimizers.agent_evolve.schema import CandidateResult
from mimarsinan.search.results import Candidate, ObjectiveSpec, SearchResult

from .backend import MimarsinanLayoutBackend
from .sink import CandidateRecord, MultiObjectiveSink


def build_search_result(
    *,
    sink: MultiObjectiveSink,
    objectives: List[ObjectiveSpec],
    elapsed_ms: float,
    backend: MimarsinanLayoutBackend,
    invalid_penalty: float,
) -> SearchResult:
    valid_results: List[CandidateResult] = []
    all_candidates: List[Candidate] = []
    active_names = {o.name for o in objectives}

    penalty_objectives = {
        spec.name: (0.0 if spec.goal == "max" else float(invalid_penalty))
        for spec in objectives
    }

    for rec in sink.records():
        cfg = configuration_for(backend, rec)
        layout_md = layout_metadata_for(backend, rec.candidate_id)
        if rec.rejected or not rec.objectives:
            cr = CandidateResult(
                configuration=dict(cfg),
                objectives=dict(penalty_objectives),
                is_valid=False,
                error_message=rec.reject_reason or "compile or evaluation failed",
                failure_phase="compile",
            )
            metadata = {"is_pareto": False, "valid": False}
            if layout_md:
                metadata["layout"] = layout_md
            all_candidates.append(result_to_candidate(cr, metadata))
            continue

        objs: Dict[str, float] = {
            name: float(rec.objectives.get(name, 0.0)) for name in active_names
        }
        cr = CandidateResult(
            configuration=dict(cfg), objectives=objs, is_valid=True,
        )
        valid_results.append(cr)
        metadata = {
            "is_pareto": False,
            "candidate_id": rec.candidate_id,
        }
        if layout_md:
            metadata["layout"] = layout_md
        all_candidates.append(result_to_candidate(cr, metadata))

    pareto = compute_pareto_front(valid_results, objectives)
    pareto_sorted = sort_pareto_results_minimax_first(pareto, objectives)
    best_result = select_best_candidate_minimax(pareto, objectives)
    if best_result is not None:
        best = result_to_candidate(best_result, {"is_pareto": True})
    else:
        best = Candidate(configuration={}, objectives={}, metadata={})

    pareto_candidates = [
        result_to_candidate(r, {"is_pareto": True}) for r in pareto_sorted
    ]

    pareto_keys = {
        json.dumps(c.configuration, sort_keys=True, default=str)
        for c in pareto_candidates
    }
    for c in all_candidates:
        key = json.dumps(c.configuration, sort_keys=True, default=str)
        if key in pareto_keys:
            c.metadata["is_pareto"] = True

    history = [{
        "gen": 1,
        "valid_count": len(valid_results),
        "failed_count": len(sink.failed_records()),
        "pareto_size": len(pareto),
        "elapsed_ms": elapsed_ms,
    }]

    sink.emit_pareto_update(
        pareto_front=[
            {
                "candidate_id": (c.metadata or {}).get("candidate_id"),
                "configuration": c.configuration,
                "objectives": dict(c.objectives),
            }
            for c in pareto_candidates
        ]
    )
    sink.emit_session_complete(
        total_valid=len(valid_results),
        total_failed=len(sink.failed_records()),
        pareto_size=len(pareto),
        best_objectives=dict(best.objectives) if best.objectives else None,
        elapsed_ms=float(elapsed_ms),
    )

    return SearchResult(
        objectives=objectives,
        best=best,
        pareto_front=pareto_candidates,
        all_candidates=all_candidates,
        history=history,
    )


def configuration_for(
    backend: MimarsinanLayoutBackend, rec: CandidateRecord,
) -> Dict[str, Any]:
    try:
        payload = backend.get_candidate_payload(rec.candidate_id)
    except KeyError:
        payload = None
    if payload and isinstance(payload.get("config"), dict):
        return dict(payload["config"])
    if rec.configuration:
        return dict(rec.configuration)
    return {"description": rec.description}


def layout_metadata_for(
    backend: MimarsinanLayoutBackend, candidate_id: str,
) -> Optional[Dict[str, Any]]:
    try:
        payload = backend.get_candidate_payload(candidate_id)
    except KeyError:
        return None
    layout_stats = payload.get("layout_stats", {}) or {}
    return {
        "softcore_count": len(payload.get("softcores", [])),
        "per_layer": payload.get("per_layer", []),
        "summary": {
            "total_cores": layout_stats.get("total_cores"),
            "total_softcores": layout_stats.get("total_softcores"),
            "neural_segment_count": layout_stats.get("neural_segment_count"),
            "threshold_group_count": layout_stats.get("threshold_group_count"),
            "fragmentation_pct": layout_stats.get("fragmentation_pct"),
            "mapped_params_pct": layout_stats.get("mapped_params_pct"),
            "schedule_pass_count": layout_stats.get("schedule_pass_count"),
            "schedule_sync_count": layout_stats.get("schedule_sync_count"),
        },
    }


def build_workload_instance(spec):
    from compilagent import WorkloadInstance

    return WorkloadInstance(
        spec=spec, forward=lambda: None, example_inputs=(),
        metadata={"workload_id": spec.id},
    )
