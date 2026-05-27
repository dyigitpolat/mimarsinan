"""Observation-event handling for MultiObjectiveSink."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

from compilagent import EventKind, ObservationEvent

from .sink import CandidateRecord, MultiObjectiveSink


def observe_event(sink: MultiObjectiveSink, event: ObservationEvent) -> None:
    kind = event.kind
    cid_from_payload = (event.payload or {}).get("candidate_id") if event.payload else None
    cid = event.candidate_id or cid_from_payload

    if kind == EventKind.SESSION_STARTED.value:
        sink._emit_search_event({
            "type": "compilagent_session_observed",
            "run_id": (event.payload or {}).get("run_id"),
            "device": (event.payload or {}).get("device"),
            "max_candidates": (event.payload or {}).get("max_candidates"),
        })
        return
    if kind == EventKind.SEARCH_SPACE_DERIVED.value:
        sink._emit_search_event({
            "type": "compilagent_search_space_derived",
            "lever_count": (event.payload or {}).get("lever_count"),
        })
        return
    if kind == EventKind.COMPILER_PASS.value:
        sink._emit_search_event({
            "type": "compilagent_compile_phase",
            "candidate_id": cid,
            "stage": (event.payload or {}).get("stage"),
            "name": (event.payload or {}).get("name"),
            "duration_ms": (event.payload or {}).get("duration_ms"),
        })
        return
    if kind in (EventKind.TOOL_CALL_STARTED.value, EventKind.TOOL_CALL_COMPLETED.value):
        payload = event.payload or {}
        sink._emit_search_event({
            "type": "compilagent_tool_call",
            "phase": (
                "started" if kind == EventKind.TOOL_CALL_STARTED.value else "completed"
            ),
            "tool_name": payload.get("tool_name"),
            "args": payload.get("args"),
            "candidate_id": cid,
        })
        return
    if kind == EventKind.AGENT_THINKING_DELTA.value:
        sink._emit_search_event({
            "type": "compilagent_agent_thinking",
            "part_index": (event.payload or {}).get("part_index"),
            "text": (event.payload or {}).get("text", ""),
        })
        return
    if kind == EventKind.AGENT_TEXT_DELTA.value:
        sink._emit_search_event({
            "type": "compilagent_agent_text",
            "part_index": (event.payload or {}).get("part_index"),
            "text": (event.payload or {}).get("text", ""),
        })
        return
    if kind == EventKind.RUN_PROGRESS.value:
        payload = event.payload or {}
        sink._emit_search_event({
            "type": "compilagent_run_progress",
            "successful_count": payload.get("successful_count"),
            "failed_attempts": payload.get("failed_attempts"),
            "max_candidates": payload.get("max_candidates"),
            "slots_remaining": payload.get("slots_remaining"),
        })
        return
    if kind == EventKind.RUN_CONTINUATION.value:
        payload = event.payload or {}
        sink._emit_search_event({
            "type": "compilagent_continuation",
            "iteration": payload.get("iteration"),
            "successful_count": payload.get("successful_count"),
            "slots_remaining": payload.get("slots_remaining"),
            "reason_to_continue": payload.get("reason_to_continue"),
        })
        return
    if kind == EventKind.LEADERBOARD_UPDATED.value:
        sink._emit_search_event({
            "type": "compilagent_leaderboard",
            "rows": (event.payload or {}).get("rows", []),
        })
        return

    if not cid:
        return
    cid = str(cid)
    if cid == "baseline":
        return

    if kind == EventKind.CANDIDATE_PROPOSED.value:
        payload = event.payload or {}
        rec = sink._record_for(cid)
        rec.description = str(payload.get("description") or "")
        rec.expected_effect = str(payload.get("expected_effect") or "")
        rec.interventions = list(payload.get("interventions") or [])
        sink._emit_search_event({
            "type": "compilagent_candidate_proposed",
            "candidate_id": cid,
            "idx": sink._candidate_order[cid],
            "description": rec.description,
            "expected_effect": rec.expected_effect,
            "interventions": rec.interventions,
        })
    elif kind == EventKind.COMPILE_STARTED.value:
        sink._emit_search_event({
            "type": "compilagent_candidate_compiling",
            "candidate_id": cid,
            "idx": sink._record_for(cid) and sink._candidate_order.get(cid, 0),
        })
    elif kind == EventKind.COMPILE_COMPLETED.value:
        payload = event.payload or {}
        rec = sink._record_for(cid)
        rec.compile_ok = bool(payload.get("ok"))
        rec.compile_diagnostics = payload.get("diagnostics")
        rec.compile_warnings = list(payload.get("warnings") or [])
        sink._emit_search_event({
            "type": "compilagent_candidate_compiled",
            "candidate_id": cid,
            "idx": sink._candidate_order[cid],
            "ok": rec.compile_ok,
            "elapsed_ms": payload.get("elapsed_ms"),
            "diagnostics": rec.compile_diagnostics,
            "warnings": rec.compile_warnings,
        })
    elif kind == EventKind.BENCHMARK_COMPLETED.value:
        payload = event.payload or {}
        rec = sink._record_for(cid)
        rec.timing_median_ms = payload.get("median_ms")
        rec.speedup_vs_baseline = payload.get("speedup_vs_baseline")
        sink._emit_search_event({
            "type": "compilagent_candidate_benchmarked",
            "candidate_id": cid,
            "idx": sink._candidate_order[cid],
            "median_ms": payload.get("median_ms"),
            "p20_ms": payload.get("p20_ms"),
            "p80_ms": payload.get("p80_ms"),
            "speedup_vs_baseline": payload.get("speedup_vs_baseline"),
        })
    elif kind == EventKind.OBJECTIVES_RECORDED.value:
        rec = sink._record_for(cid)
        objectives = (event.payload or {}).get("objectives") or {}
        objective_values: Dict[str, float] = {}
        objective_meta: Dict[str, Dict[str, Any]] = {}
        for name, value in objectives.items():
            if isinstance(value, Mapping):
                objective_values[str(name)] = float(value.get("value", 0.0))
                objective_meta[str(name)] = dict(value)
            else:
                objective_values[str(name)] = float(value)
                objective_meta[str(name)] = {
                    "name": str(name),
                    "value": float(value),
                    "goal": "min",
                    "unit": "",
                }
        rec.objectives = objective_values
        rec.objective_metadata = objective_meta
        sink._emit_search_event({
            "type": "compilagent_candidate_objectives",
            "candidate_id": cid,
            "idx": sink._candidate_order[cid],
            "objectives": objective_values,
            "metadata": objective_meta,
        })
    elif kind == EventKind.CANDIDATE_REJECTED.value:
        payload = event.payload or {}
        rec = sink._record_for(cid)
        rec.rejected = True
        rec.reject_reason = str(payload.get("reason") or "")
        if rec.compile_ok is False and rec.compile_diagnostics:
            rec.compile_failure_phase = "compile"
        sink._emit_search_event({
            "type": "compilagent_candidate_rejected",
            "candidate_id": cid,
            "idx": sink._candidate_order[cid],
            "reason": rec.reject_reason,
            "diagnostics": rec.compile_diagnostics,
        })
