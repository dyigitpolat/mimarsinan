"""``MultiObjectiveSink`` — captures per-candidate objectives + configs.

Compilagent's ``OptimizationSession`` emits one ``ObservationEvent`` per
beat (``CANDIDATE_PROPOSED``, ``COMPILE_COMPLETED``,
``BENCHMARK_COMPLETED``, ``OBJECTIVES_RECORDED``, agent thinking deltas,
tool calls, …). The default sinks (``TraceStore``, ``NullSink``) persist
or drop them. ``CompilagentOptimizer`` needs a structured per-candidate
stream so it can rebuild a multi-objective Pareto front; this sink is
the in-memory tap that records exactly the events the optimizer cares
about while still forwarding everything to a wrapped sink (so the trace
store and any WebSocket fan-out still get full fidelity).

In addition to the per-candidate record book-keeping, the sink optionally
forwards a curated set of events to the live reporter under the
``compilagent_*`` ``search_event`` envelope (the GUI's
``compilagent-live.js`` consumes them). The events emitted are
intentionally distinct from the AgentEvolve event vocabulary
(``generation_start``, ``candidate_result``, ``llm_trace``, ...), so the
two live monitors stay decoupled.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from compilagent import EventKind, ObservationEvent, ObservationSink


@dataclass
class CandidateRecord:
    """One candidate as observed by the sink.

    ``configuration`` is populated when the optimizer pre-registers the
    decoded ``(model_config, platform_constraints)`` mapping for the
    candidate (the agent's plan); otherwise it stays empty and the
    optimizer falls back to recomputing from the plan it stashed.
    """

    candidate_id: str
    objectives: Dict[str, float] = field(default_factory=dict)
    objective_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    rejected: bool = False
    reject_reason: Optional[str] = None
    description: str = ""
    expected_effect: str = ""
    interventions: list = field(default_factory=list)
    compile_ok: Optional[bool] = None
    compile_diagnostics: Optional[str] = None
    compile_warnings: list = field(default_factory=list)
    compile_failure_phase: Optional[str] = None
    timing_median_ms: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None


class MultiObjectiveSink:
    """Wrap a base sink and accumulate per-candidate state.

    Forwards every event to the wrapped sink while also tapping a small
    set of `EventKind` values to populate `CandidateRecord`s and emit a
    curated ``search_event`` stream for the compilagent live monitor.
    """

    def __init__(
        self,
        base: ObservationSink,
        *,
        live_reporter: Optional[Any] = None,
    ) -> None:
        self._base = base
        self._records: Dict[str, CandidateRecord] = {}
        self._configurations: Dict[str, Dict[str, Any]] = {}
        # Optional reporter that turns selected events into the
        # ``search_event`` JSON envelope ``compilagent-live.js`` consumes.
        self._live_reporter = live_reporter
        # Track the order candidates first appear so the live monitor
        # can render them with stable indices.
        self._candidate_order: Dict[str, int] = {}
        # Session-level fields populated from SESSION_STARTED + the
        # baseline COMPILE_COMPLETED so the live monitor can render the
        # header without scraping subsequent events.
        self._session_meta: Dict[str, Any] = {}

    def attach_live_reporter(self, reporter: Optional[Any]) -> None:
        self._live_reporter = reporter

    # External handles for the optimizer

    def attach_configuration(
        self, candidate_id: str, configuration: Mapping[str, Any],
    ) -> None:
        """Pre-bind a decoded configuration to a candidate id."""

        self._configurations[candidate_id] = dict(configuration)
        record = self._records.get(candidate_id)
        if record is not None:
            record.configuration = dict(configuration)

    def emit_session_start(
        self,
        *,
        workload_id: str,
        backend_id: str,
        model: str,
        harness: str,
        max_candidates: int,
        max_continuations: int,
        objectives: Sequence[Dict[str, str]],
    ) -> None:
        """Emit the ``compilagent_session_start`` event used by the live monitor.

        The objective catalogue is sent as a list of ``{name, goal}`` so
        the live monitor can render the COMPILAGENT title bar with the
        full multi-objective context. There is intentionally no
        ``primary_objective`` field — every axis is equally weighted in
        the multi-objective leaderboard added in this extension.
        """

        meta = {
            "type": "compilagent_session_start",
            "workload_id": workload_id,
            "backend_id": backend_id,
            "model": model,
            "harness": harness,
            "max_candidates": int(max_candidates),
            "max_continuations": int(max_continuations),
            "objectives": list(objectives),
        }
        self._session_meta.update(meta)
        self._emit_search_event(meta)

    def emit_session_complete(
        self,
        *,
        total_valid: int,
        total_failed: int,
        pareto_size: int,
        best_objectives: Optional[Dict[str, Any]],
        elapsed_ms: float,
    ) -> None:
        self._emit_search_event({
            "type": "compilagent_session_complete",
            "total_valid": int(total_valid),
            "total_failed": int(total_failed),
            "pareto_size": int(pareto_size),
            "best_objectives": best_objectives or {},
            "elapsed_ms": float(elapsed_ms),
        })

    def emit_pareto_update(
        self, *, pareto_front: Sequence[Dict[str, Any]],
    ) -> None:
        self._emit_search_event({
            "type": "compilagent_pareto_update",
            "pareto_front": list(pareto_front),
        })

    def emit_guidance(self, *, text: str, target_tool: str) -> None:
        """Forward a ``[GUIDANCE]`` or ``[BASELINE FOOTPRINT]`` block.

        Fired by :class:`GuidedToolset` whenever it augments a tool's
        result. The live monitor renders these as a dedicated activity
        row so the operator sees the same guidance the agent receives.
        """

        self._emit_search_event({
            "type": "compilagent_guidance",
            "target_tool": str(target_tool),
            "text": str(text),
        })

    def records(self) -> Sequence[CandidateRecord]:
        """All candidate records observed so far, in insertion order."""

        return tuple(self._records.values())

    def successful_records(self) -> Sequence[CandidateRecord]:
        return tuple(r for r in self._records.values() if not r.rejected and r.objectives)

    def failed_records(self) -> Sequence[CandidateRecord]:
        return tuple(r for r in self._records.values() if r.rejected)

    # ObservationSink protocol — forward + tap

    def emit(self, event: ObservationEvent) -> None:
        self._base.emit(event)
        self._observe(event)

    def emit_kv(
        self,
        kind: Any,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        artifact_paths: Optional[Sequence[Any]] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> None:
        self._base.emit_kv(
            kind,
            payload=payload,
            artifact_paths=artifact_paths,
            session_id=session_id,
            run_id=run_id,
            candidate_id=candidate_id,
        )
        # Build a synthetic event for the tap so ``_observe`` stays single-path.
        paths_tuple = tuple(str(p) for p in (artifact_paths or ()))
        self._observe(
            ObservationEvent.make(
                kind,
                session_id=session_id,
                run_id=run_id,
                candidate_id=candidate_id,
                payload=payload,
                artifact_paths=paths_tuple,
            )
        )

    # Internal

    def _record_for(self, candidate_id: str) -> CandidateRecord:
        rec = self._records.get(candidate_id)
        if rec is None:
            rec = CandidateRecord(candidate_id=candidate_id)
            cfg = self._configurations.get(candidate_id)
            if cfg is not None:
                rec.configuration = dict(cfg)
            self._records[candidate_id] = rec
            self._candidate_order[candidate_id] = len(self._candidate_order)
        return rec

    def _observe(self, event: ObservationEvent) -> None:
        kind = event.kind
        cid_from_payload = (event.payload or {}).get("candidate_id") if event.payload else None
        cid = event.candidate_id or cid_from_payload

        # ── Generic (non-candidate) events ──────────────────────────────
        if kind == EventKind.SESSION_STARTED.value:
            self._emit_search_event({
                "type": "compilagent_session_observed",
                "run_id": (event.payload or {}).get("run_id"),
                "device": (event.payload or {}).get("device"),
                "max_candidates": (event.payload or {}).get("max_candidates"),
            })
            return
        if kind == EventKind.SEARCH_SPACE_DERIVED.value:
            self._emit_search_event({
                "type": "compilagent_search_space_derived",
                "lever_count": (event.payload or {}).get("lever_count"),
            })
            return
        if kind == EventKind.COMPILER_PASS.value:
            self._emit_search_event({
                "type": "compilagent_compile_phase",
                "candidate_id": cid,
                "stage": (event.payload or {}).get("stage"),
                "name": (event.payload or {}).get("name"),
                "duration_ms": (event.payload or {}).get("duration_ms"),
            })
            return
        if kind in (EventKind.TOOL_CALL_STARTED.value, EventKind.TOOL_CALL_COMPLETED.value):
            payload = event.payload or {}
            self._emit_search_event({
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
            self._emit_search_event({
                "type": "compilagent_agent_thinking",
                "part_index": (event.payload or {}).get("part_index"),
                "text": (event.payload or {}).get("text", ""),
            })
            return
        if kind == EventKind.AGENT_TEXT_DELTA.value:
            self._emit_search_event({
                "type": "compilagent_agent_text",
                "part_index": (event.payload or {}).get("part_index"),
                "text": (event.payload or {}).get("text", ""),
            })
            return
        if kind == EventKind.RUN_PROGRESS.value:
            payload = event.payload or {}
            self._emit_search_event({
                "type": "compilagent_run_progress",
                "successful_count": payload.get("successful_count"),
                "failed_attempts": payload.get("failed_attempts"),
                "max_candidates": payload.get("max_candidates"),
                "slots_remaining": payload.get("slots_remaining"),
            })
            return
        if kind == EventKind.RUN_CONTINUATION.value:
            payload = event.payload or {}
            self._emit_search_event({
                "type": "compilagent_continuation",
                "iteration": payload.get("iteration"),
                "successful_count": payload.get("successful_count"),
                "slots_remaining": payload.get("slots_remaining"),
                "reason_to_continue": payload.get("reason_to_continue"),
            })
            return
        if kind == EventKind.LEADERBOARD_UPDATED.value:
            self._emit_search_event({
                "type": "compilagent_leaderboard",
                "rows": (event.payload or {}).get("rows", []),
            })
            return

        # ── Per-candidate events ───────────────────────────────────────
        if not cid:
            return
        cid = str(cid)
        # The session emits COMPILE_STARTED / COMPILE_COMPLETED with
        # candidate_id="baseline" during bootstrap. Those are not real
        # search candidates — skip them so the record book remains a
        # 1-1 map of agent-proposed candidates.
        if cid == "baseline":
            return

        if kind == EventKind.CANDIDATE_PROPOSED.value:
            payload = event.payload or {}
            rec = self._record_for(cid)
            rec.description = str(payload.get("description") or "")
            rec.expected_effect = str(payload.get("expected_effect") or "")
            rec.interventions = list(payload.get("interventions") or [])
            self._emit_search_event({
                "type": "compilagent_candidate_proposed",
                "candidate_id": cid,
                "idx": self._candidate_order[cid],
                "description": rec.description,
                "expected_effect": rec.expected_effect,
                "interventions": rec.interventions,
            })
        elif kind == EventKind.COMPILE_STARTED.value:
            self._emit_search_event({
                "type": "compilagent_candidate_compiling",
                "candidate_id": cid,
                "idx": self._record_for(cid)
                and self._candidate_order.get(cid, 0),
            })
        elif kind == EventKind.COMPILE_COMPLETED.value:
            payload = event.payload or {}
            rec = self._record_for(cid)
            rec.compile_ok = bool(payload.get("ok"))
            rec.compile_diagnostics = payload.get("diagnostics")
            rec.compile_warnings = list(payload.get("warnings") or [])
            self._emit_search_event({
                "type": "compilagent_candidate_compiled",
                "candidate_id": cid,
                "idx": self._candidate_order[cid],
                "ok": rec.compile_ok,
                "elapsed_ms": payload.get("elapsed_ms"),
                "diagnostics": rec.compile_diagnostics,
                "warnings": rec.compile_warnings,
            })
        elif kind == EventKind.BENCHMARK_COMPLETED.value:
            payload = event.payload or {}
            rec = self._record_for(cid)
            rec.timing_median_ms = payload.get("median_ms")
            rec.speedup_vs_baseline = payload.get("speedup_vs_baseline")
            self._emit_search_event({
                "type": "compilagent_candidate_benchmarked",
                "candidate_id": cid,
                "idx": self._candidate_order[cid],
                "median_ms": payload.get("median_ms"),
                "p20_ms": payload.get("p20_ms"),
                "p80_ms": payload.get("p80_ms"),
                "speedup_vs_baseline": payload.get("speedup_vs_baseline"),
            })
        elif kind == EventKind.OBJECTIVES_RECORDED.value:
            rec = self._record_for(cid)
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
            self._emit_search_event({
                "type": "compilagent_candidate_objectives",
                "candidate_id": cid,
                "idx": self._candidate_order[cid],
                "objectives": objective_values,
                "metadata": objective_meta,
            })
        elif kind == EventKind.CANDIDATE_REJECTED.value:
            payload = event.payload or {}
            rec = self._record_for(cid)
            rec.rejected = True
            rec.reject_reason = str(payload.get("reason") or "")
            # Failure phase comes from the cached compile diagnostics
            # (set when COMPILE_COMPLETED arrived first).
            if rec.compile_ok is False and rec.compile_diagnostics:
                rec.compile_failure_phase = "compile"
            self._emit_search_event({
                "type": "compilagent_candidate_rejected",
                "candidate_id": cid,
                "idx": self._candidate_order[cid],
                "reason": rec.reject_reason,
                "diagnostics": rec.compile_diagnostics,
            })

    def _emit_search_event(self, event: Dict[str, Any]) -> None:
        if self._live_reporter is None:
            return
        try:
            self._live_reporter("search_event", json.dumps(event, default=str))
        except Exception:
            pass


__all__ = ["CandidateRecord", "MultiObjectiveSink"]
