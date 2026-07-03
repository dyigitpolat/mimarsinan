"""MultiObjectiveSink — captures per-candidate objectives + configs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from compilagent import ObservationEvent, ObservationSink
from compilagent import ObservationEvent as OE

from mimarsinan.search.optimizers.llm.trace import emit_search_event


@dataclass
class CandidateRecord:
    """One candidate as observed by the sink."""

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
    """Wrap a base sink and accumulate per-candidate state."""

    def __init__(
        self,
        base: ObservationSink,
        *,
        live_reporter: Optional[Any] = None,
    ) -> None:
        self._base = base
        self._records: Dict[str, CandidateRecord] = {}
        self._configurations: Dict[str, Dict[str, Any]] = {}
        self._live_reporter = live_reporter
        self._candidate_order: Dict[str, int] = {}
        self._session_meta: Dict[str, Any] = {}

    def attach_live_reporter(self, reporter: Optional[Any]) -> None:
        self._live_reporter = reporter

    def attach_configuration(
        self, candidate_id: str, configuration: Mapping[str, Any],
    ) -> None:
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
        self._emit_search_event({
            "type": "compilagent_guidance",
            "target_tool": str(target_tool),
            "text": str(text),
        })

    def records(self) -> Sequence[CandidateRecord]:
        return tuple(self._records.values())

    def successful_records(self) -> Sequence[CandidateRecord]:
        return tuple(r for r in self._records.values() if not r.rejected and r.objectives)

    def failed_records(self) -> Sequence[CandidateRecord]:
        return tuple(r for r in self._records.values() if r.rejected)

    def emit(self, event: ObservationEvent) -> None:
        from ..sink.sink_events import observe_event

        self._base.emit(event)
        observe_event(self, event)

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
        from ..sink.sink_events import observe_event

        paths_tuple = tuple(str(p) for p in (artifact_paths or ()))
        observe_event(
            self,
            OE.make(
                kind,
                session_id=session_id,
                run_id=run_id,
                candidate_id=candidate_id,
                payload=payload,
                artifact_paths=paths_tuple,
            ),
        )

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

    def _emit_search_event(self, event: Dict[str, Any]) -> None:
        emit_search_event(self._live_reporter, event)


__all__ = ["CandidateRecord", "MultiObjectiveSink"]
