"""Tests for ``MultiObjectiveSink``."""

from __future__ import annotations

from compilagent import (
    CapturingSink,
    EventKind,
    NullSink,
    ObservationEvent,
)

from mimarsinan.search.optimizers.compilagent.sink import (
    CandidateRecord,
    MultiObjectiveSink,
)


def _event(kind, candidate_id, payload):
    return ObservationEvent.make(
        kind, candidate_id=candidate_id, payload=payload,
    )


class TestForwarding:
    def test_emit_forwards_to_base_sink(self):
        base = CapturingSink()
        sink = MultiObjectiveSink(base)
        ev = _event(EventKind.LOG_LINE, "cand-1", {"level": "info", "message": "x"})
        sink.emit(ev)
        assert len(base.events) == 1
        assert base.events[0].kind == EventKind.LOG_LINE.value

    def test_emit_kv_forwards_to_base_sink(self):
        base = CapturingSink()
        sink = MultiObjectiveSink(base)
        sink.emit_kv(EventKind.LOG_LINE, payload={"x": 1}, candidate_id="cand-2")
        assert len(base.events) == 1


class TestObservation:
    def test_objectives_event_populates_record(self):
        sink = MultiObjectiveSink(NullSink())
        sink.emit(
            _event(
                EventKind.OBJECTIVES_RECORDED,
                "cand-3",
                {
                    "candidate_id": "cand-3",
                    "objectives": {
                        "accuracy": {"name": "accuracy", "value": 0.9, "goal": "max", "unit": ""},
                        "latency_ms": {"name": "latency_ms", "value": 12.5, "goal": "min", "unit": "ms"},
                    },
                },
            )
        )
        records = sink.records()
        assert len(records) == 1
        rec = records[0]
        assert rec.candidate_id == "cand-3"
        assert rec.objectives == {"accuracy": 0.9, "latency_ms": 12.5}
        assert rec.objective_metadata["latency_ms"]["unit"] == "ms"
        assert rec.objective_metadata["accuracy"]["goal"] == "max"

    def test_objectives_event_with_plain_floats_populates_record(self):
        sink = MultiObjectiveSink(NullSink())
        sink.emit(
            _event(
                EventKind.OBJECTIVES_RECORDED,
                "cand-4",
                {"candidate_id": "cand-4", "objectives": {"x": 1.5}},
            )
        )
        rec = sink.records()[0]
        assert rec.objectives == {"x": 1.5}
        assert rec.objective_metadata["x"]["goal"] == "min"

    def test_proposed_event_populates_description(self):
        sink = MultiObjectiveSink(NullSink())
        sink.emit(
            _event(
                EventKind.CANDIDATE_PROPOSED,
                "cand-5",
                {"candidate_id": "cand-5", "description": "try thing"},
            )
        )
        rec = sink.records()[0]
        assert rec.description == "try thing"

    def test_rejected_event_marks_failed(self):
        sink = MultiObjectiveSink(NullSink())
        sink.emit(
            _event(EventKind.CANDIDATE_PROPOSED, "cand-6", {"description": "x"})
        )
        sink.emit(
            _event(
                EventKind.CANDIDATE_REJECTED,
                "cand-6",
                {"candidate_id": "cand-6", "reason": "compile_failed"},
            )
        )
        rec = sink.records()[0]
        assert rec.rejected is True
        assert rec.reject_reason == "compile_failed"
        assert sink.failed_records() and sink.failed_records()[0].candidate_id == "cand-6"


class TestConfigurationAttachment:
    def test_attach_configuration_before_event(self):
        sink = MultiObjectiveSink(NullSink())
        sink.attach_configuration(
            "cand-7", {"model_config": {"a": 1}, "platform_constraints": {}}
        )
        sink.emit(
            _event(
                EventKind.OBJECTIVES_RECORDED,
                "cand-7",
                {"candidate_id": "cand-7", "objectives": {"x": 1.0}},
            )
        )
        rec = sink.records()[0]
        assert rec.configuration == {"model_config": {"a": 1}, "platform_constraints": {}}

    def test_attach_configuration_after_event_overrides(self):
        sink = MultiObjectiveSink(NullSink())
        sink.emit(
            _event(
                EventKind.OBJECTIVES_RECORDED,
                "cand-8",
                {"candidate_id": "cand-8", "objectives": {"x": 1.0}},
            )
        )
        sink.attach_configuration(
            "cand-8", {"model_config": {"b": 2}, "platform_constraints": {}}
        )
        rec = sink.records()[0]
        assert rec.configuration == {"model_config": {"b": 2}, "platform_constraints": {}}


class TestLiveReporter:
    def _capture_reporter(self):
        events: list = []

        def _r(name, value):
            if name == "search_event":
                import json as _json
                try:
                    events.append(_json.loads(value))
                except Exception:
                    pass

        return events, _r

    def test_objectives_event_emits_dedicated_event(self):
        events, reporter = self._capture_reporter()
        sink = MultiObjectiveSink(NullSink(), live_reporter=reporter)
        sink.emit(
            _event(
                EventKind.OBJECTIVES_RECORDED,
                "cand-9",
                {"candidate_id": "cand-9", "objectives": {"x": {"value": 1.0, "goal": "min"}}},
            )
        )
        types = {ev["type"] for ev in events}
        # Only the dedicated compilagent event is emitted; we no longer
        # forge AgentEvolve-shaped `candidate_result` events.
        assert "compilagent_candidate_objectives" in types
        assert "candidate_result" not in types

    def test_compiler_pass_event_is_forwarded(self):
        events, reporter = self._capture_reporter()
        sink = MultiObjectiveSink(NullSink(), live_reporter=reporter)
        sink.emit(
            _event(
                EventKind.COMPILER_PASS,
                None,
                {"stage": "validate", "name": "validate_detailed", "duration_ms": 1.5},
            )
        )
        assert any(ev["type"] == "compilagent_compile_phase" for ev in events)

    def test_tool_call_events_are_forwarded(self):
        events, reporter = self._capture_reporter()
        sink = MultiObjectiveSink(NullSink(), live_reporter=reporter)
        sink.emit(
            _event(
                EventKind.TOOL_CALL_STARTED,
                None,
                {"tool_name": "inspect_softcores"},
            )
        )
        assert any(ev["type"] == "compilagent_tool_call" for ev in events)

    def test_compile_completed_records_diagnostics_and_emits(self):
        events, reporter = self._capture_reporter()
        sink = MultiObjectiveSink(NullSink(), live_reporter=reporter)
        sink.emit(
            _event(
                EventKind.COMPILE_COMPLETED,
                "cand-x",
                {
                    "candidate_id": "cand-x",
                    "ok": False,
                    "elapsed_ms": 12.0,
                    "diagnostics": "axon overflow: 1024 > 256",
                    "warnings": ["weight matrix too wide"],
                },
            )
        )
        rec = sink.records()[0]
        assert rec.compile_ok is False
        assert "axon overflow" in (rec.compile_diagnostics or "")
        assert any(
            ev["type"] == "compilagent_candidate_compiled" and ev["ok"] is False
            for ev in events
        )

    def test_session_helpers_emit_named_events(self):
        events, reporter = self._capture_reporter()
        sink = MultiObjectiveSink(NullSink(), live_reporter=reporter)
        sink.emit_session_start(
            workload_id="w1", backend_id="mimarsinan_layout",
            model="mistral:m", harness="pydantic_ai",
            max_candidates=4, max_continuations=2,
            objectives=[{"name": "x", "goal": "min"}],
        )
        sink.emit_pareto_update(pareto_front=[{"objectives": {"x": 1.0}}])
        sink.emit_session_complete(
            total_valid=2, total_failed=1, pareto_size=2,
            best_objectives={"x": 0.5}, elapsed_ms=1234.5,
        )
        types = [ev["type"] for ev in events]
        assert types == [
            "compilagent_session_start",
            "compilagent_pareto_update",
            "compilagent_session_complete",
        ]
        assert events[0]["model"] == "mistral:m"
        # No `primary_objective` in the session_start event — every axis
        # is equally weighted in the multi-objective leaderboard.
        assert "primary_objective" not in events[0]
        assert events[2]["best_objectives"] == {"x": 0.5}


class TestSplits:
    def test_successful_records_excludes_rejected_and_objectiveless(self):
        sink = MultiObjectiveSink(NullSink())
        # cand-A: full objectives
        sink.emit(
            _event(
                EventKind.OBJECTIVES_RECORDED,
                "cand-A",
                {"candidate_id": "cand-A", "objectives": {"x": 1.0}},
            )
        )
        # cand-B: rejected
        sink.emit(_event(EventKind.CANDIDATE_PROPOSED, "cand-B", {}))
        sink.emit(
            _event(
                EventKind.CANDIDATE_REJECTED,
                "cand-B",
                {"candidate_id": "cand-B", "reason": "compile_failed"},
            )
        )
        # cand-C: only proposed, no objectives, not rejected -> excluded from both
        sink.emit(_event(EventKind.CANDIDATE_PROPOSED, "cand-C", {}))
        successful = [r.candidate_id for r in sink.successful_records()]
        failed = [r.candidate_id for r in sink.failed_records()]
        assert successful == ["cand-A"]
        assert failed == ["cand-B"]
