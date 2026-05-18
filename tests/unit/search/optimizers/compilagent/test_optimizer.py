"""End-to-end tests for ``CompilagentOptimizer``.

Drive the optimizer through a synthetic harness that proposes a few
``Plan``s against the fake ``JointArchHwProblem`` from
``test_backend.py``. The goal is to assert that the resulting
``SearchResult`` mirrors the shape AgentEvolve and NSGA2 produce: a
populated Pareto front, a non-empty ``best``, ``all_candidates`` covering
both successes and failures, and a one-row ``history`` summary.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pytest
from compilagent import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
    harness_registry,
)

from mimarsinan.search.optimizers.compilagent.compilagent_optimizer import (
    CompilagentOptimizer,
)
from mimarsinan.search.search_space_description import SearchSpaceDescription

from .test_backend import _make_problem  # noqa: E402


# A deterministic harness mirrors the compilagent test pattern: it
# yields a fixed sequence of TOOL_CALL/TOOL_RESULT events that exercise
# the canonical 8-tool surface, then ends with RUN_FINISHED.


@dataclass
class _ScriptedHarness:
    """Drives a fixed sequence of plans against the session.

    Each entry in ``plans`` is a dict with `description` + a list of
    `(target_kind, selector, payload)` triples. The harness:

    1. inspects workload + search space (read-only),
    2. proposes each plan via `propose_candidate`,
    3. runs each through `run_candidate`,
    4. compares runs and finishes.
    """

    plans: List[dict]
    fail_at: int = -1  # candidate index that should produce a compile failure
    id: str = "scripted"
    supported_providers: tuple = ("scripted",)
    example_models: tuple = ()

    def build_continuation_request(self, previous, snapshot):  # noqa: ANN001
        # Single-iteration test — never asked for a continuation.
        raise AssertionError("continuation should not be requested")

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset

        async def _call(name: str, args: dict, call_id: str) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                tool_name=name,
                tool_call_id=call_id,
                tool_args=args,
            )
            try:
                result = toolset.by_name(name).invoke(args)
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_RESULT,
                    tool_name=name,
                    tool_call_id=call_id,
                    tool_result=result,
                )
            except ValueError as exc:
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_ERROR,
                    tool_name=name,
                    tool_call_id=call_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        async for ev in _call("inspect_workload", {}, "init-1"):
            yield ev
        async for ev in _call("inspect_search_space", {}, "init-2"):
            yield ev

        for idx, plan_spec in enumerate(self.plans):
            interventions = [
                {"target_kind": k, "target_selector": s, "payload": p}
                for (k, s, p) in plan_spec["interventions"]
            ]
            propose_args = {
                "interventions": interventions,
                "description": plan_spec["description"],
                "expected_effect": "exercise the path",
            }
            registered_id = None
            async for ev in _call(
                "propose_candidate", propose_args, f"prop-{idx}",
            ):
                if ev.kind is StreamEventKind.TOOL_RESULT and ev.tool_result:
                    try:
                        registered_id = json.loads(ev.tool_result)["id"]
                    except Exception:
                        registered_id = None
                yield ev
            if registered_id:
                async for ev in _call(
                    "run_candidate", {"candidate_id": registered_id}, f"run-{idx}",
                ):
                    yield ev

        async for ev in _call("compare_runs", {}, "cmp"):
            yield ev

        yield StreamEvent(kind=StreamEventKind.RUN_FINISHED, text="done")


@pytest.fixture
def scripted_harness_registered():
    """Register a fresh scripted harness factory for the duration of one test."""

    holder = {"plans": [], "fail_at": -1}

    def _factory():
        return _ScriptedHarness(plans=list(holder["plans"]), fail_at=int(holder["fail_at"]))

    harness_id = "scripted"
    # Make sure no stale registration interferes
    if harness_id in harness_registry.ids():
        harness_registry._factories.pop(harness_id)
    harness_registry.register(harness_id, _factory)
    try:
        yield holder
    finally:
        harness_registry._factories.pop(harness_id, None)


def _make_optimizer(workspace: Path) -> CompilagentOptimizer:
    return CompilagentOptimizer(
        pop_size=4,
        description=None,
        model="scripted:test",
        harness_id="scripted",
        max_candidates=4,
        max_continuations=0,
        active_objective_names=(
            "total_param_capacity",
            "param_utilization_pct",
            "fragmentation_pct",
        ),
        workspace_dir=str(workspace),
        verbose=False,
    )


class TestSuccessfulRun:
    def test_optimizer_returns_search_result_with_pareto_front(
        self, scripted_harness_registered, tmp_path,
    ):
        scripted_harness_registered["plans"] = [
            {
                "description": "boost axons on core 0",
                "interventions": [
                    ("hw.core", "0.max_axons", 512),
                ],
            },
            {
                "description": "boost neurons on core 0",
                "interventions": [
                    ("hw.core", "0.max_neurons", 512),
                ],
            },
            {
                "description": "shrink count",
                "interventions": [
                    ("hw.core", "0.count", 50),
                ],
            },
        ]
        problem = _make_problem()
        optimizer = _make_optimizer(tmp_path)
        result = optimizer.optimize(problem)

        # Three valid candidates evaluated
        assert len(result.all_candidates) == 3
        valid = [c for c in result.all_candidates if c.metadata.get("valid", True)]
        assert len(valid) == 3
        # Pareto front non-empty and `best` populated
        assert result.pareto_front
        assert result.best.objectives
        # History summary is one entry
        assert len(result.history) == 1
        h = result.history[0]
        assert h["valid_count"] == 3
        assert h["failed_count"] == 0
        assert h["pareto_size"] == len(result.pareto_front)


class TestRejectedCandidates:
    def test_failed_compile_records_invalid_candidate(
        self, scripted_harness_registered, tmp_path,
    ):
        # Use count=9999 — the fake problem's `validate_detailed`
        # rejects this with a simulated packing failure that surfaces
        # only AFTER basic intervention validation passes. This is the
        # realistic "compile_failed" path: validate_intervention is
        # happy but the deeper problem-side check rejects.
        scripted_harness_registered["plans"] = [
            {
                "description": "broken plan",
                "interventions": [
                    ("hw.core", "0.count", 9999),
                ],
            },
            {
                "description": "good plan",
                "interventions": [
                    ("hw.core", "0.max_axons", 256),
                ],
            },
        ]
        problem = _make_problem()
        optimizer = _make_optimizer(tmp_path)
        result = optimizer.optimize(problem)

        assert len(result.all_candidates) == 2
        valid = [c for c in result.all_candidates if c.metadata.get("valid", True)]
        invalid = [c for c in result.all_candidates if not c.metadata.get("valid", True)]
        assert len(valid) == 1
        assert len(invalid) == 1
        assert result.history[0]["failed_count"] == 1
        # The pareto front should contain only the valid candidate
        assert len(result.pareto_front) == 1


class TestReporterEmitsSearchEvents:
    def test_search_events_use_compilagent_vocabulary(
        self, scripted_harness_registered, tmp_path,
    ):
        """The compilagent optimizer emits its own dedicated event
        vocabulary (``compilagent_session_start`` / ``compilagent_*`` per
        beat / ``compilagent_session_complete``). It deliberately does
        not emit AgentEvolve-shaped events so the two live monitors stay
        decoupled."""

        scripted_harness_registered["plans"] = [
            {
                "description": "good plan",
                "interventions": [
                    ("hw.core", "0.max_axons", 256),
                ],
            },
        ]
        problem = _make_problem()
        optimizer = _make_optimizer(tmp_path)

        events: list[dict] = []

        def _reporter(name, value):
            if name == "search_event":
                try:
                    events.append(json.loads(value))
                except Exception:
                    pass

        optimizer.optimize(problem, reporter=_reporter)
        types = {ev.get("type") for ev in events}
        assert {
            "compilagent_session_start",
            "compilagent_session_complete",
            "compilagent_pareto_update",
        } <= types
        # No AgentEvolve-shaped types leak through.
        assert not (types & {"generation_start", "generation_complete", "search_complete", "candidate_result", "llm_trace"})
