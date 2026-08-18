"""[TS3] The compilagent driver seals what it asked, and stops when told to.

The session loop is compilagent's, not ours: it drives the harness, continues
it, and hands back a summary whose usage metadata describes only the LAST
continuation. So mimarsinan observes the harness stream itself — every
iteration's usage report lands in one accumulator, and a run that continued
three times seals three iterations of tokens rather than the last one's.

The driver's natural boundary is a PROPOSAL: the tool that runs candidates is
the only seam that buys evaluations, so once the TS1 accountant says the
distinct budget is spent, that tool refuses and the ledger says the budget cut
the run short. Refusing is deterministic and needs no cooperation from the
model — an agent that keeps proposing simply gets nothing built.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest
from compilagent import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
    harness_registry,
)

from mimarsinan.search.optimizers.budget import EvaluationBudget
from mimarsinan.search.optimizers.compilagent.compilagent_optimizer import (
    CompilagentOptimizer,
)

from .real_problem import make_problem as _real_problem

HARNESS_ID = "usage-scripted"

# Three chips that differ in core COUNT alone: three candidate identities, so
# the spend the accountant reports is the one the harness proposed.
PLANS: List[Dict[str, Any]] = [
    {"description": f"chip of {count} cores",
     "interventions": [("hw.core", "0.count", count)]}
    for count in (50, 60, 70)
]

#: The session's own baseline analysis lays out one candidate before the agent
#: proposes anything — a real resolution, charged like any other (TS1).
BASELINE_SPEND = 1


@dataclass
class _UsageReportingHarness:
    """Runs one plan per iteration and reports what that iteration spent."""

    plans: List[Dict[str, Any]]
    usage_per_run: Dict[str, int]
    id: str = HARNESS_ID
    supported_providers: tuple = ("scripted",)
    example_models: tuple = ()
    iterations: int = 0
    #: Every tool result the scripted agent saw, in order.
    results: List[str] = field(default_factory=list)

    def build_continuation_request(
        self, previous: HarnessRunRequest, snapshot: Any,
    ) -> HarnessRunRequest:
        return previous

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        toolset = request.toolset
        index = self.iterations
        self.iterations += 1

        async def _call(name: str, args: dict, call_id: str) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                tool_name=name, tool_call_id=call_id, tool_args=args,
            )
            result = toolset.by_name(name).invoke(args)
            self.results.append(result)
            yield StreamEvent(
                kind=StreamEventKind.TOOL_RESULT,
                tool_name=name, tool_call_id=call_id, tool_result=result,
            )

        plan = self.plans[index % len(self.plans)]
        propose_args = {
            "interventions": [
                {"target_kind": k, "target_selector": s, "payload": p}
                for (k, s, p) in plan["interventions"]
            ],
            "description": plan["description"],
            "expected_effect": "exercise the path",
        }
        candidate_id = None
        async for ev in _call("propose_candidate", propose_args, f"prop-{index}"):
            if ev.kind is StreamEventKind.TOOL_RESULT and ev.tool_result:
                candidate_id = json.loads(ev.tool_result).get("id")
            yield ev
        if candidate_id:
            async for ev in _call(
                "run_candidate", {"candidate_id": candidate_id}, f"run-{index}",
            ):
                yield ev

        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text="done",
            extra={"usage": dict(self.usage_per_run)},
        )


@pytest.fixture
def usage_harness():
    """Register one scripted harness for the duration of a test."""
    holder: Dict[str, Any] = {
        "usage": {"request_tokens": 100, "response_tokens": 20, "total_tokens": 120},
        "harness": None,
    }

    def _factory():
        harness = _UsageReportingHarness(
            plans=list(PLANS), usage_per_run=dict(holder["usage"]),
        )
        holder["harness"] = harness
        return harness

    harness_registry._factories.pop(HARNESS_ID, None)
    harness_registry.register(HARNESS_ID, _factory)
    try:
        yield holder
    finally:
        harness_registry._factories.pop(HARNESS_ID, None)


def _optimizer(workspace: Path, *, max_continuations: int) -> CompilagentOptimizer:
    return CompilagentOptimizer(
        pop_size=4,
        description=None,
        model="scripted:test",
        harness_id=HARNESS_ID,
        max_candidates=4,
        max_continuations=max_continuations,
        active_objective_names=("total_param_capacity", "fragmentation_pct"),
        workspace_dir=str(workspace),
        verbose=False,
    )


def _metered_problem(limit):
    problem = _real_problem("hardware")
    problem.evaluation_budget = EvaluationBudget(limit=limit)
    return problem


class TestTheSessionSealsWhatItAsked:
    def test_every_continuation_of_the_run_lands_in_one_usage_total(
        self, usage_harness, tmp_path,
    ):
        problem = _metered_problem(limit=None)
        result = _optimizer(tmp_path, max_continuations=1).optimize(problem)

        assert usage_harness["harness"].iterations == 2, (
            "the session continued once — the fixture is only interesting then"
        )
        assert result.ledger is not None
        llm = result.ledger.llm
        assert llm is not None
        assert llm.model == "scripted:test"
        assert llm.tokens_in == 200, "both iterations, not just the last one"
        assert llm.tokens_out == 40
        assert llm.calls == 2

    def test_an_unmetered_session_seals_no_ledger(self, usage_harness, tmp_path):
        problem = _real_problem("hardware")
        result = _optimizer(tmp_path, max_continuations=0).optimize(problem)

        assert result.ledger is None


class TestTheProposalBoundaryStopsTheAgent:
    def test_the_run_tool_refuses_once_the_distinct_budget_is_spent(
        self, usage_harness, tmp_path,
    ):
        problem = _metered_problem(limit=BASELINE_SPEND + 1)
        result = _optimizer(tmp_path, max_continuations=2).optimize(problem)

        harness = usage_harness["harness"]
        refusals = [r for r in harness.results if "evaluation budget" in r]
        assert len(refusals) == 2, "the agent is told why the last two built nothing"
        assert result.ledger is not None
        assert result.ledger.evaluations_distinct == BASELINE_SPEND + 1
        assert result.ledger.stopped_at_boundary is True

        assert len(result.all_candidates) == 3, "the agent kept proposing"
        built = [c for c in result.all_candidates if c.metadata.get("valid", True)]
        assert len(built) == 1, "only the candidate the budget paid for was built"

    def test_a_session_within_its_budget_is_never_refused(
        self, usage_harness, tmp_path,
    ):
        problem = _metered_problem(limit=8)
        result = _optimizer(tmp_path, max_continuations=1).optimize(problem)

        harness = usage_harness["harness"]
        assert not [r for r in harness.results if "evaluation budget" in r]
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is False
