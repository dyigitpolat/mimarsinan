"""[TS3] What an LLM driver asked of a model, and where it stops asking.

A campaign that compares an LLM driver against NSGA-II at equal spend needs two
numbers the driver alone can report: what it asked of the model (TS1's
``LlmUsage`` — the model's name and both token directions, never dollars) and
whether the evaluation budget cut it short. Both are sealed in the SAME ledger
the classical driver seals, through the same ``SearchResult`` path.

Usage is counted where the request is MADE, not where it succeeds: pydantic-ai
accumulates into the usage object it is handed, so a run that retried inside
itself counts every request, and a run that ended in an exception still reports
the tokens it spent. A driver that only counted returned results would seal a
number that shrinks exactly when a model misbehaves most.

Stopping is the driver's decision at ITS natural boundary (the TS1 owner
decision: boundary stop + exact ledger). For AgentEvolve that boundary is the
BATCH: once the accountant says the distinct budget is spent, the driver runs no
further batch and opens no further generation — and it asks the model nothing
more either, because a proposal nobody will evaluate is spend nobody authorized.

No test here touches the network: the pydantic-ai agent is replaced at the one
seam that creates it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    search_result_to_jsonable,
)
from mimarsinan.search.optimizers.agent_evolve import AgentEvolveOptimizer
from mimarsinan.search.optimizers.budget import (
    BoundaryStop,
    EvaluationBudget,
    LlmUsage,
    ResourceLedger,
    charge_evaluation,
)
from mimarsinan.search.optimizers.llm.usage import LlmUsageAccumulator, model_name
from mimarsinan.search.results import Candidate, ObjectiveSpec, SearchResult

CANDIDATE_SCHEMA: Dict[str, type] = {
    "reasoning": str,
    "candidates": List[Dict[str, Any]],  # pyright: ignore[reportAssignmentType] — the driver's own schema vocabulary
}
INSIGHTS_SCHEMA: Dict[str, type] = {
    "insights": List[str],  # pyright: ignore[reportAssignmentType] — same vocabulary
}


@dataclass
class _ScriptedRequest:
    """One model request as the fake agent performs it: what it spends, and how it ends."""

    requests: int = 1
    tokens_in: int = 0
    tokens_out: int = 0
    raises: bool = False


class _FakeAgent:
    """A pydantic-ai ``Agent`` stand-in: spends into the usage object it is handed.

    The real agent accumulates every request of one run — retries included —
    into that object, and hands it back mutated whether the run returns or
    raises. This double does exactly that and nothing else.
    """

    def __init__(
        self,
        script: Sequence[_ScriptedRequest],
        *,
        candidates_per_call: int = 2,
    ) -> None:
        self._script = list(script)
        self._candidates_per_call = int(candidates_per_call)
        self.calls = 0
        self.prompts: List[str] = []
        self._next_value = 0

    def _step(self) -> _ScriptedRequest:
        index = min(self.calls, len(self._script) - 1)
        return self._script[index]

    def _candidates(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(self._candidates_per_call):
            self._next_value += 1
            out.append({"x": float(self._next_value)})
        return out

    async def run(
        self,
        prompt: str,
        *,
        output_type: Any = None,
        usage: Any = None,
    ) -> Any:
        step = self._step()
        self.calls += 1
        self.prompts.append(prompt)
        if usage is not None:
            usage.requests += step.requests
            usage.input_tokens += step.tokens_in
            usage.output_tokens += step.tokens_out
        if step.raises:
            raise RuntimeError("the model refused this request")
        if output_type is str:
            return SimpleNamespace(output=json.dumps({
                "reasoning": "scripted",
                "candidates": self._candidates(),
            }))
        return SimpleNamespace(output=SimpleNamespace(
            insights=["scripted insight"],
            constraint_instruction="stay in bounds",
            updated_instruction="stay in bounds",
            performance_insights="bigger x scores better",
            updated_insights="bigger x scores better",
        ))


class _StubbedAgentEvolve(AgentEvolveOptimizer):
    """AgentEvolve with the ONE seam that creates a pydantic-ai agent replaced."""

    fake_agent: Any = None

    def _make_agent(self) -> Any:
        return self.fake_agent


@dataclass
class _ChargingProblem:
    """A cheap problem that charges the TS1 accountant where a real one does."""

    evaluation_budget: Optional[EvaluationBudget] = None
    valid: bool = True
    #: Every configuration this problem was asked about, in order.
    asked: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return [ObjectiveSpec("score", "max")]

    @staticmethod
    def _key(configuration: Dict[str, Any]) -> str:
        return json.dumps(configuration, sort_keys=True, default=str)

    def validate(self, configuration: Dict[str, Any]) -> bool:
        self.asked.append(dict(configuration))
        charge_evaluation(
            self.evaluation_budget, self._key(configuration),
            hit=False, channel="validate",
        )
        return self.valid

    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        charge_evaluation(
            self.evaluation_budget, self._key(configuration),
            hit=False, channel="evaluate",
        )
        return {"score": float(configuration.get("x", 0.0))}

    def meta(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        return {}


def _optimizer(
    agent: _FakeAgent,
    *,
    pop_size: int = 8,
    generations: int = 2,
    candidates_per_batch: int = 2,
    max_regen_rounds: int = 4,
) -> _StubbedAgentEvolve:
    optimizer = _StubbedAgentEvolve(
        pop_size=pop_size,
        generations=generations,
        candidates_per_batch=candidates_per_batch,
        max_regen_rounds=max_regen_rounds,
        model="fake:model",
        config_schema={"x": "float"},
        example_config={"x": 1.0},
        constraints_description="x is a float",
        verbose=False,
    )
    optimizer.fake_agent = agent
    return optimizer


def _call(optimizer: _StubbedAgentEvolve, schema: Dict[str, type]) -> Any:
    return asyncio.run(optimizer._llm_call(  # pyright: ignore[reportPrivateUsage] — the traced call IS the unit under test
        template="prompt", output_schema=schema, call_kind="initial_candidates",
    ))


class TestTheTracedCallCountsWhatTheRequestSpent:
    """One call path, one capture — whatever the request does afterwards."""

    def test_a_run_that_retried_inside_itself_counts_every_request(self):
        agent = _FakeAgent([_ScriptedRequest(requests=3, tokens_in=120, tokens_out=45)])
        optimizer = _optimizer(agent)
        optimizer._llm_usage = LlmUsageAccumulator(model="fake:model")

        _call(optimizer, CANDIDATE_SCHEMA)

        usage = optimizer._llm_usage
        assert usage.calls == 3, "a run's retried requests are requests the run made"
        assert usage.tokens_in == 120
        assert usage.tokens_out == 45

    def test_a_call_that_ended_in_an_exception_still_counts_its_tokens(self):
        agent = _FakeAgent([
            _ScriptedRequest(requests=2, tokens_in=80, tokens_out=10, raises=True),
        ])
        optimizer = _optimizer(agent)
        optimizer._llm_usage = LlmUsageAccumulator(model="fake:model")

        with pytest.raises(RuntimeError):
            _call(optimizer, CANDIDATE_SCHEMA)

        usage = optimizer._llm_usage
        assert usage.calls == 2, "the requests a failed run made were still made"
        assert usage.tokens_in == 80
        assert usage.tokens_out == 10

    def test_both_schema_paths_report_through_the_same_capture(self):
        agent = _FakeAgent([_ScriptedRequest(requests=1, tokens_in=10, tokens_out=5)])
        optimizer = _optimizer(agent)
        optimizer._llm_usage = LlmUsageAccumulator(model="fake:model")

        _call(optimizer, CANDIDATE_SCHEMA)
        _call(optimizer, INSIGHTS_SCHEMA)

        usage = optimizer._llm_usage
        assert usage.calls == 2
        assert (usage.tokens_in, usage.tokens_out) == (20, 10)

    def test_the_accumulator_seals_the_frozen_fact(self):
        usage = LlmUsageAccumulator(model="fake:model")
        usage.record(calls=2, tokens_in=7, tokens_out=3)

        assert usage.sealed() == LlmUsage(
            model="fake:model", calls=2, tokens_in=7, tokens_out=3,
        )

    def test_a_model_object_is_named_by_the_name_it_answers_to(self):
        assert model_name("openai:gpt-4o") == "openai:gpt-4o"
        assert model_name(SimpleNamespace(model_name="function:respond:")) == (
            "function:respond:"
        )


class TestTheDriverSealsWhatItAsked:
    """The LLM block rides the same ledger the classical driver seals."""

    def test_a_metered_search_seals_the_model_and_both_token_directions(self):
        agent = _FakeAgent([_ScriptedRequest(requests=1, tokens_in=100, tokens_out=20)])
        problem = _ChargingProblem(evaluation_budget=EvaluationBudget(limit=None))
        result = _optimizer(agent, generations=2).optimize(problem)

        assert result.ledger is not None
        llm = result.ledger.llm
        assert llm is not None
        assert llm.model == "fake:model"
        assert llm.calls == agent.calls, "every request the search made is counted"
        assert llm.tokens_in == 100 * agent.calls
        assert llm.tokens_out == 20 * agent.calls

    def test_an_unmetered_search_seals_no_ledger_at_all(self):
        agent = _FakeAgent([_ScriptedRequest()])
        result = _optimizer(agent, generations=1).optimize(_ChargingProblem())

        assert result.ledger is None, (
            "a run nobody metered has no counts, and a ledger of zeros would be a "
            "claim about a search that was never measured"
        )

    def test_a_search_the_budget_never_cut_short_says_so(self):
        agent = _FakeAgent([_ScriptedRequest()])
        problem = _ChargingProblem(evaluation_budget=EvaluationBudget(limit=1000))
        result = _optimizer(agent, generations=1, pop_size=2).optimize(problem)

        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is False


class TestTheBatchLoopStopsAtItsBoundary:
    """The batch is AgentEvolve's boundary: no further batch, no further ask."""

    def test_the_driver_runs_no_second_batch_once_the_budget_is_spent(self):
        agent = _FakeAgent([_ScriptedRequest()], candidates_per_call=4)
        problem = _ChargingProblem(evaluation_budget=EvaluationBudget(limit=4))
        # pop_size the first batch cannot fill and four regen rounds to fill it
        # with: only the budget can end this search early.
        result = _optimizer(
            agent, pop_size=8, generations=2, candidates_per_batch=4,
            max_regen_rounds=4,
        ).optimize(problem)

        assert len(problem.asked) == 4, (
            "the batch that spent the budget is the last batch the driver runs"
        )
        assert result.ledger is not None
        assert result.ledger.evaluations_distinct == 4
        assert result.ledger.stopped_at_boundary is True
        assert len(result.all_candidates) == 4

    def test_the_driver_asks_the_model_nothing_after_the_boundary(self):
        agent = _FakeAgent([_ScriptedRequest()], candidates_per_call=4)
        problem = _ChargingProblem(evaluation_budget=EvaluationBudget(limit=4))
        _optimizer(
            agent, pop_size=8, generations=3, candidates_per_batch=4,
            max_regen_rounds=4,
        ).optimize(problem)

        # One proposal, then the insights pass over what it produced. A
        # generation the driver will not evaluate is never proposed.
        assert agent.calls == 2

    def test_the_boundary_only_reports_a_stop_that_denied_more_work(self):
        spent = EvaluationBudget(limit=1)
        boundary = BoundaryStop(spent)
        assert boundary.should_stop() is False
        assert boundary.stopped is False

        spent.charge("evaluate", "candidate-1", hit=False)
        assert boundary.should_stop() is True
        assert boundary.stopped is True

    def test_a_driver_without_an_accountant_never_stops(self):
        boundary = BoundaryStop(None)
        assert boundary.should_stop() is False
        assert boundary.stopped is False


class TestTheLlmBlockReachesTheArtifact:
    """A sealed run is re-priceable research-side, so the artifact must carry it."""

    def test_the_llm_block_round_trips_through_the_result_json(self):
        ledger = ResourceLedger(
            wall_s=1.5,
            evaluations_raw=8,
            evaluations_distinct=6,
            identities_asked=6,
            identities_reasked=1,
            duplicate_rate=1 / 6,
            budget_limit=6,
            stopped_at_boundary=True,
            llm=LlmUsage(
                model="fake:model", calls=9, tokens_in=1234, tokens_out=567,
            ),
        )
        result: SearchResult[Dict[str, Any]] = SearchResult(
            objectives=[ObjectiveSpec("score", "max")],
            best=Candidate(configuration={}, objectives={}, metadata={}),
            ledger=ledger,
        )

        payload = json.loads(json.dumps(search_result_to_jsonable(result)))

        assert payload["ledger"]["llm"] == {
            "model": "fake:model", "calls": 9, "tokens_in": 1234, "tokens_out": 567,
        }
        assert ResourceLedger.from_dict(payload["ledger"]) == ledger

    def test_a_driver_that_asked_no_model_seals_no_llm_block(self):
        ledger = ResourceLedger(
            wall_s=0.5, evaluations_raw=2, evaluations_distinct=2,
            identities_asked=2, identities_reasked=0, duplicate_rate=0.0,
        )
        assert ledger.to_dict()["llm"] is None
