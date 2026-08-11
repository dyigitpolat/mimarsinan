"""The classical optimizer feeds the SAME live-search channel the LLM ones do.

Before this, ``NSGA2Optimizer`` emitted only scalar metrics, so the live search
panel (``gui/static/js/search-live.js``) stayed blank for the whole of a
classical run. The panel's vocabulary is fixed by its handlers; these tests pin
that NSGA2 speaks it, once per GENERATION (a per-candidate stream would be a
1000-frame flood), and that the incumbent leads the reported front.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.optimizers.search_events import PARETO_FRONT_PREVIEW
from mimarsinan.search.results import ObjectiveSpec, order_by_minimax_rank

import pytest

SPECS = (ObjectiveSpec("estimated_accuracy", "max"), ObjectiveSpec("total_params", "min"))


class _ToyProblem:
    """Two continuous variables, two objectives; x0 < 0.1 is infeasible."""

    n_var = 2
    xl = [0.0, 0.0]
    xu = [1.0, 1.0]
    objectives = SPECS

    def decode(self, x) -> Dict[str, Any]:
        return {"a": float(x[0]), "b": float(x[1])}

    def validate(self, cfg) -> bool:
        return cfg["a"] >= 0.1

    def constraint_violation(self, cfg) -> float:
        return 0.0 if cfg["a"] >= 0.1 else 1.0

    def evaluate(self, cfg) -> Dict[str, float]:
        return {
            "estimated_accuracy": 1.0 - abs(cfg["a"] - 0.7),
            "total_params": 1000.0 * cfg["b"] + 10.0,
        }


class _CapturingReporter:
    """Stands in for ``GUIReporter.report`` — records every metric name/value."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def __call__(self, name: str, value: Any, step: int | None = None) -> None:
        self.calls.append((name, value))

    def events(self) -> List[Dict[str, Any]]:
        return [
            json.loads(value) for name, value in self.calls if name == "search_event"
        ]


class _VerdictProblem(_ToyProblem):
    """The toy problem with a WIDE infeasible region and a record of its own verdicts.

    ``constraint_violation`` is called exactly once per evaluated candidate, in
    evaluation order, so this list is the ground truth the reported
    valid/failed counts are checked against — the frames are not allowed to
    merely add up, they must say which is which.
    """

    def __init__(self, infeasible_below: float) -> None:
        self.infeasible_below = float(infeasible_below)
        self.verdicts: List[bool] = []

    def validate(self, cfg) -> bool:
        return cfg["a"] >= self.infeasible_below

    def constraint_violation(self, cfg) -> float:
        feasible = cfg["a"] >= self.infeasible_below
        self.verdicts.append(feasible)
        return 0.0 if feasible else 1.0


class _PinnedProblem(_ToyProblem):
    """A space with exactly ONE point: every bound pinned.

    A user pins a dimension by declaring equal bounds (``core_neurons_bounds:
    [256, 256]``); duplicate elimination then leaves a batch far smaller than
    the configured population, which is precisely when "how many candidates
    does this generation have" stops being "pop_size".
    """

    xl = [0.5, 0.5]
    xu = [0.5, 0.5]


POP_SIZE = 6
GENERATIONS = 3


@pytest.fixture(scope="module")
def run_capture():
    reporter = _CapturingReporter()
    optimizer = NSGA2Optimizer(
        pop_size=POP_SIZE, generations=GENERATIONS, seed=0, verbose=False,
    )
    result = optimizer.optimize(_ToyProblem(), reporter=reporter)
    return reporter, result


@pytest.fixture(scope="module")
def events(run_capture):
    return run_capture[0].events()


def _of_type(events, kind):
    return [e for e in events if e["type"] == kind]


class TestTheChannelIsFed:
    def test_the_search_emits_search_event_frames_at_all(self, events):
        assert events, "a classical run must not leave the live panel blank"

    def test_frames_are_the_panels_own_vocabulary(self, events):
        # search-live.js dispatches on ev.type; an unknown type is dropped
        # silently, which is exactly the blank panel this unit removes.
        assert {e["type"] for e in events} == {
            "generation_start", "candidates_generated",
            "generation_complete", "search_complete",
        }

    def test_one_start_and_one_complete_per_generation_in_order(self, events):
        starts = _of_type(events, "generation_start")
        completes = _of_type(events, "generation_complete")
        assert [e["gen"] for e in starts] == list(range(1, GENERATIONS + 1))
        assert [e["gen"] for e in completes] == list(range(1, GENERATIONS + 1))

    def test_emission_is_per_generation_not_per_candidate(self, events):
        # The whole point: pop_size x generations candidates produce
        # 3 x generations + 1 frames, never one frame per candidate.
        assert len(events) == 3 * GENERATIONS + 1
        assert not _of_type(events, "candidate_result")

    def test_generation_start_carries_the_search_shape(self, events):
        for i, ev in enumerate(_of_type(events, "generation_start")):
            assert ev["total_gens"] == GENERATIONS
            assert ev["pop_size"] == POP_SIZE
            assert ev["phase"] == ("initial" if i == 0 else "evolution")
            assert ev["objectives"] == [
                {"name": "estimated_accuracy", "goal": "max"},
                {"name": "total_params", "goal": "min"},
            ]

    def test_population_size_is_reported_as_candidates_generated(self, events):
        counts = [e["count"] for e in _of_type(events, "candidates_generated")]
        assert len(counts) == GENERATIONS
        assert all(c > 0 for c in counts)

    def test_generation_complete_accounts_for_every_evaluated_candidate(self, events):
        by_gen = {e["gen"]: e for e in _of_type(events, "generation_complete")}
        for ev in _of_type(events, "candidates_generated"):
            done = by_gen[ev["gen"]]
            assert done["valid_count"] + done["failed_count"] == ev["count"]

    def test_generation_complete_carries_the_pareto_front_size(self, events):
        for ev in _of_type(events, "generation_complete"):
            assert ev["pareto_size"] >= 1
            assert len(ev["pareto_front"]) == min(
                ev["pareto_size"], PARETO_FRONT_PREVIEW
            )

    def test_reported_objective_vectors_are_in_user_space(self, events):
        # pymoo minimizes internally; a leaked minimization vector would show
        # accuracy as a negative number in the panel.
        for ev in _of_type(events, "generation_complete"):
            for row in ev["pareto_front"]:
                assert set(row) == {"estimated_accuracy", "total_params"}
                assert 0.0 <= row["estimated_accuracy"] <= 1.0
                assert row["total_params"] >= 10.0

    def test_the_incumbent_leads_the_reported_front(self, events):
        # search-live.js renders pareto_front[0] first; the incumbent is the
        # minimax-rank pick, the same rule select_minimax_rank deploys.
        # Only untruncated fronts can be re-ranked exactly (ranking a prefix
        # is not the induced order), so the check runs on those.
        untruncated = [
            ev for ev in _of_type(events, "generation_complete")
            if ev["pareto_size"] == len(ev["pareto_front"])
        ]
        assert untruncated, "fixture must produce at least one untruncated front"
        for ev in untruncated:
            front = ev["pareto_front"]
            assert order_by_minimax_rank(front, SPECS) == list(range(len(front)))

    def test_search_complete_totals_the_run(self, events, run_capture):
        _, result = run_capture
        done = _of_type(events, "search_complete")
        assert len(done) == 1
        assert done[0]["final_pareto_size"] == len(result.pareto_front)
        assert (
            done[0]["total_valid"] + done[0]["total_failed"]
            == len(result.all_candidates)
        )

    def test_generation_totals_account_for_every_evaluated_candidate(
        self, events, run_capture,
    ):
        _, result = run_capture
        completes = _of_type(events, "generation_complete")
        tallied = sum(e["valid_count"] + e["failed_count"] for e in completes)
        assert tallied == len(result.all_candidates)

    def test_scalar_generation_metrics_still_report(self, run_capture):
        reporter, _ = run_capture
        names = {name for name, _ in reporter.calls}
        assert "Search generation" in names
        assert "Search Pareto size" in names


class TestFrontRowsAreRealEvaluations:
    def test_final_front_rows_come_from_evaluated_candidates(self, events, run_capture):
        _, result = run_capture
        final = _of_type(events, "generation_complete")[-1]
        evaluated = [
            tuple(round(c.objectives[s.name], 9) for s in SPECS)
            for c in result.all_candidates
        ]
        for row in final["pareto_front"]:
            key = tuple(round(row[s.name], 9) for s in SPECS)
            assert key in evaluated, f"{row} is not any evaluated candidate"


class TestCandidateGenerationTagging:
    def test_every_candidate_is_tagged_with_a_one_based_generation(self, run_capture):
        _, result = run_capture
        gens = {c.metadata["generation"] for c in result.all_candidates}
        assert gens == set(range(1, GENERATIONS + 1))


class TestOneGenerationNumberingPerResult:
    """A ``SearchResult`` counts its generations exactly one way.

    Candidate tags were 1-based (as pymoo's ``n_gen`` and the emitted frames
    are) while the history rows enumerated from 0, so one generation answered
    to two different ordinals inside a single result — and the report's x-axis
    read one lower than every badge on the same page. The LLM backends' history
    has always been 1-based; this is the classical one agreeing with it.
    """

    def test_history_rows_carry_the_frames_own_ordinals(self, run_capture, events):
        _, result = run_capture
        assert [h["gen"] for h in result.history] == [
            e["gen"] for e in _of_type(events, "generation_complete")
        ]

    def test_history_rows_are_one_based_and_complete(self, run_capture):
        _, result = run_capture
        assert [h["gen"] for h in result.history] == list(range(1, GENERATIONS + 1))

    def test_no_candidate_claims_a_generation_the_history_never_saw(self, run_capture):
        _, result = run_capture
        assert {c.metadata["generation"] for c in result.all_candidates} == {
            h["gen"] for h in result.history
        }


VERDICT_POP = 10
VERDICT_GENS = 3


@pytest.fixture(scope="module")
def verdict_run():
    """A run with a WIDE infeasible region: both verdicts happen, in unequal numbers."""
    reporter = _CapturingReporter()
    problem = _VerdictProblem(infeasible_below=0.5)
    result = NSGA2Optimizer(
        pop_size=VERDICT_POP, generations=VERDICT_GENS, seed=0, verbose=False,
    ).optimize(problem, reporter=reporter)
    return reporter, result, problem


class TestTheVerdictCountsSayWhichIsWhich:
    """``valid_count``/``failed_count`` are ATTRIBUTED, not merely balanced.

    Every check on those two used to be symmetric under swapping them (they
    only ever had to add up), so a frame that reported failures as successes
    passed the whole file — and the panel would paint a dying search green.
    """

    def test_the_fixture_can_tell_the_two_apart(self, verdict_run):
        _, _, problem = verdict_run
        failed = problem.verdicts.count(False)
        assert failed > 0, "a fixture with no failures proves nothing about failed_count"
        assert failed != problem.verdicts.count(True)

    def test_each_generation_reports_the_verdicts_it_actually_reached(self, verdict_run):
        reporter, _, problem = verdict_run
        cursor = 0
        for frame in _of_type(reporter.events(), "generation_complete"):
            size = frame["valid_count"] + frame["failed_count"]
            batch = problem.verdicts[cursor:cursor + size]
            cursor += size
            assert frame["valid_count"] == batch.count(True)
            assert frame["failed_count"] == batch.count(False)
        assert cursor == len(problem.verdicts), "every evaluation belongs to a generation"

    def test_the_run_totals_are_the_runs_own_verdicts(self, verdict_run):
        reporter, _, problem = verdict_run
        done = _of_type(reporter.events(), "search_complete")[0]
        assert done["total_valid"] == problem.verdicts.count(True)
        assert done["total_failed"] == problem.verdicts.count(False)


PINNED_POP = 6


@pytest.fixture(scope="module")
def pinned_run():
    """A search whose space holds ONE point: the batch cannot be the budget."""
    reporter = _CapturingReporter()
    result = NSGA2Optimizer(
        pop_size=PINNED_POP, generations=GENERATIONS, seed=0, verbose=False,
    ).optimize(_PinnedProblem(), reporter=reporter)
    return reporter, result


class TestTheReportedCountIsTheBatchNotTheBudget:
    """``candidates_generated.count`` is what this generation produced.

    On an open space every generation happens to evaluate exactly ``pop_size``
    candidates, so reporting the CONFIGURED budget looks identical to reporting
    the batch. Pin the bounds and the two part company: duplicate elimination
    leaves one candidate, then none, while the budget still says six.
    """

    def test_a_pinned_space_reports_the_one_candidate_it_has(self, pinned_run):
        reporter, _ = pinned_run
        counts = [e["count"] for e in _of_type(reporter.events(), "candidates_generated")]
        assert counts[0] == 1
        assert counts[0] != PINNED_POP

    def test_no_generation_claims_a_candidate_it_did_not_evaluate(self, pinned_run):
        reporter, result = pinned_run
        events = reporter.events()
        by_gen = {e["gen"]: e for e in _of_type(events, "generation_complete")}
        batches = _of_type(events, "candidates_generated")
        for ev in batches:
            done = by_gen[ev["gen"]]
            assert ev["count"] == done["valid_count"] + done["failed_count"]
        assert sum(e["count"] for e in batches) == len(result.all_candidates)

    def test_the_budget_is_still_reported_as_the_budget(self, pinned_run):
        # pop_size keeps its own frame field — the fix is that `count` stops
        # borrowing it, not that the panel loses the configured population.
        reporter, _ = pinned_run
        for ev in _of_type(reporter.events(), "generation_start"):
            assert ev["pop_size"] == PINNED_POP

    def test_the_collapsed_search_still_hands_over_its_winner(self, pinned_run):
        _, result = pinned_run
        assert result.best.configuration == {"a": 0.5, "b": 0.5}


class TestEmissionIsTelemetry:
    def test_a_broken_reporter_cannot_kill_the_search(self):
        def exploding(name, value, step=None):
            raise RuntimeError("monitor is gone")

        optimizer = NSGA2Optimizer(pop_size=4, generations=2, seed=0, verbose=False)
        result = optimizer.optimize(_ToyProblem(), reporter=exploding)
        assert result.best.configuration, "telemetry failure must not lose the winner"

    def test_no_reporter_is_silent_and_still_searches(self):
        optimizer = NSGA2Optimizer(pop_size=4, generations=2, seed=0, verbose=False)
        result = optimizer.optimize(_ToyProblem(), reporter=None)
        assert result.best.configuration
