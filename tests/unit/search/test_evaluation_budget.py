"""[TS1] The search's ONE evaluation accountant, and the ledger it seals.

A campaign compares optimizers at EQUAL SPEND, so "what did this search cost"
needs exactly one definition and exactly one place that counts it. The currency
is the candidate IDENTITY: the first channel that spends evaluator work on it
pays (a DISTINCT evaluation), and every later ask about that identity is a
DUPLICATE — it bought no new candidate. No driver counts for itself, and no
count is a driver's opinion.

The COMPARABLE axis is coarser still. A driver asks about one candidate through
as many channels as it likes (NSGA-II screens then scores, so two per
proposal), which puts a structural floor under any rate counted in CALLS — the
measured 0.5556 for a run whose drivers re-proposed 2 of 18 candidates. So
``duplicate_rate`` counts ROUNDS of asking: every channel's first look at a
candidate belongs to the round its proposal opened, and a channel asking again
about a candidate it already asked about opens a NEW round — the re-proposal.

The ledger seals FACTS only — wall, raw/distinct evaluations, both sides of the
rate, the declared limit, whether the run stopped at a boundary, and (later
stages) the LLM usage. Money is priced research-side from a price table; a
framework that seals dollars seals a price list nobody can re-run.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    make_platform_resolver,
    search_result_to_jsonable,
)
from mimarsinan.search.optimizers.budget import (
    BoundaryStop,
    EvaluationBudget,
    LlmUsage,
    ResourceLedger,
    charge_evaluation,
    problem_budget,
    seal_ledger,
)
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.optimizers.sampling_optimizer import (
    RandomStrategy,
    SamplingOptimizer,
)
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.problems.joint.types import (
    CONSTRAINT_CHANNEL,
    EVALUATE_CHANNEL,
    LAYOUT_CHANNEL,
)
from mimarsinan.search.results import Candidate, ObjectiveSpec, SearchResult

HW_OBJECTIVES = ["total_param_capacity", "param_utilization_pct", "fragmentation_pct"]
SPECS = (
    ObjectiveSpec("estimated_accuracy", "max"),
    ObjectiveSpec("total_params", "min"),
)


class TestTheAccountantCountsDistinctWork:
    """Distinct == evaluator work done; duplicate == work the cache absorbed."""

    def test_the_first_charge_of_a_key_is_distinct(self):
        budget = EvaluationBudget(limit=4)
        budget.charge(EVALUATE_CHANNEL, "a", hit=False)

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 1

    def test_an_identity_is_charged_once_however_many_channels_spend_on_it(self):
        # The currency is the candidate identity, not the call: a candidate
        # screened by the constraint channel, scored by the evaluate channel
        # and laid out for an agent cost the run ONE evaluation, and the count
        # must say so whichever channel got there first. The CALLS still
        # happened, and ``raw_calls`` is a call count, so it says three.
        budget = EvaluationBudget(limit=4)
        for channel in (CONSTRAINT_CHANNEL, EVALUATE_CHANNEL, LAYOUT_CHANNEL):
            budget.charge(channel, "a", hit=False)

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 3
        # Three channels looking at ONE proposal is one round of asking, so
        # nothing here was re-proposed — the axis a campaign compares must not
        # move because a driver happens to consult a third channel.
        assert (budget.identities_asked, budget.identities_reasked) == (1, 0)
        assert budget.duplicate_rate == 0.0

    def test_a_call_about_an_identity_nobody_spent_is_not_an_evaluation_call(self):
        # A cheap predicate refused this candidate before anything was built,
        # so the run never evaluated it. Neither the refusal nor the re-asks a
        # cache answers afterwards are evaluation calls — counting them would
        # let a search that resolves NOTHING seal a duplicate rate of 1.0.
        budget = EvaluationBudget(limit=4)
        for _ in range(5):
            budget.charge(EVALUATE_CHANNEL, "a", hit=True)

        assert (budget.distinct_spent, budget.raw_calls) == (0, 0)
        assert (budget.identities_asked, budget.identities_reasked) == (0, 0)
        assert budget.duplicate_rate == 0.0

    def test_a_channel_that_redoes_the_work_still_buys_no_new_candidate(self):
        # An UNCACHED channel (the layout introspection seam) really does the
        # resolution again, but the currency is the identity and this run has
        # already paid for it: the call is counted, the budget is not spent.
        budget = EvaluationBudget(limit=2)
        budget.charge(LAYOUT_CHANNEL, "a", hit=False)
        budget.charge(LAYOUT_CHANNEL, "a", hit=False)

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 2
        assert not budget.exhausted

    def test_every_charge_the_accountant_is_handed_is_a_call(self):
        # ``raw_calls`` is a CALL count, not a number derived from the identity
        # set: a seam that charges one identity twice made two calls, and a
        # ledger field named ``evaluations_raw`` must be able to say so.
        budget = EvaluationBudget(limit=None)
        budget.charge(EVALUATE_CHANNEL, "a", hit=False)
        budget.charge(EVALUATE_CHANNEL, "a", hit=False)

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 2

    def test_the_accountant_says_which_identities_it_has_spent(self):
        # The one question the seam helper asks to tell a first spend from a
        # re-ask; without it every seam would have to keep its own set.
        budget = EvaluationBudget(limit=None)
        budget.charge(EVALUATE_CHANNEL, "a", hit=False)

        assert budget.has_spent("a")
        assert not budget.has_spent("b")

    def test_a_duplicate_is_never_charged_against_the_budget(self):
        budget = EvaluationBudget(limit=4)
        budget.charge(EVALUATE_CHANNEL, "a", hit=False)
        for _ in range(10):
            budget.charge(EVALUATE_CHANNEL, "a", hit=True)

        assert budget.distinct_spent == 1, "a cache hit is not evaluator work"
        assert budget.raw_calls == 11, "the raw call still happened and is recorded"

    def test_exhaustion_lands_at_exactly_the_limit(self):
        budget = EvaluationBudget(limit=3)
        for i in range(2):
            budget.charge(EVALUATE_CHANNEL, f"k{i}", hit=False)
            assert not budget.exhausted

        budget.charge(EVALUATE_CHANNEL, "k2", hit=False)
        assert budget.exhausted, "B distinct evaluations exhaust a budget of B"

    def test_duplicates_alone_never_exhaust_a_budget(self):
        # Exhaustion reads DISTINCT spend, not raw calls: a search that keeps
        # re-proposing one candidate has bought exactly one evaluation.
        budget = EvaluationBudget(limit=3)
        budget.charge(EVALUATE_CHANNEL, "a", hit=False)
        budget.charge(EVALUATE_CHANNEL, "b", hit=False)
        for _ in range(5):
            budget.charge(EVALUATE_CHANNEL, "a", hit=True)

        assert budget.raw_calls > 3, "the fixture must out-call the limit"
        assert not budget.exhausted
        budget.charge(EVALUATE_CHANNEL, "c", hit=False)
        assert budget.exhausted

    def test_a_budget_without_a_limit_meters_but_never_stops(self):
        budget = EvaluationBudget(limit=None)
        for i in range(50):
            budget.charge(EVALUATE_CHANNEL, f"k{i}", hit=False)

        assert budget.distinct_spent == 50
        assert not budget.exhausted

    def test_an_unused_accountant_reports_no_duplicates(self):
        assert EvaluationBudget(limit=2).duplicate_rate == 0.0

    def test_charging_through_the_seam_helper_needs_no_none_check(self):
        budget = EvaluationBudget(limit=None)
        charge_evaluation(budget, "a", hit=False, channel=EVALUATE_CHANNEL)
        charge_evaluation(budget, "a", hit=True, channel=EVALUATE_CHANNEL)
        charge_evaluation(None, "a", hit=False, channel=EVALUATE_CHANNEL)

        assert (budget.distinct_spent, budget.raw_calls) == (1, 2)


class TestTheComparableAxisCountsRoundsOfAsking:
    """``duplicate_rate`` must mean the same thing to a one-channel driver and
    a three-channel one, or a campaign ranking optimizers by it reads the
    channel count instead of the search behaviour."""

    def test_the_same_channel_asking_again_is_a_re_proposal(self):
        budget = EvaluationBudget(limit=None)
        for _ in range(4):
            budget.charge(CONSTRAINT_CHANNEL, "a", hit=False)

        assert (budget.identities_asked, budget.identities_reasked) == (4, 3)
        assert budget.duplicate_rate == pytest.approx(0.75)

    def test_a_second_channels_first_look_joins_the_round_it_did_not_open(self):
        budget = EvaluationBudget(limit=None)
        for key in ("a", "b", "c"):
            budget.charge(CONSTRAINT_CHANNEL, key, hit=False)
            budget.charge(EVALUATE_CHANNEL, key, hit=True)

        assert budget.raw_calls == 6, "six calls really happened"
        assert (budget.identities_asked, budget.identities_reasked) == (3, 0)
        assert budget.duplicate_rate == 0.0

    def test_adding_a_channel_does_not_move_the_rate(self):
        # THE cross-driver property: the same proposal stream, screened and
        # scored instead of only scored, must seal the same number.
        proposals = ["a", "b", "a", "c"]
        one_channel = EvaluationBudget(limit=None)
        two_channels = EvaluationBudget(limit=None)
        for key in proposals:
            one_channel.charge(EVALUATE_CHANNEL, key, hit=False)
            two_channels.charge(CONSTRAINT_CHANNEL, key, hit=False)
            two_channels.charge(EVALUATE_CHANNEL, key, hit=True)

        assert two_channels.raw_calls == 2 * one_channel.raw_calls
        assert two_channels.duplicate_rate == one_channel.duplicate_rate
        assert two_channels.duplicate_rate == pytest.approx(0.25)

    def test_a_batched_driver_that_screens_all_then_scores_all_reads_zero(self):
        # A driver need not interleave its channels: agent-style backends
        # validate a whole batch and only then evaluate it. Nothing there was
        # re-proposed, so the rate must be 0.0 — a rule keyed on "the previous
        # ask was about another candidate" would have read 1.0 here.
        budget = EvaluationBudget(limit=None)
        batch = ["a", "b", "c"]
        for key in batch:
            budget.charge(CONSTRAINT_CHANNEL, key, hit=False)
        for key in batch:
            budget.charge(EVALUATE_CHANNEL, key, hit=True)

        assert (budget.identities_asked, budget.identities_reasked) == (3, 0)
        assert budget.duplicate_rate == 0.0

    def test_a_re_proposal_reopens_the_round_for_every_channel(self):
        budget = EvaluationBudget(limit=None)
        for _ in range(3):
            budget.charge(CONSTRAINT_CHANNEL, "a", hit=True)
            budget.charge(EVALUATE_CHANNEL, "a", hit=True)
        # Nothing was ever spent on "a", so nothing above counted at all.
        assert (budget.identities_asked, budget.raw_calls) == (0, 0)

        budget.charge(CONSTRAINT_CHANNEL, "a", hit=False)
        for _ in range(3):
            budget.charge(CONSTRAINT_CHANNEL, "a", hit=True)
            budget.charge(EVALUATE_CHANNEL, "a", hit=True)

        assert (budget.identities_asked, budget.identities_reasked) == (4, 3)
        assert budget.duplicate_rate == pytest.approx(0.75)

    def test_a_problem_without_an_accountant_reports_none(self):
        assert problem_budget(object()) is None

    def test_a_problem_carrying_something_else_fails_loud(self):
        class _Wrong:
            evaluation_budget = 12

        with pytest.raises(TypeError, match="evaluation_budget"):
            problem_budget(_Wrong())


class TestTheLedgerIsFacts:
    def test_the_ledger_reads_the_accountants_own_counts(self):
        budget = EvaluationBudget(limit=8)
        budget.charge(EVALUATE_CHANNEL, "a", hit=False)
        budget.charge(EVALUATE_CHANNEL, "b", hit=False)
        budget.charge(EVALUATE_CHANNEL, "a", hit=True)

        ledger = seal_ledger(budget, wall_s=1.5, stopped_at_boundary=True)

        assert ledger is not None
        assert ledger.evaluations_distinct == 2
        assert ledger.evaluations_raw == 3
        assert (ledger.identities_asked, ledger.identities_reasked) == (3, 1)
        assert ledger.duplicate_rate == pytest.approx(1.0 / 3.0)
        assert ledger.budget_limit == 8
        assert ledger.stopped_at_boundary is True
        assert ledger.wall_s == pytest.approx(1.5)

    def test_the_sealed_rate_is_derivable_from_the_two_counts_beside_it(self):
        # The rate is a DERIVED fact, so the ledger seals its numerator and
        # denominator too: an analysis can re-derive it, pool it across runs,
        # or normalize it without re-running the search.
        budget = EvaluationBudget(limit=None)
        for key in ("a", "b", "a", "a"):
            budget.charge(EVALUATE_CHANNEL, key, hit=False)

        ledger = seal_ledger(budget, wall_s=0.5, stopped_at_boundary=False)

        assert ledger is not None
        assert (ledger.identities_asked, ledger.identities_reasked) == (4, 2)
        assert ledger.duplicate_rate == pytest.approx(
            ledger.identities_reasked / ledger.identities_asked
        )

    def test_no_accountant_seals_no_ledger(self):
        assert seal_ledger(None, wall_s=1.0, stopped_at_boundary=False) is None

    def test_the_ledger_round_trips_through_json(self):
        ledger = ResourceLedger(
            wall_s=12.5,
            evaluations_raw=40,
            evaluations_distinct=32,
            identities_asked=20,
            identities_reasked=4,
            duplicate_rate=0.2,
            budget_limit=32,
            stopped_at_boundary=True,
            llm=LlmUsage(model="a-model", calls=3, tokens_in=120, tokens_out=45),
        )

        restored = ResourceLedger.from_dict(json.loads(json.dumps(ledger.to_dict())))

        assert restored == ledger

    def test_a_ledger_without_llm_usage_round_trips_too(self):
        ledger = ResourceLedger(
            wall_s=0.25,
            evaluations_raw=4,
            evaluations_distinct=4,
            identities_asked=4,
            identities_reasked=0,
            duplicate_rate=0.0,
        )

        assert ResourceLedger.from_dict(ledger.to_dict()) == ledger
        assert ledger.to_dict()["llm"] is None

    def test_llm_usage_seals_the_model_and_both_token_directions(self):
        usage = LlmUsage(model="m", calls=2, tokens_in=10, tokens_out=20)

        assert usage.to_dict() == {
            "model": "m", "calls": 2, "tokens_in": 10, "tokens_out": 20,
        }

    def test_the_ledger_prices_nothing(self):
        # Owner decision: the framework seals raw facts; dollars are computed
        # research-side from a price table, so a sealed run can be re-priced.
        payload = ResourceLedger(
            wall_s=1.0, evaluations_raw=1, evaluations_distinct=1,
            identities_asked=1, identities_reasked=0,
            duplicate_rate=0.0, llm=LlmUsage(model="m", calls=1),
        ).to_dict()
        flat = json.dumps(payload).lower()

        for word in ("cost", "price", "usd", "dollar"):
            assert word not in flat


LEGACY_RESULT_KEYS = {
    "objectives", "best", "pareto_front", "all_candidates", "history",
}


def _bare_result() -> SearchResult[Dict[str, Any]]:
    best = Candidate(configuration={"a": 1}, objectives={"total_params": 3.0})
    return SearchResult(objectives=[SPECS[1]], best=best, pareto_front=[best])


class TestSearchResultCarriesTheLedger:
    def test_a_result_defaults_to_no_ledger(self):
        # Additive-optional: an artifact written before TS1 still loads.
        assert _bare_result().ledger is None

    def test_a_ledgerless_result_serializes_exactly_as_it_used_to(self):
        payload = search_result_to_jsonable(_bare_result())

        assert set(payload) == LEGACY_RESULT_KEYS

    def test_a_sealed_ledger_reaches_the_artifact(self):
        ledger = ResourceLedger(
            wall_s=2.0, evaluations_raw=9, evaluations_distinct=7,
            identities_asked=9, identities_reasked=2,
            duplicate_rate=2.0 / 9.0, budget_limit=7, stopped_at_boundary=True,
        )
        result = SearchResult(
            objectives=_bare_result().objectives,
            best=_bare_result().best,
            ledger=ledger,
        )

        payload = search_result_to_jsonable(result)

        assert set(payload) == LEGACY_RESULT_KEYS | {"ledger"}
        assert payload["ledger"] == ledger.to_dict()
        assert ResourceLedger.from_dict(payload["ledger"]) == ledger


def _hw_pipeline_config(allow_scheduling: bool = True) -> Dict[str, Any]:
    return {
        "device": "cpu",
        "input_shape": (1, 8, 8),
        "num_classes": 4,
        "target_tq": 4,
        "weight_bits": 4,
        "lr": 0.001,
        "allow_scheduling": allow_scheduling,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "model_config": {
            "mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU",
        },
    }


def _hw_problem(
    budget=None, validate_fn=None, floor=0.0, constraint_fn=None,
    core_count_bounds=(8, 64), core_dim_bounds=(64, 256), allow_scheduling=True,
) -> JointArchHwProblem:
    """A hardware-only search: no training, so an evaluation is pure mapping work."""
    cfg = _hw_pipeline_config(allow_scheduling)
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(cfg["input_shape"]),
        num_classes=cfg["num_classes"],
        target_tq=cfg["target_tq"],
        lr=cfg["lr"],
        search_mode="hardware",
        builder_factory=SimpleMLPBuilder,
        arch_options=(),
        model_config_assembler=lambda raw: dict(raw),
        fixed_model_config=dict(cfg["model_config"]),
        validate_fn=validate_fn,
        constraint_fn=constraint_fn,
        onchip_min_fraction=floor,
        platform_resolver=make_platform_resolver(cfg),
        active_objective_names=HW_OBJECTIVES,
        num_core_types=1,
        core_axons_bounds=core_dim_bounds,
        core_neurons_bounds=core_dim_bounds,
        core_count_bounds=core_count_bounds,
        evaluation_budget=budget,
    )


def _unpackable_problem(budget) -> JointArchHwProblem:
    """One 64x64 core and no scheduling: the model RESOLVES, then fails to pack.

    The candidate that costs a full evaluation and is scored a penalty anyway —
    a structural refusal, by contrast, builds nothing.
    """
    return _hw_problem(
        budget, core_count_bounds=(1, 1), core_dim_bounds=(64, 64),
        allow_scheduling=False,
    )


def _configuration_at(problem: JointArchHwProblem, fraction: float) -> Dict[str, Any]:
    x = np.asarray(problem.xl) * (1.0 - fraction) + np.asarray(problem.xu) * fraction
    return problem.decode(x)


def _mid_configuration(problem: JointArchHwProblem) -> Dict[str, Any]:
    return _configuration_at(problem, 0.5)


class TestTheCacheSeamIsTheAccountant:
    """The joint problem charges where a cache decides whether work happens."""

    def test_the_first_evaluation_of_a_candidate_is_distinct(self):
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget)

        problem.evaluate(_mid_configuration(problem))

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 1

    def test_re_evaluating_the_same_candidate_is_a_duplicate(self):
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget)
        configuration = _mid_configuration(problem)

        first = problem.evaluate(configuration)
        again = problem.evaluate(configuration)

        assert again == first, "the fixture must actually re-ask the same candidate"
        assert budget.distinct_spent == 1, "a cached candidate costs no evaluation"
        assert budget.raw_calls == 2

    def test_a_penalty_evaluation_is_charged(self):
        # An infeasible candidate consumed real evaluator work before it was
        # scored a penalty; a budget that forgave it would pay for search
        # strategies that propose junk.
        budget = EvaluationBudget(limit=None)
        problem = _unpackable_problem(budget)

        objectives = problem.evaluate(_mid_configuration(problem))

        assert set(objectives) == set(HW_OBJECTIVES)
        assert budget.distinct_spent == 1

    def test_a_repeated_penalty_candidate_is_a_duplicate_too(self):
        budget = EvaluationBudget(limit=None)
        problem = _unpackable_problem(budget)
        configuration = _mid_configuration(problem)

        problem.evaluate(configuration)
        problem.evaluate(configuration)

        assert (budget.distinct_spent, budget.raw_calls) == (1, 2)

    def test_without_an_accountant_the_evaluation_is_byte_identical(self):
        with_budget = _hw_problem(EvaluationBudget(limit=1))
        without = _hw_problem(None)
        configuration = _mid_configuration(without)

        assert json.dumps(without.evaluate(configuration), sort_keys=True) == (
            json.dumps(with_budget.evaluate(configuration), sort_keys=True)
        )

    def test_an_exhausted_budget_never_refuses_an_evaluation(self):
        # The accountant METERS; stopping is the driver's decision at its own
        # boundary. A problem that started refusing would silently corrupt the
        # front of every search that overshot by one candidate.
        budget = EvaluationBudget(limit=1)
        problem = _hw_problem(budget)
        problem.evaluate(_mid_configuration(problem))
        assert budget.exhausted

        x = np.asarray(problem.xl) * 0.25 + np.asarray(problem.xu) * 0.75
        objectives = problem.evaluate(problem.decode(x))

        assert set(objectives) == set(HW_OBJECTIVES)
        assert budget.distinct_spent == 2


class TestEveryChannelThatSpendsWorkIsCharged:
    """A candidate screened out before ``evaluate`` still cost a full resolution.

    NSGA-II asks ``constraint_violation`` FIRST, and on this problem that walks
    the same model build → conversion → packing an evaluation walks. Work the
    accountant never hears about is a budget that bounds nothing: a search whose
    every offspring is rejected would seal a ledger claiming it spent zero.
    """

    def test_a_candidate_rejected_at_a_declared_floor_is_charged(self):
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget, floor=1.0)

        cv = problem.constraint_violation(_mid_configuration(problem))

        assert cv > 0.0, "the fixture must actually reject at the constraint channel"
        assert budget.distinct_spent == 1
        assert budget.raw_calls == 1

    def test_re_proposing_a_rejected_candidate_is_a_duplicate_like_any_other(self):
        # The DUPLICATE axis has to cover the constraint channel too, or a
        # driver that keeps re-proposing REJECTED candidates seals a duplicate
        # rate of 0.0 while a driver re-proposing ACCEPTED ones seals a
        # positive one — and a campaign comparing the two reads the wrong sign.
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget, floor=1.0)
        configuration = _mid_configuration(problem)

        for _ in range(5):
            assert problem.constraint_violation(configuration) > 0.0

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 5, "four asks a cache answered are four calls"
        # One channel, five proposals of one candidate: four of them repeats.
        assert (budget.identities_asked, budget.identities_reasked) == (5, 4)
        assert budget.duplicate_rate == pytest.approx(0.8)

    def test_a_structural_rejection_costs_what_every_declaration_only_check_costs(self):
        # ``validate_fn`` and ``constraint_fn`` are the SAME shape of cheap
        # predicate over the declaration — neither builds anything. The
        # equal-spend currency must not depend on which caller hook a search
        # space happens to wire its predicate into.
        structural = EvaluationBudget(limit=None)
        declared = EvaluationBudget(limit=None)
        by_validate_fn = _hw_problem(
            structural, validate_fn=lambda mc, pcfg, shape: False,
        )
        by_constraint_fn = _hw_problem(
            declared, constraint_fn=lambda mc, pcfg, shape: 1.0,
        )

        for problem in (by_validate_fn, by_constraint_fn):
            assert problem.constraint_violation(_mid_configuration(problem)) > 0.0

        assert (structural.distinct_spent, structural.raw_calls) == (0, 0)
        assert (declared.distinct_spent, declared.raw_calls) == (0, 0)

    def test_re_asking_a_structurally_rejected_candidate_stays_free(self):
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget, validate_fn=lambda mc, pcfg, shape: False)
        configuration = _mid_configuration(problem)

        for _ in range(4):
            problem.constraint_violation(configuration)

        assert (budget.distinct_spent, budget.raw_calls) == (0, 0)

    def test_re_evaluating_a_structurally_rejected_candidate_stays_free_too(self):
        # The EVALUATE channel's twin of the pin above, and the one place the
        # objective cache answers about an identity the run never spent
        # anything on: the first evaluate() records the refusal's penalty, the
        # second is served from that cache. Charging THAT hit would invent an
        # evaluation out of a candidate nothing was ever built for — and would
        # do it once per re-proposal, so a search rejecting everything
        # structurally would seal a spend it never made.
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget, validate_fn=lambda mc, pcfg, shape: False)
        configuration = _mid_configuration(problem)

        first = problem.evaluate(configuration)
        again = problem.evaluate(configuration)

        assert again == first, "the fixture must actually re-ask the same candidate"
        assert (budget.distinct_spent, budget.raw_calls) == (0, 0)

    def test_one_candidate_seen_by_both_channels_is_one_evaluation(self):
        # The evaluate channel re-asks the identity the constraint channel just
        # resolved. Charging per CHANNEL would price one candidate twice and put
        # a floor under every NSGA run's distinct spend.
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget)
        configuration = _mid_configuration(problem)

        assert problem.constraint_violation(configuration) == 0.0
        problem.evaluate(configuration)

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 2, "both channels asked; only one bought work"
        # ONE proposal, screened then scored: the evaluate channel's first look
        # is part of the round the constraint channel opened, not a repeat.
        assert (budget.identities_asked, budget.identities_reasked) == (1, 0)
        assert budget.duplicate_rate == 0.0

    def test_a_candidate_asked_again_through_both_channels_is_all_duplicates(self):
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget)
        configuration = _mid_configuration(problem)

        for _ in range(4):
            assert problem.constraint_violation(configuration) == 0.0
            problem.evaluate(configuration)

        assert budget.distinct_spent == 1
        assert budget.raw_calls == 8, "four proposals through two channels"
        # The comparable axis reads the PROPOSALS, not the calls: four rounds
        # of asking, three of them repeats. A call-based rate would say 7/8
        # here and 3/4 for the same behaviour from a one-channel driver.
        assert (budget.identities_asked, budget.identities_reasked) == (4, 3)
        assert budget.duplicate_rate == pytest.approx(0.75)

    def test_a_declared_constraint_that_resolves_nothing_costs_nothing(self):
        # ``constraint_fn`` is a cheap predicate over the DECLARATION: it never
        # builds a candidate, so charging it would price work nobody did.
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget, constraint_fn=lambda mc, pcfg, shape: 1.0)

        assert problem.constraint_violation(_mid_configuration(problem)) > 0.0
        assert (budget.distinct_spent, budget.raw_calls) == (0, 0)


class TestTheIntrospectionSeamSpendsWhatItResolves:
    """``candidate_layout`` walks the whole resolution with no cache of its own.

    It is the compilagent optimizer's introspection tool, so an agent that
    introspects instead of evaluating would otherwise spend an unbounded amount
    of real evaluator work entirely off-budget — the same hole the constraint
    channel had, left open for the driver TS3 must thread.
    """

    def test_laying_out_a_candidate_costs_a_distinct_evaluation(self):
        budget = EvaluationBudget(limit=2)
        problem = _hw_problem(budget)

        problem.candidate_layout(_mid_configuration(problem))

        assert (budget.distinct_spent, budget.raw_calls) == (1, 1)

    def test_an_introspecting_driver_exhausts_the_budget_it_spends(self):
        budget = EvaluationBudget(limit=2)
        problem = _hw_problem(budget)

        for fraction in (0.25, 0.5, 0.75):
            problem.candidate_layout(_configuration_at(problem, fraction))

        assert budget.distinct_spent == 3, "three chips were really built"
        assert budget.exhausted

    def test_re_introspecting_one_candidate_buys_no_new_evaluation(self):
        budget = EvaluationBudget(limit=None)
        problem = _hw_problem(budget)
        configuration = _mid_configuration(problem)

        for _ in range(3):
            problem.candidate_layout(configuration)

        assert (budget.distinct_spent, budget.raw_calls) == (1, 3)
        # One channel asking three times about one candidate: two re-asks.
        assert (budget.identities_asked, budget.identities_reasked) == (3, 2)


REJECTING_POP = 4
REJECTING_GENERATIONS = 3
REJECTING_LIMIT = 2


@pytest.fixture(scope="module")
def rejecting_run():
    """A real search whose every candidate is screened out at the floor."""
    budget = EvaluationBudget(limit=REJECTING_LIMIT)
    problem = _hw_problem(budget, floor=1.0)
    result = NSGA2Optimizer(
        pop_size=REJECTING_POP, generations=REJECTING_GENERATIONS,
        seed=0, verbose=False,
    ).optimize(problem, reporter=None)
    return budget, problem, result


class TestABudgetBoundsASearchThatRejectsEveryCandidate:
    """The measured case the optimizer's own R6 note names (72/72 offspring
    rejected on the ViT cell): if the screen is free, B buys nothing."""

    def test_every_candidate_really_was_screened_out(self, rejecting_run):
        _, problem, _ = rejecting_run
        assert problem.constraint_census()

    def test_the_run_stops_at_the_first_boundary(self, rejecting_run):
        _, _, result = rejecting_run
        assert [h["gen"] for h in result.history] == [1], (
            "an exhausted budget must bound a rejecting search too"
        )

    def test_the_ledger_seals_what_the_screen_actually_spent(self, rejecting_run):
        _, _, result = rejecting_run

        assert result.ledger is not None
        assert result.ledger.evaluations_distinct == REJECTING_POP
        assert result.ledger.budget_limit == REJECTING_LIMIT
        assert result.ledger.stopped_at_boundary is True


TWO_CHANNEL_POP = 6
TWO_CHANNEL_GENERATIONS = 3
#: Measured on this exact fixture (pop 6 x gen 3, seed 0): pymoo proposed 18
#: candidates of which 16 were new, so the driver re-proposed 2. It asked
#: through BOTH channels every time — 36 calls — which is why a call-based rate
#: read 0.5556 for a search that repeated one candidate in nine.
TWO_CHANNEL_DISTINCT = 16


@pytest.fixture(scope="module")
def two_channel_run():
    """A real NSGA-II search on the real hardware-mode problem, unmetered."""
    budget = EvaluationBudget(limit=None)
    problem = _hw_problem(budget)
    result = NSGA2Optimizer(
        pop_size=TWO_CHANNEL_POP, generations=TWO_CHANNEL_GENERATIONS,
        seed=0, verbose=False,
    ).optimize(problem, reporter=None)
    return budget, problem, result


class TestTheSealedRateIsTheDriversReProposalRate:
    """The axis a campaign ranks optimizers by must describe the SEARCH, not
    how many channels the driver happens to ask through."""

    def test_the_fixture_really_asks_through_two_channels_per_proposal(
        self, two_channel_run,
    ):
        _, _, result = two_channel_run
        ledger = result.ledger

        assert ledger is not None
        assert ledger.identities_asked == TWO_CHANNEL_POP * TWO_CHANNEL_GENERATIONS
        assert ledger.evaluations_raw == 2 * ledger.identities_asked, (
            "NSGA-II screens then scores: two calls per proposal"
        )

    def test_the_sealed_rate_is_the_re_proposal_rate_not_the_channel_floor(
        self, two_channel_run,
    ):
        _, _, result = two_channel_run
        ledger = result.ledger

        assert ledger is not None
        assert ledger.evaluations_distinct == TWO_CHANNEL_DISTINCT
        assert ledger.identities_reasked == (
            ledger.identities_asked - ledger.evaluations_distinct
        ), "every proposal after the first of an identity is a re-proposal"
        assert ledger.duplicate_rate == pytest.approx(2 / 18)
        assert ledger.duplicate_rate < 0.2, (
            "counted in CALLS this search sealed 0.5556 — a structural ~0.5 "
            "floor no single-channel driver can reach, so the two were never "
            "comparable"
        )

    def test_the_call_counts_are_still_sealed_as_facts(self, two_channel_run):
        # The identity axis REPLACES the rate, not the counts: an analysis that
        # wants calls per proposal must still be able to read them.
        budget, _, result = two_channel_run
        ledger = result.ledger

        assert ledger is not None
        assert ledger.evaluations_raw == 36
        assert ledger.evaluations_distinct == budget.distinct_spent


class _ToyProblem:
    """Two continuous variables, two objectives, and the same cache seam.

    The joint problem's accounting is pinned above on the real thing; this
    fixture exists so the DRIVER's boundary behaviour can be measured without
    a mapping run per candidate.
    """

    n_var = 2
    xl = [0.0, 0.0]
    xu = [1.0, 1.0]
    objectives = SPECS

    def __init__(self, budget: Any = None) -> None:
        self.evaluation_budget = budget
        self._cache: Dict[str, Dict[str, float]] = {}

    def decode(self, x) -> Dict[str, Any]:
        return {"a": float(x[0]), "b": float(x[1])}

    def validate(self, cfg) -> bool:
        return True

    def constraint_violation(self, cfg) -> float:
        return 0.0

    def evaluate(self, cfg) -> Dict[str, float]:
        key = json.dumps(cfg, sort_keys=True)
        cached = self._cache.get(key)
        charge_evaluation(
            self.evaluation_budget, key,
            hit=cached is not None, channel=EVALUATE_CHANNEL,
        )
        if cached is not None:
            return cached
        objectives = {
            "estimated_accuracy": 1.0 - abs(cfg["a"] - 0.7),
            "total_params": 1000.0 * cfg["b"] + 10.0,
        }
        self._cache[key] = objectives
        return objectives


POP_SIZE = 6
GENERATIONS = 4
BUDGET_LIMIT = 3


def _run_nsga(budget: Any):
    problem = _ToyProblem(budget)
    result = NSGA2Optimizer(
        pop_size=POP_SIZE, generations=GENERATIONS, seed=0, verbose=False,
    ).optimize(problem, reporter=None)
    return problem, result


@pytest.fixture(scope="module")
def boundary_run():
    """A budget smaller than one population: the first boundary is the stop."""
    budget = EvaluationBudget(limit=BUDGET_LIMIT)
    problem, result = _run_nsga(budget)
    return budget, problem, result


REQUIREMENTS = pathlib.Path(__file__).resolve().parents[3] / "requirements.txt"


class TestTheBoundaryMechanismsDependencyIsDeclared:
    def test_pymoo_carries_a_version_bound_in_the_dependency_declaration(self):
        # The generation-boundary stop depends on pymoo calling
        # ``termination.update(algorithm)`` BEFORE the callback, which is why
        # the callback must re-update it. That ordering is not API — a bare
        # ``pymoo`` requirement lets a release reorder it and buy one extra
        # generation past every budget boundary. The bound belongs in the
        # dependency declaration, not only in the test that would go red.
        lines = [
            line.strip() for line in REQUIREMENTS.read_text().splitlines()
            if re.match(r"^\s*pymoo\b", line)
        ]

        assert len(lines) == 1, f"expected exactly one pymoo requirement, got {lines}"
        assert re.search(r"(==|>=|~=|<)", lines[0]), (
            f"pymoo must declare a version bound; requirements.txt says {lines[0]!r}"
        )


class TestNsga2StopsAtTheFirstBoundary:
    def test_the_search_stops_at_the_generation_boundary(self, boundary_run):
        _, _, result = boundary_run
        generations = {c.metadata["generation"] for c in result.all_candidates}

        assert generations == {1}, "an exhausted budget must not buy generation 2"
        assert [h["gen"] for h in result.history] == [1]

    def test_the_stop_is_a_whole_generation_not_the_bare_limit(self, boundary_run):
        # Boundary-stop semantics: the generation in flight finishes, and the
        # ledger records the exact overshoot instead of hiding it.
        budget, _, _ = boundary_run
        assert budget.distinct_spent == POP_SIZE > BUDGET_LIMIT

    def test_the_sealed_spend_is_the_accountants_exact_count(self, boundary_run):
        budget, _, result = boundary_run
        ledger = result.ledger

        assert ledger is not None
        assert ledger.evaluations_distinct == budget.distinct_spent
        assert ledger.budget_limit == BUDGET_LIMIT
        assert ledger.stopped_at_boundary is True

    def test_the_ledger_times_the_search(self, boundary_run):
        _, _, result = boundary_run
        assert result.ledger is not None
        assert result.ledger.wall_s > 0.0

    def test_the_search_still_hands_over_a_winner(self, boundary_run):
        _, _, result = boundary_run
        assert result.best.configuration
        assert result.pareto_front


class TestTheLedgerCoversTheSearchItself:
    """Every fact in one ledger describes ONE interval: the search."""

    def test_the_counts_stop_where_the_clock_stops(self, boundary_run):
        # ``wall_s`` is measured around ``minimize``; counts sealed beside it
        # must cover the same interval, or a campaign comparing duplicate rates
        # would be comparing each driver's post-search bookkeeping.
        _, _, result = boundary_run

        assert result.ledger is not None
        assert result.ledger.evaluations_raw == POP_SIZE
        assert result.ledger.evaluations_distinct == POP_SIZE

    def test_a_single_channel_driver_asking_each_identity_once_seals_zero(
        self, boundary_run,
    ):
        # This problem answers through ONE channel and the search proposed no
        # candidate twice, so the comparable axis reads exactly 0.0 — the
        # baseline the multi-channel problem below must agree with.
        _, _, result = boundary_run

        assert result.ledger is not None
        assert result.ledger.identities_asked == POP_SIZE
        assert result.ledger.identities_reasked == 0
        assert result.ledger.duplicate_rate == 0.0

    def test_the_front_re_read_is_the_accountants_business_not_the_ledgers(
        self, boundary_run,
    ):
        # The optimizer re-reads the front after the boundary: real calls the
        # accountant keeps counting (they are cache hits, so they buy nothing),
        # but not part of what the SEARCH spent.
        budget, _, result = boundary_run

        assert result.pareto_front
        assert budget.raw_calls == POP_SIZE + len(result.pareto_front)
        assert budget.distinct_spent == POP_SIZE


class TestTheBoundaryFlagMeansTheBudgetCutTheRunShort:
    def test_a_budget_that_denied_a_generation_is_a_boundary_stop(self, boundary_run):
        _, _, result = boundary_run
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is True

    def test_a_run_its_own_generations_ended_is_not_a_boundary_stop(self):
        # The budget runs out ON the last generation, which the termination was
        # going to end anyway. A campaign separating budget-bound from
        # generation-bound runs must not read this as the budget's doing.
        limit = POP_SIZE * (GENERATIONS - 1) + 1
        budget = EvaluationBudget(limit=limit)
        _, result = _run_nsga(budget)

        assert [h["gen"] for h in result.history] == list(range(1, GENERATIONS + 1))
        assert budget.exhausted, "the fixture must exhaust at the LAST boundary"
        assert result.ledger is not None
        assert result.ledger.evaluations_distinct >= limit
        assert result.ledger.stopped_at_boundary is False


class TestAnUnspentBudgetChangesNothing:
    def test_a_metering_budget_runs_every_generation(self):
        budget = EvaluationBudget(limit=None)
        _, result = _run_nsga(budget)

        assert [h["gen"] for h in result.history] == list(range(1, GENERATIONS + 1))
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is False
        assert result.ledger.budget_limit is None
        assert result.ledger.evaluations_distinct == budget.distinct_spent

    def test_a_run_without_an_accountant_seals_no_ledger(self):
        _, result = _run_nsga(None)
        assert result.ledger is None

    def test_a_run_without_an_accountant_is_byte_identical(self):
        # The A/B the additive claim rests on: absent budget == today's search,
        # today's artifact, down to the bytes.
        _, plain = _run_nsga(None)
        _, metered = _run_nsga(EvaluationBudget(limit=None))

        plain_payload = search_result_to_jsonable(plain)
        metered_payload = dict(search_result_to_jsonable(metered))
        metered_payload.pop("ledger")

        assert "ledger" not in plain_payload
        assert json.dumps(plain_payload, sort_keys=True) == json.dumps(
            metered_payload, sort_keys=True,
        )


class TestTheCallbackIsNotTelemetry:
    def test_a_dead_reporter_cannot_cancel_the_boundary_stop(self):
        # Reporting rides best_effort; the budget stop must not, or a broken
        # monitor would silently buy an unbounded number of evaluations.
        def exploding(name: str, value: Any, step: Any = None) -> None:
            raise RuntimeError("monitor is gone")

        problem = _ToyProblem(EvaluationBudget(limit=BUDGET_LIMIT))
        result = NSGA2Optimizer(
            pop_size=POP_SIZE, generations=GENERATIONS, seed=0, verbose=False,
        ).optimize(problem, reporter=exploding)

        generations: List[int] = [c.metadata["generation"] for c in result.all_candidates]
        assert set(generations) == {1}


@pytest.fixture
def boundary_asks(monkeypatch):
    """Every answer the ONE boundary gave during a run, in order."""
    asks: List[bool] = []
    original = BoundaryStop.should_stop

    def spy(self: BoundaryStop) -> bool:
        answer = original(self)
        asks.append(answer)
        return answer

    monkeypatch.setattr(BoundaryStop, "should_stop", spy)
    return asks


class TestEveryDriverStopsThroughTheOneBoundary:
    """[TS3] `BoundaryStop` is the LAW, not one family's helper.

    ``stopped_at_boundary`` is the flag a campaign separates budget-bound runs
    from run-bound ones by, so it must mean one thing across every backend it
    compares. A driver that decides it inline is a second definition — it
    already read the accountant, so nothing warns when the two drift. Every
    driver that can be cut short therefore asks the same object, and asks it
    only where the run would otherwise CONTINUE.
    """

    def test_the_classical_driver_asks_it(self, boundary_asks):
        budget = EvaluationBudget(limit=BUDGET_LIMIT)
        _, result = _run_nsga(budget)

        assert True in boundary_asks, "NSGA-II decided its own stop"
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is True

    def test_the_sampling_driver_asks_it(self, boundary_asks):
        budget = EvaluationBudget(limit=BUDGET_LIMIT)
        problem = _ToyProblem(budget)
        result = SamplingOptimizer(
            strategy=RandomStrategy(samples=POP_SIZE * GENERATIONS),
            pop_size=POP_SIZE, seed=0,
        ).optimize(problem, reporter=None)

        assert True in boundary_asks, "the sampler decided its own stop"
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is True

    def test_the_boundary_is_never_asked_where_nothing_would_continue(
        self, boundary_asks,
    ):
        # The budget runs out ON the last generation. Asking there would seal
        # the flag for a run its own termination was ending anyway.
        limit = POP_SIZE * (GENERATIONS - 1) + 1
        _, result = _run_nsga(EvaluationBudget(limit=limit))

        assert True not in boundary_asks
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is False
