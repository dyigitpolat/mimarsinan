"""[TS2] The sampling backends: one driver, three streams, one event vocabulary.

A campaign needs baselines that are not evolution — a uniform draw, a
low-discrepancy draw, and the whole grid — and it needs them to be the SAME
kind of object as NSGA-II: the same optimizer contract, the same live-event
frames, the same TS1 accountant doing the counting. So there is exactly one
driver here and three streams of encoded vectors feeding it, which is why a
strategy is a protocol rather than a subclass hierarchy.

The grid is the ENCODING's own enumeration, because only the encoding knows
what its coordinates mean: an arch index range, a core dimension snapped to the
declared granularity, an integer count, an option axis' choices. It refuses
LOUDLY where enumeration is not defined (a continuous axis) or would be
dishonest (a product past the caller's declared cap) — a truncated "exhaustive"
search is a baseline that quietly stops being one.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import pytest
import torch

from mimarsinan.gui.wizard.schema import get_wizard_nas_schema
from mimarsinan.common.dependency_manifest import declared_specifier
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    OPTIMIZER_BUILDERS,
    OptimizerType,
    create_optimizer,
    make_platform_resolver,
)
from mimarsinan.search.optimizers.budget import EvaluationBudget
from mimarsinan.search.optimizers.catalog import OPTIMIZER_CHOICES, OPTIMIZER_IDS
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.optimizers.sampling_optimizer import (
    DEFAULT_GRID_CAP,
    GridStrategy,
    RandomStrategy,
    SamplingOptimizer,
    SobolStrategy,
)
from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.problems.joint.types import EVALUATE_CHANNEL
from mimarsinan.search.optimizers.budget import charge_evaluation
from mimarsinan.search.results import (
    ObjectiveSpec,
    nondominated_front,
    order_by_minimax_rank,
)
from mimarsinan.search.search_space_description import CORE_DIM_GRANULARITY

SPECS = (
    ObjectiveSpec("estimated_accuracy", "max"),
    ObjectiveSpec("total_params", "min"),
)
HW_OBJECTIVES = ["total_param_capacity", "param_utilization_pct", "fragmentation_pct"]

#: The tiny declared space every grid pin is computed by hand from. Bounds sit
#: ON the core-dimension granularity, so a snapped dimension holds exactly
#: ``(hi - lo) / G + 1`` values and the product below is arithmetic, not a
#: re-implementation of the enumeration under test.
AXONS_BOUNDS = (64, 64 + 2 * CORE_DIM_GRANULARITY)
NEURONS_BOUNDS = (64, 64 + CORE_DIM_GRANULARITY)
COUNT_BOUNDS = (2, 4)
AXONS_VALUES = 3
NEURONS_VALUES = 2
COUNT_VALUES = 3
HW_GRID = AXONS_VALUES * NEURONS_VALUES * COUNT_VALUES


def _pipeline_config() -> Dict[str, Any]:
    return {
        "device": "cpu",
        "input_shape": (1, 8, 8),
        "num_classes": 4,
        "target_tq": 4,
        "weight_bits": 4,
        "lr": 0.001,
        "allow_scheduling": True,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "model_config": {"mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU"},
    }


def _problem(
    *,
    option_axes=(),
    arch_options=(),
    search_mode: str = "hardware",
    budget=None,
    axons_bounds=AXONS_BOUNDS,
    neurons_bounds=NEURONS_BOUNDS,
    count_bounds=COUNT_BOUNDS,
) -> JointArchHwProblem:
    """The real joint problem over the tiny declared space above."""
    cfg = _pipeline_config()
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(cfg["input_shape"]),
        num_classes=cfg["num_classes"],
        target_tq=cfg["target_tq"],
        lr=cfg["lr"],
        search_mode=search_mode,
        builder_factory=SimpleMLPBuilder,
        arch_options=arch_options,
        model_config_assembler=lambda raw: dict(raw),
        fixed_model_config=dict(cfg["model_config"]),
        platform_resolver=make_platform_resolver(cfg),
        active_objective_names=HW_OBJECTIVES,
        num_core_types=1,
        core_axons_bounds=axons_bounds,
        core_neurons_bounds=neurons_bounds,
        core_count_bounds=count_bounds,
        option_axes=option_axes,
        evaluation_budget=budget,
    )


class TestTheGridIsTheEncodingsOwnEnumeration:
    """Only the encoding knows what its coordinates mean, so it enumerates them."""

    def test_the_cardinality_is_the_closed_form_product(self):
        # Three snapped axon values x two neuron values x three counts. A
        # dimension the enumeration forgets divides this number; a dimension it
        # invents multiplies it.
        vectors = _problem().grid_vectors(cap=DEFAULT_GRID_CAP)

        assert len(vectors) == HW_GRID == 18

    def test_option_axes_multiply_the_grid_by_their_own_value_sets(self):
        # A choice axis contributes its choices, an integral numeric axis its
        # whole inclusive range — both index-coded exactly as ``decode`` reads
        # them.
        axes = build_option_axes(
            {"encoding_layer_placement": None, "weight_bits": {"bounds": [2, 4]}},
        )
        vectors = _problem(option_axes=axes).grid_vectors(cap=DEFAULT_GRID_CAP)

        assert len(vectors) == HW_GRID * 2 * 3

    def test_the_model_dimensions_are_enumerated_too(self):
        arch_options = (("mlp_width_1", [8, 16, 32]), ("base_activation", ["ReLU", "LeakyReLU"]))
        vectors = _problem(
            arch_options=arch_options, search_mode="joint",
        ).grid_vectors(cap=DEFAULT_GRID_CAP)

        assert len(vectors) == 3 * 2 * HW_GRID

    def test_every_vector_has_the_encodings_shape(self):
        problem = _problem()
        vectors = problem.grid_vectors(cap=DEFAULT_GRID_CAP)

        assert all(v.shape == (problem.n_var,) for v in vectors)

    def test_every_vector_lies_inside_the_declared_box(self):
        problem = _problem(option_axes=build_option_axes(["encoding_layer_placement"]))
        lo, hi = problem.xl, problem.xu

        for vector in problem.grid_vectors(cap=DEFAULT_GRID_CAP):
            assert np.all(vector >= lo) and np.all(vector <= hi)

    def test_the_grid_visits_every_point_exactly_once(self):
        vectors = _problem().grid_vectors(cap=DEFAULT_GRID_CAP)

        assert len({tuple(v.tolist()) for v in vectors}) == len(vectors)

    def test_the_enumerated_points_decode_to_distinct_candidates(self):
        # The currency the TS1 accountant spends is the decoded IDENTITY, so an
        # enumeration whose points collapse onto one another would buy the same
        # candidate twice — the exact failure snapping a dimension can cause.
        problem = _problem()
        decoded = {
            json.dumps(problem.decode(v)["platform_constraints"]["cores"], sort_keys=True)
            for v in problem.grid_vectors(cap=DEFAULT_GRID_CAP)
        }

        assert len(decoded) == HW_GRID

    def test_the_snapped_dimension_values_are_the_declared_grid_lines(self):
        problem = _problem()
        axons = {
            problem.decode(v)["platform_constraints"]["cores"][0]["max_axons"]
            for v in problem.grid_vectors(cap=DEFAULT_GRID_CAP)
        }

        assert axons == {
            AXONS_BOUNDS[0] + i * CORE_DIM_GRANULARITY for i in range(AXONS_VALUES)
        }

    def test_a_continuous_option_axis_is_refused_by_its_own_name(self):
        axes = build_option_axes({"activity_factor": {"bounds": [0.0, 0.5]}})
        problem = _problem(option_axes=axes)

        with pytest.raises(ValueError, match="activity_factor"):
            problem.grid_vectors(cap=DEFAULT_GRID_CAP)

    def test_a_grid_past_the_declared_cap_is_refused_naming_both_numbers(self):
        cap = HW_GRID - 1

        with pytest.raises(ValueError) as excinfo:
            _problem().grid_vectors(cap=cap)

        message = str(excinfo.value)
        assert str(HW_GRID) in message, message
        assert str(cap) in message, message

    def test_the_cap_refusal_never_silently_truncates(self):
        # The whole point of the refusal: a caller must not be able to mistake
        # a shortened enumeration for the exhaustive one.
        with pytest.raises(ValueError):
            _problem().grid_vectors(cap=1)

    def test_a_cap_at_the_grids_own_size_is_not_a_refusal(self):
        assert len(_problem().grid_vectors(cap=HW_GRID)) == HW_GRID


class _BoxProblem:
    """A bare encoded box: what a sampler needs and nothing else."""

    n_var = 3
    xl = [0.0, -2.0, 5.0]
    xu = [1.0, 2.0, 6.0]
    objectives = SPECS

    def decode(self, x) -> Dict[str, Any]:
        return {"x": [float(v) for v in x]}


SAMPLES = 7


def _drawn(strategy, seed: int = 0) -> List[np.ndarray]:
    return list(strategy.vectors(_BoxProblem(), np.random.default_rng(seed)))


@pytest.mark.parametrize(
    "strategy", [RandomStrategy(samples=SAMPLES), SobolStrategy(samples=SAMPLES)],
    ids=["random", "sobol"],
)
class TestTheSamplersDrawInsideTheDeclaredBox:
    def test_the_stream_is_exactly_as_long_as_declared(self, strategy):
        assert len(_drawn(strategy)) == SAMPLES

    def test_every_vector_has_the_problems_shape(self, strategy):
        assert all(v.shape == (_BoxProblem.n_var,) for v in _drawn(strategy))

    def test_no_vector_leaves_the_box(self, strategy):
        # A sampler drawing in [0, 1] and forgetting to scale would put every
        # candidate at one corner of a search space it never explored.
        lo = np.asarray(_BoxProblem.xl, dtype=float)
        hi = np.asarray(_BoxProblem.xu, dtype=float)

        for vector in _drawn(strategy):
            assert np.all(vector >= lo) and np.all(vector <= hi)

    def test_the_stream_actually_moves_through_the_box(self, strategy):
        drawn = np.array(_drawn(strategy))

        assert np.all(drawn.max(axis=0) > drawn.min(axis=0)), "a constant stream is not a sample"

    def test_the_same_seed_redraws_the_same_stream(self, strategy):
        assert np.allclose(np.array(_drawn(strategy, 5)), np.array(_drawn(strategy, 5)))

    def test_a_different_seed_draws_a_different_stream(self, strategy):
        assert not np.allclose(np.array(_drawn(strategy, 5)), np.array(_drawn(strategy, 6)))


class TestTheSamplersDependencyIsDeclared:
    def test_scipy_carries_the_bound_the_sobol_seam_needs(self):
        # ``qmc.Sobol`` takes its generator as ``rng=`` only from scipy 1.15;
        # on an older release the seeded draw is a TypeError, so the floor
        # belongs in the dependency declaration, not only in a red test.
        specifier = declared_specifier("scipy")

        assert specifier is not None, "pyproject.toml declares no scipy requirement"
        assert re.search(r"(==|>=|~=)", specifier), specifier


class TestSobolIsALowDiscrepancySequence:
    def test_the_draw_is_scrambled_not_the_bare_lattice(self):
        # An unscrambled Sobol' sequence starts at the box's lower corner, and
        # a run whose first candidate is always the same corner is not a
        # randomized baseline.
        first = _drawn(SobolStrategy(samples=4), seed=3)[0]

        assert not np.allclose(first, np.asarray(_BoxProblem.xl, dtype=float))

    def test_it_spreads_more_evenly_than_a_uniform_draw(self, ):
        # The reason a campaign asks for Sobol' at all: at equal spend it
        # covers the box better. Measured on the same seed and count via the
        # largest gap between consecutive points on the first axis.
        def worst_gap(vectors) -> float:
            values = sorted(float(v[0]) for v in vectors)
            edges = [0.0] + values + [1.0]
            return max(b - a for a, b in zip(edges, edges[1:]))

        sobol = worst_gap(_drawn(SobolStrategy(samples=16), seed=1))
        uniform = worst_gap(_drawn(RandomStrategy(samples=16), seed=1))

        assert sobol < uniform


class _ToyGridProblem:
    """A 3x3 integer grid whose front can be read off by hand.

    ``estimated_accuracy`` is ``a`` and ``total_params`` is ``10a + b``, so for
    every ``a`` the ``b = 0`` point dominates its column and nothing dominates
    across columns: the front is exactly the three ``b = 0`` points. The
    evaluate seam charges the TS1 accountant the way the real problem does, so
    the driver's spend can be measured without a mapping run per candidate.
    """

    n_var = 2
    xl = [0.0, 0.0]
    xu = [2.0, 2.0]
    objectives = SPECS
    SIDE = 3

    def __init__(self, budget: Any = None) -> None:
        self.evaluation_budget = budget
        self._cache: Dict[str, Dict[str, float]] = {}
        self.evaluated: List[Dict[str, int]] = []

    def grid_vectors(self, *, cap: int) -> List[np.ndarray]:
        points = [
            np.array([a, b], dtype=float)
            for a in range(self.SIDE)
            for b in range(self.SIDE)
        ]
        if len(points) > cap:
            raise ValueError(f"grid of {len(points)} points exceeds the cap {cap}")
        return points

    def decode(self, x) -> Dict[str, Any]:
        return {"a": int(round(float(x[0]))), "b": int(round(float(x[1])))}

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
        self.evaluated.append(dict(cfg))
        objectives = {
            "estimated_accuracy": float(cfg["a"]),
            "total_params": float(10 * cfg["a"] + cfg["b"]),
        }
        self._cache[key] = objectives
        return objectives


TOY_POINTS = _ToyGridProblem.SIDE ** 2
TOY_POP = 3
TOY_BATCHES = TOY_POINTS // TOY_POP
EXHAUSTIVE_FRONT = [{"a": 0, "b": 0}, {"a": 1, "b": 0}, {"a": 2, "b": 0}]


@dataclass
class _FixedStrategy:
    """A hand-written plan — the protocol is a shape, not a base class."""

    plan: Sequence[np.ndarray]

    def vectors(self, problem, rng) -> Iterator[np.ndarray]:
        return iter(self.plan)


def _sampling_run(strategy=None, budget=None, pop_size: int = TOY_POP):
    problem = _ToyGridProblem(budget)
    optimizer = SamplingOptimizer(
        strategy=strategy or GridStrategy(cap=DEFAULT_GRID_CAP),
        pop_size=pop_size, seed=0,
    )
    return problem, optimizer


@pytest.fixture(scope="module")
def run():
    """One exhaustive run over the toy grid — every pin below reads it."""
    problem, optimizer = _sampling_run()
    return problem, optimizer.optimize(problem, reporter=None)


class TestTheDriverReturnsTheExhaustiveFront:
    def test_every_grid_point_is_evaluated(self, run):
        problem, result = run

        assert len(problem.evaluated) == TOY_POINTS
        assert len(result.all_candidates) == TOY_POINTS

    def test_the_front_is_the_hand_computed_one(self, run):
        _, result = run
        front = sorted(
            (c.configuration for c in result.pareto_front), key=lambda c: c["a"],
        )

        assert front == EXHAUSTIVE_FRONT

    def test_no_dominated_candidate_reaches_the_front(self, run):
        _, result = run
        rows = [c.objectives for c in result.all_candidates]
        front_rows = [c.objectives for c in result.pareto_front]

        assert sorted(nondominated_front(rows, SPECS)) == sorted(
            i for i, row in enumerate(rows) if row in front_rows
        )

    def test_the_winner_is_the_minimax_pick_over_that_front(self, run):
        _, result = run
        order = order_by_minimax_rank([c.objectives for c in result.pareto_front], SPECS)

        assert result.best.configuration == result.pareto_front[order[0]].configuration
        # Hand-check: ranks are (2, 2) for a=1 against (3, 1) and (1, 3).
        assert result.best.configuration == {"a": 1, "b": 0}

    def test_the_front_agrees_with_what_an_evolutionary_run_can_reach(self, run):
        # The baseline's whole purpose: on a space small enough to enumerate,
        # the exhaustive front is the ground truth NSGA-II is measured against.
        _, exhaustive = run
        evolved = NSGA2Optimizer(
            pop_size=6, generations=4, seed=0, verbose=False,
        ).optimize(_ToyGridProblem(), reporter=None)

        best_params = min(c.objectives["total_params"] for c in evolved.pareto_front)
        assert best_params >= min(
            c.objectives["total_params"] for c in exhaustive.pareto_front
        )

    def test_every_candidate_carries_its_batch_and_its_vector(self, run):
        _, result = run
        generations = [c.metadata["generation"] for c in result.all_candidates]

        assert generations == sorted(generations)
        assert set(generations) == set(range(1, TOY_BATCHES + 1))
        assert all(len(c.metadata["x"]) == _ToyGridProblem.n_var for c in result.all_candidates)

    def test_front_membership_is_marked_on_the_candidates(self, run):
        _, result = run
        marked = [c for c in result.all_candidates if c.metadata["is_pareto"]]

        assert len(marked) == len(result.pareto_front)

    def test_the_history_counts_the_batches_one_based(self, run):
        _, result = run

        assert [row["gen"] for row in result.history] == list(range(1, TOY_BATCHES + 1))


class _CapturingReporter:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def __call__(self, name: str, value: Any, step: Any = None) -> None:
        self.calls.append((name, value))

    def events(self) -> List[Dict[str, Any]]:
        return [json.loads(v) for name, v in self.calls if name == "search_event"]


def _of_type(events, kind):
    return [e for e in events if e["type"] == kind]


@pytest.fixture(scope="module")
def events():
    reporter = _CapturingReporter()
    problem, optimizer = _sampling_run()
    optimizer.optimize(problem, reporter=reporter)
    return reporter.events()


class TestTheDriverSpeaksTheSharedEventVocabulary:
    """All six optimizers feed ONE live panel; a sampler that stayed silent
    would leave it blank exactly as the classical backend once did."""

    def test_the_frames_are_the_panels_own_vocabulary(self, events):
        assert {e["type"] for e in events} == {
            "generation_start", "candidates_generated",
            "generation_complete", "search_complete",
        }

    def test_emission_is_per_batch_not_per_candidate(self, events):
        assert len(events) == 3 * TOY_BATCHES + 1

    def test_each_batch_opens_and_closes_once_in_order(self, events):
        for kind in ("generation_start", "generation_complete"):
            assert [e["gen"] for e in _of_type(events, kind)] == list(
                range(1, TOY_BATCHES + 1)
            )

    def test_the_batch_carries_the_runs_declared_shape(self, events):
        for ev in _of_type(events, "generation_start"):
            assert ev["total_gens"] == TOY_BATCHES
            assert ev["pop_size"] == TOY_POP
            assert ev["objectives"] == [
                {"name": "estimated_accuracy", "goal": "max"},
                {"name": "total_params", "goal": "min"},
            ]

    def test_a_sampled_batch_never_claims_to_have_evolved(self, events):
        phases = [e["phase"] for e in _of_type(events, "generation_start")]

        assert phases[0] == "initial"
        assert set(phases[1:]) == {"sampling"}

    def test_every_batch_reports_the_candidates_it_drew(self, events):
        counts = [e["count"] for e in _of_type(events, "candidates_generated")]

        assert counts == [TOY_POP] * TOY_BATCHES

    def test_the_verdict_counts_account_for_the_whole_batch(self, events):
        for ev in _of_type(events, "generation_complete"):
            assert ev["valid_count"] + ev["failed_count"] == TOY_POP
            assert ev["failed_count"] == 0

    def test_the_reported_front_is_in_user_space_incumbent_first(self, events):
        for ev in _of_type(events, "generation_complete"):
            assert ev["pareto_size"] >= 1
            rows = ev["pareto_front"]
            assert all(set(row) == {"estimated_accuracy", "total_params"} for row in rows)
            if ev["pareto_size"] == len(rows):
                assert order_by_minimax_rank(rows, SPECS) == list(range(len(rows)))

    def test_the_run_totals_are_sealed_once(self, events):
        done = _of_type(events, "search_complete")

        assert len(done) == 1
        assert done[0]["total_valid"] == TOY_POINTS
        assert done[0]["final_pareto_size"] == len(EXHAUSTIVE_FRONT)

    def test_a_broken_reporter_cannot_kill_the_search(self):
        def exploding(name, value, step=None):
            raise RuntimeError("monitor is gone")

        problem, optimizer = _sampling_run()
        result = optimizer.optimize(problem, reporter=exploding)

        assert result.best.configuration, "telemetry failure must not lose the winner"


class TestTheAccountantMetersTheStream:
    def test_duplicate_vectors_spend_no_distinct_budget(self):
        # The identity law, seen from the driver: three vectors that decode to
        # ONE candidate cost one evaluation, whatever the stream believed it
        # was proposing. A driver counting its own draws would bill three.
        repeated = np.array([1.0, 1.0])
        plan = [repeated, repeated, repeated, np.array([2.0, 0.0])]
        budget = EvaluationBudget(limit=None)
        problem, optimizer = _sampling_run(
            strategy=_FixedStrategy(plan), budget=budget, pop_size=len(plan),
        )

        result = optimizer.optimize(problem, reporter=None)

        assert budget.distinct_spent == 2, "two candidates were actually built"
        assert budget.raw_calls == len(plan), "every draw really did ask"
        assert len(problem.evaluated) == 2
        assert result.ledger is not None
        assert result.ledger.evaluations_distinct == 2
        assert result.ledger.evaluations_raw == len(plan)

    def test_an_exhausted_budget_stops_the_run_at_the_batch_boundary(self):
        budget = EvaluationBudget(limit=TOY_POP - 1)
        problem, optimizer = _sampling_run(budget=budget)

        result = optimizer.optimize(problem, reporter=None)

        assert len(problem.evaluated) == TOY_POP, "the batch in flight finishes"
        assert {c.metadata["generation"] for c in result.all_candidates} == {1}
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is True
        assert result.ledger.evaluations_distinct == TOY_POP
        assert result.ledger.budget_limit == TOY_POP - 1

    def test_a_stopped_run_still_hands_over_a_winner(self):
        budget = EvaluationBudget(limit=1)
        problem, optimizer = _sampling_run(budget=budget)

        result = optimizer.optimize(problem, reporter=None)

        assert result.best.configuration
        assert result.pareto_front

    def test_a_run_its_own_stream_ended_is_not_a_boundary_stop(self):
        # The budget is spent exactly as the last batch closes; nothing was
        # denied, so a campaign separating budget-bound runs from stream-bound
        # ones must not read this as the budget's doing.
        budget = EvaluationBudget(limit=TOY_POINTS)
        problem, optimizer = _sampling_run(budget=budget)

        result = optimizer.optimize(problem, reporter=None)

        assert budget.exhausted
        assert len(problem.evaluated) == TOY_POINTS
        assert result.ledger is not None
        assert result.ledger.stopped_at_boundary is False

    def test_an_unmetered_run_seals_no_ledger(self):
        problem, optimizer = _sampling_run()

        assert optimizer.optimize(problem, reporter=None).ledger is None

    def test_the_ledger_times_the_search(self):
        problem, optimizer = _sampling_run(budget=EvaluationBudget(limit=None))

        result = optimizer.optimize(problem, reporter=None)

        assert result.ledger is not None
        assert result.ledger.wall_s > 0.0

    def test_a_random_draw_is_metered_the_same_way(self):
        budget = EvaluationBudget(limit=None)
        problem = _ToyGridProblem(budget)
        SamplingOptimizer(
            strategy=RandomStrategy(samples=8), pop_size=4, seed=0,
        ).optimize(problem, reporter=None)

        assert budget.raw_calls == 8
        assert budget.distinct_spent == len(problem.evaluated)


class TestTheDriverRefusesWhatItCannotStream:
    def test_a_problem_that_cannot_enumerate_itself_is_named(self):
        problem = _BoxProblem()
        optimizer = SamplingOptimizer(strategy=GridStrategy(cap=8), pop_size=2, seed=0)

        with pytest.raises(TypeError, match="grid_vectors"):
            optimizer.optimize(problem, reporter=None)  # type: ignore[arg-type]

    def test_a_problem_without_objectives_is_refused(self):
        class _Objectiveless(_ToyGridProblem):
            objectives = ()

        problem, optimizer = _sampling_run()

        with pytest.raises(ValueError, match="objectives"):
            optimizer.optimize(_Objectiveless(), reporter=None)

    def test_a_non_positive_sample_count_is_refused(self):
        with pytest.raises(ValueError, match="samples"):
            RandomStrategy(samples=0)

    def test_a_non_positive_population_is_refused(self):
        with pytest.raises(ValueError, match="pop_size"):
            SamplingOptimizer(strategy=RandomStrategy(samples=4), pop_size=0)


class TestNondominatedFront:
    """The one pure front rule the sampling drivers select with."""

    ROWS = [
        {"estimated_accuracy": 0.9, "total_params": 100.0},
        {"estimated_accuracy": 0.8, "total_params": 200.0},
        {"estimated_accuracy": 0.5, "total_params": 50.0},
    ]

    def test_a_dominated_row_is_excluded(self):
        # Row 1 is worse on BOTH axes than row 0 — the mutant this pins.
        assert nondominated_front(self.ROWS, SPECS) == [0, 2]

    def test_indices_come_back_in_input_order(self):
        assert nondominated_front(self.ROWS, SPECS) == sorted(
            nondominated_front(self.ROWS, SPECS)
        )

    def test_goal_direction_is_honored(self):
        # Read as if accuracy were minimized, row 2 would dominate row 0.
        flipped = (ObjectiveSpec("estimated_accuracy", "min"), SPECS[1])

        assert nondominated_front(self.ROWS, flipped) == [2]

    def test_equal_rows_both_survive(self):
        rows = [self.ROWS[0], dict(self.ROWS[0])]

        assert nondominated_front(rows, SPECS) == [0, 1]

    def test_an_empty_set_has_an_empty_front(self):
        assert nondominated_front([], SPECS) == []

    def test_a_single_row_is_its_own_front(self):
        assert nondominated_front(self.ROWS[:1], SPECS) == [0]

    def test_a_missing_axis_reads_the_way_the_rest_of_the_module_reads_it(self):
        # results.py has ONE convention for an unreported axis — 0.0, the same
        # default the minimax ranking uses — so the front and the incumbent
        # can never disagree about what a row is worth.
        assert nondominated_front([{"estimated_accuracy": 1.0}, {}], SPECS[:1]) == [0]


#: A space with exactly two candidates: one core geometry, two core counts. The
#: real problem builds, converts and packs a model per point, so the pin below
#: buys the whole seam — encoding, driver, accountant — for two mapping runs.
TWO_POINT_COUNTS = (2, 3)


@pytest.fixture(scope="module")
def real_exhaustive_run():
    """The exhaustive driver over the REAL joint problem, metered."""
    budget = EvaluationBudget(limit=None)
    problem = _problem(
        budget=budget, axons_bounds=(64, 64), neurons_bounds=(64, 64),
        count_bounds=TWO_POINT_COUNTS,
    )
    result = SamplingOptimizer(
        strategy=GridStrategy(cap=DEFAULT_GRID_CAP), pop_size=2, seed=0,
    ).optimize(problem, reporter=None)
    return budget, problem, result


class TestTheDriverRunsTheRealEncoding:
    """The toy problems above measure the driver; this measures the SEAM —
    the encoding's grid decoded into real chips, scored by the real evaluation,
    counted by the real accountant."""

    def test_the_grid_is_the_two_candidates_the_bounds_declare(
        self, real_exhaustive_run,
    ):
        _, _, result = real_exhaustive_run

        assert len(result.all_candidates) == 2

    def test_each_point_becomes_a_distinct_resolved_chip(self, real_exhaustive_run):
        _, _, result = real_exhaustive_run
        counts = [
            c.configuration["platform_constraints"]["cores"][0]["count"]
            for c in result.all_candidates
        ]

        assert sorted(counts) == list(TWO_POINT_COUNTS)

    def test_the_ledger_seals_one_distinct_evaluation_per_point(
        self, real_exhaustive_run,
    ):
        budget, _, result = real_exhaustive_run

        assert budget.distinct_spent == 2
        assert result.ledger is not None
        assert result.ledger.evaluations_distinct == 2
        assert result.ledger.stopped_at_boundary is False

    def test_the_run_scores_the_declared_objectives_and_picks_a_winner(
        self, real_exhaustive_run,
    ):
        _, _, result = real_exhaustive_run

        assert set(result.best.objectives) == set(HW_OBJECTIVES)
        assert result.best.configuration["platform_constraints"]["cores"]


ARCH_CFG: Dict[str, Any] = {"grid_cap": 64}


def _create(name: str):
    return create_optimizer(
        optimizer_type=name,  # type: ignore[arg-type]
        arch_cfg=dict(ARCH_CFG),
        search_mode="hardware",
        arch_options=[],
        seed=3,
        pop_size=4,
        generations=2,
        target_tq=4,
    )


class TestTheFactoryIsOneDeclarativeTable:
    def test_every_catalogued_name_builds_an_optimizer(self):
        for name in OPTIMIZER_IDS:
            assert hasattr(_create(name), "optimize"), name

    def test_the_sampling_names_build_the_one_driver_with_their_own_stream(self):
        expected = {
            "random": RandomStrategy, "sobol": SobolStrategy, "exhaustive": GridStrategy,
        }
        for name, strategy_cls in expected.items():
            optimizer = _create(name)

            assert isinstance(optimizer, SamplingOptimizer), name
            assert isinstance(optimizer.strategy, strategy_cls), name

    def test_a_drawing_backend_plans_the_declared_number_of_candidates(self):
        # pop_size x generations is the same budget knob NSGA-II spends, so an
        # equal-budget campaign declares one number for every backend.
        strategy = _create("random").strategy

        assert isinstance(strategy, RandomStrategy)
        assert strategy.samples == 4 * 2

    def test_the_exhaustive_backend_takes_the_declared_cap(self):
        strategy = _create("exhaustive").strategy

        assert isinstance(strategy, GridStrategy)
        assert strategy.cap == ARCH_CFG["grid_cap"]

    def test_an_undeclared_cap_falls_back_to_the_frameworks_own(self):
        optimizer = create_optimizer(
            optimizer_type="exhaustive", arch_cfg={}, search_mode="hardware",
            arch_options=[], seed=0, pop_size=4, generations=2, target_tq=4,
        )

        assert isinstance(optimizer, SamplingOptimizer)
        assert isinstance(optimizer.strategy, GridStrategy)
        assert optimizer.strategy.cap == DEFAULT_GRID_CAP

    def test_an_unknown_optimizer_fails_loud_naming_the_choices(self):
        with pytest.raises(ValueError) as excinfo:
            _create("simulated_annealing")

        message = str(excinfo.value)
        assert "simulated_annealing" in message
        for name in OPTIMIZER_IDS:
            assert name in message, message

    def test_the_seed_reaches_the_driver(self):
        optimizer = _create("sobol")

        assert isinstance(optimizer, SamplingOptimizer)
        assert optimizer.seed == 3


class TestTheOptimizerCatalogueIsTheOneList:
    """Three surfaces name the backends — the type, the builders, the wizard —
    and a run declares a name against all three at once."""

    def test_the_builder_table_covers_exactly_the_catalogue(self):
        assert set(OPTIMIZER_BUILDERS) == set(OPTIMIZER_IDS)

    def test_the_declared_type_covers_exactly_the_catalogue(self):
        assert set(OptimizerType.__args__) == set(OPTIMIZER_IDS)  # type: ignore[attr-defined]

    def test_the_catalogue_names_all_six_backends(self):
        assert set(OPTIMIZER_IDS) == {
            "nsga2", "agent_evolve", "compilagent", "random", "sobol", "exhaustive",
        }

    def test_the_wizard_offers_the_catalogue_itself(self):
        offered = get_wizard_nas_schema()["optimizer_options"]

        assert [o["id"] for o in offered] == list(OPTIMIZER_IDS)
        assert [o["label"] for o in offered] == [c.label for c in OPTIMIZER_CHOICES]

    def test_the_wizard_declares_the_exhaustive_cap(self):
        # The wizard is the configurability SSOT: a knob a run can declare must
        # be declarable there, or the GUI cannot express the run.
        fields = get_wizard_nas_schema()["exhaustive_fields"]

        assert fields["grid_cap"]["type"] == "int"
        assert fields["grid_cap"]["default"] == DEFAULT_GRID_CAP

    def test_every_catalogued_label_is_distinct_and_non_empty(self):
        labels = [c.label for c in OPTIMIZER_CHOICES]

        assert len(set(labels)) == len(labels)
        assert all(label.strip() for label in labels)


class TestTheGridCapIsAFrameworkBound:
    def test_the_default_cap_is_a_positive_count(self):
        assert isinstance(DEFAULT_GRID_CAP, int) and DEFAULT_GRID_CAP > 0

    def test_the_cap_bounds_a_space_a_campaign_could_actually_run(self):
        # Every grid point costs a full candidate resolution, so the default is
        # a guard rail against an enumeration nobody could finish, not a limit
        # a real declared space is expected to hit.
        assert DEFAULT_GRID_CAP <= 100_000

    def test_the_real_encoding_can_out_grow_the_default_cap(self):
        # The refusal is not theoretical: the shipped default core bounds
        # already enumerate past any sane cap, which is why exhaustive search
        # is a TINY-SPACE baseline and says so by refusing.
        problem = JointArchHwProblem(
            data_provider_factory=None,
            device=torch.device("cpu"),
            input_shape=(1, 8, 8),
            num_classes=4,
            target_tq=4,
            lr=0.001,
            search_mode="hardware",
            builder_factory=SimpleMLPBuilder,
            model_config_assembler=lambda raw: dict(raw),
            fixed_model_config={"mlp_width_1": 16},
            platform_resolver=make_platform_resolver(_pipeline_config()),
            active_objective_names=HW_OBJECTIVES,
        )

        with pytest.raises(ValueError, match=str(DEFAULT_GRID_CAP)):
            problem.grid_vectors(cap=DEFAULT_GRID_CAP)


def test_the_grid_cardinality_matches_a_hand_rolled_product():
    """A second, independent statement of the cardinality claim.

    The pin above reads a literal; this one multiplies the per-dimension value
    sets the encoding declares, so a dropped dimension cannot pass by being
    dropped from both the code and one test.
    """
    axes = build_option_axes(["encoding_layer_placement"])
    problem = _problem(option_axes=axes)
    per_dimension = [
        AXONS_VALUES, NEURONS_VALUES, COUNT_VALUES, len(axes[0].choices),
    ]

    assert len(problem.grid_vectors(cap=DEFAULT_GRID_CAP)) == math.prod(per_dimension)
