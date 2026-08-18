"""[TS2] The non-evolutionary backends: one driver, three streams of vectors.

A campaign needs baselines an evolutionary search is measured AGAINST: a
uniform draw, a low-discrepancy draw, and — where the declared space is small
enough — the whole grid. They differ in one thing, which vectors they propose,
so there is ONE driver behind a :class:`SamplingStrategy` protocol instead of
three optimizers each re-deriving decoding, penalties, the front, the ledger
and the live-event frames.

The driver never counts its own draws: the TS1 currency is the decoded
candidate IDENTITY — two vectors can be one candidate (a snapped core
dimension), and one vector proposed twice is one candidate asked twice.
Stopping is its own decision at a BATCH boundary, so an exhausted budget buys a
whole batch and the ledger seals the exact spend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import (
    Any, Dict, Iterator, List, NamedTuple, Optional, Protocol, Sequence, Tuple,
)

import numpy as np
from scipy.stats import qmc

from mimarsinan.common.best_effort import best_effort
from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.optimizers.budget import (
    BoundaryStop, ResourceLedger, problem_budget, seal_ledger,
)
from mimarsinan.search.optimizers.pymoo_bridge import penalty_objectives
from mimarsinan.search.optimizers.search_events import (
    candidates_generated_event, emit_search_event, generation_complete_event,
    generation_start_event, search_complete_event,
)
from mimarsinan.search.problem import CandidateInfeasibleError
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.results import (
    Candidate, ObjectiveSpec, SearchResult, nondominated_front,
    order_by_minimax_rank, select_minimax_rank,
)

logger = logging.getLogger(__name__)

#: A guard rail, not a search policy: a grid point costs a full candidate
#: resolution, so a larger enumeration is a run nobody finishes — refused by
#: name unless the caller declares ``arch_search.grid_cap``.
DEFAULT_GRID_CAP = 4096


class SamplingStrategy(Protocol):
    """A finite stream of encoded vectors over a problem's declared box."""

    def vectors(
        self, problem: EncodedProblem[Any], rng: np.random.Generator,
    ) -> Iterator[np.ndarray]: ...


class _Evaluated(NamedTuple):
    x: np.ndarray
    configuration: Dict[str, Any]
    objectives: Dict[str, float]
    gen: int


def _box(problem: EncodedProblem[Any]) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(problem.xl, dtype=float), np.asarray(problem.xu, dtype=float)


def _positive(value: int, name: str) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive count, got {value!r}")


@dataclass(frozen=True)
class RandomStrategy:
    """Uniform draws inside ``[xl, xu]`` — the "did search help at all" baseline."""

    samples: int

    def __post_init__(self) -> None:
        _positive(self.samples, "samples")

    def vectors(
        self, problem: EncodedProblem[Any], rng: np.random.Generator,
    ) -> Iterator[np.ndarray]:
        low, high = _box(problem)
        for _ in range(int(self.samples)):
            yield rng.uniform(low, high)


@dataclass(frozen=True)
class SobolStrategy:
    """Scrambled Sobol' draws — the same spend, spread evenly over the box."""

    samples: int

    def __post_init__(self) -> None:
        _positive(self.samples, "samples")

    def vectors(
        self, problem: EncodedProblem[Any], rng: np.random.Generator,
    ) -> Iterator[np.ndarray]:
        low, high = _box(problem)
        engine = qmc.Sobol(d=int(problem.n_var), scramble=True, rng=rng)
        # Sobol' balance is a property of 2^m blocks, so the run draws the next
        # power of two and takes the sequence's leading points — the points a
        # non-power-of-two draw returns, minus scipy's warning about a property
        # this never claimed.
        block = 1 << max(0, (int(self.samples) - 1).bit_length())
        for point in qmc.scale(engine.random(block)[: int(self.samples)], low, high):
            yield np.asarray(point, dtype=float)


@dataclass(frozen=True)
class GridStrategy:
    """The encoding's own enumeration — exhaustive, or a loud refusal."""

    cap: int = DEFAULT_GRID_CAP

    def __post_init__(self) -> None:
        _positive(self.cap, "cap")

    def vectors(
        self, problem: EncodedProblem[Any], rng: np.random.Generator,
    ) -> Iterator[np.ndarray]:
        enumerate_grid = getattr(problem, "grid_vectors", None)
        if enumerate_grid is None:
            raise TypeError(
                f"exhaustive search needs an encoding that enumerates itself; "
                f"{type(problem).__name__} declares no grid_vectors()"
            )
        return iter(enumerate_grid(cap=int(self.cap)))


@dataclass
class SamplingOptimizer(SearchOptimizer[Dict[str, Any]]):
    """Streams a strategy's vectors through the problem, in batches.

    The batch IS the generation every optimizer's live frames talk about: it
    bounds the budget stop, tags the candidates and paces the panel — a unit of
    accounting, which is why a sampler shares the evolutionary vocabulary.
    """

    strategy: SamplingStrategy
    pop_size: int = 12
    seed: int = 0
    invalid_penalty: float = 1e18
    _verdicts: Dict[int, List[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _positive(self.pop_size, "pop_size")

    def optimize(  # pyright: ignore[reportIncompatibleMethodOverride] — vector-encoded optimizer requires EncodedProblem
        self, problem: EncodedProblem[Dict[str, Any]], reporter=None,
    ) -> SearchResult[Dict[str, Any]]:
        specs = list(problem.objectives)
        if not specs:
            raise ValueError("EncodedProblem.objectives must not be empty")

        rng = np.random.default_rng(int(self.seed))
        # Materialized so the run declares its SHAPE in the frames (the batches
        # a viewer waits for); every strategy is finite by construction.
        plan = list(self.strategy.vectors(problem, rng))
        size = int(self.pop_size)
        batches = [plan[i:i + size] for i in range(0, len(plan), size)]
        budget = problem_budget(problem)
        boundary = BoundaryStop(budget)

        # The tally belongs to THIS run: a reused optimizer must not report the
        # batches of the last one.
        self._verdicts = {}
        rows: List[_Evaluated] = []
        started = perf_counter()
        for gen, batch in enumerate(batches, start=1):
            self._verdicts[gen] = [0, 0]
            rows.extend(self._score(problem, x, specs, gen) for x in batch)
            self._emit_batch_frames(reporter, gen, len(batches), specs, rows)
            # Asked only where a batch is left to DENY: the last one was ending
            # on its own, and only a stop that cut the stream short is one.
            if gen < len(batches) and boundary.should_stop():
                break
        wall_s = perf_counter() - started
        # Sealed where the clock stops: one interval, the search itself.
        ledger = seal_ledger(
            budget, wall_s=wall_s, stopped_at_boundary=boundary.stopped,
        )

        return self._result(specs, rows, ledger, reporter)

    def _score(
        self, problem: EncodedProblem[Dict[str, Any]], x: np.ndarray,
        specs: Sequence[ObjectiveSpec], gen: int,
    ) -> _Evaluated:
        """One draw, scored through the same screen-then-score channels an
        evolutionary run uses — a penalty row where the candidate is infeasible."""
        penalty = penalty_objectives(specs, float(self.invalid_penalty))
        configuration: Dict[str, Any] = {}
        try:
            configuration = problem.decode(np.array(x, dtype=float))
            if float(problem.constraint_violation(configuration)) > 0.0:
                self._tally(gen, False)
                return _Evaluated(x, configuration, penalty, gen)
            objectives = problem.evaluate(configuration)
        except CandidateInfeasibleError as exc:
            logger.warning(
                "sampled candidate infeasible (%s: %s) for x=%s; penalty row",
                type(exc).__name__, exc, np.asarray(x, dtype=float).tolist(),
                exc_info=True,
            )
            self._tally(gen, False)
            return _Evaluated(x, configuration, penalty, gen)
        self._tally(gen, True)
        return _Evaluated(x, configuration, objectives, gen)

    def _tally(self, gen: int, is_valid: bool) -> None:
        counts = self._verdicts.setdefault(gen, [0, 0])
        counts[0 if is_valid else 1] += 1

    def _emit_batch_frames(
        self, reporter, gen: int, total_gens: int,
        specs: Sequence[ObjectiveSpec], rows: Sequence[_Evaluated],
    ) -> None:
        """The shared start/count/complete triple — telemetry, never the search."""
        if reporter is None:
            return
        valid, failed = self._verdicts.get(gen, [0, 0])
        with best_effort("sampling generation search_event frames", logger=logger):
            emit_search_event(reporter, generation_start_event(
                gen=gen, total_gens=total_gens,
                # A drawn batch never EVOLVED; the panel shows no pill for a
                # phase it does not know, which is the honest rendering.
                phase="initial" if gen == 1 else "sampling",
                objectives=specs, pop_size=int(self.pop_size),
            ))
            emit_search_event(reporter, candidates_generated_event(
                gen=gen, count=valid + failed,
            ))
            # The front SO FAR, incumbent first: the panel renders the head.
            objectives = [row.objectives for row in rows]
            front = [objectives[i] for i in nondominated_front(objectives, specs)]
            emit_search_event(reporter, generation_complete_event(
                gen=gen, valid_count=valid, failed_count=failed,
                pareto_objectives=[
                    front[i] for i in order_by_minimax_rank(front, specs)
                ],
            ))

    def _result(
        self, specs: Sequence[ObjectiveSpec], rows: Sequence[_Evaluated],
        ledger: Optional[ResourceLedger], reporter,
    ) -> SearchResult[Dict[str, Any]]:
        front_indices = set(nondominated_front([row.objectives for row in rows], specs))
        all_candidates = [
            Candidate(
                configuration=row.configuration, objectives=row.objectives,
                metadata={
                    "x": np.asarray(row.x, dtype=float).tolist(),
                    "generation": row.gen,
                    "is_pareto": index in front_indices,
                },
            )
            for index, row in enumerate(rows)
        ]
        pareto = [c for c in all_candidates if c.metadata["is_pareto"]]
        empty: Candidate[Dict[str, Any]] = Candidate(configuration={}, objectives={})
        emit_search_event(reporter, search_complete_event(
            total_valid=sum(counts[0] for counts in self._verdicts.values()),
            total_failed=sum(counts[1] for counts in self._verdicts.values()),
            final_pareto_size=len(pareto),
        ))
        return SearchResult(
            objectives=list(specs), pareto_front=pareto,
            best=select_minimax_rank(pareto, specs) or empty,
            all_candidates=all_candidates,
            history=self._history(specs, rows), ledger=ledger,
        )

    @staticmethod
    def _history(
        specs: Sequence[ObjectiveSpec], rows: Sequence[_Evaluated],
    ) -> List[Dict[str, Any]]:
        """The report's 1-based rows: the best value per axis UP TO each batch."""
        return [
            {"gen": gen, "best": {
                spec.name: (max if spec.goal == "max" else min)(
                    float(row.objectives.get(spec.name, 0.0))
                    for row in rows if row.gen <= gen
                )
                for spec in specs
            }}
            for gen in sorted({row.gen for row in rows})
        ]
