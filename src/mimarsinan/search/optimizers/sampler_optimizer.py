from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

import torch.multiprocessing as mp

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import Candidate, ObjectiveSpec, SearchResult


ConfigT = TypeVar("ConfigT")


class ConfigurationSampler(Generic[ConfigT]):
    """
    Legacy sampler interface (already implemented by BasicConfigurationSampler).
    """

    def sample(self, n: int) -> List[ConfigT]:
        raise NotImplementedError

    def update(self, metrics: List[Tuple[ConfigT, float]]) -> None:
        raise NotImplementedError


def _evaluate_runner(args):
    problem, configuration = args
    return configuration, problem.evaluate(configuration), problem.meta(configuration)


@dataclass
class SamplerOptimizer(SearchOptimizer[ConfigT]):
    """
    Simple single-objective optimizer driven by a configuration sampler with feedback.

    This is a refactor of the legacy BasicArchitectureSearcher logic into a reusable optimizer.
    """

    sampler: ConfigurationSampler[ConfigT]
    cycles: int = 5
    batch_size: int = 50
    workers: int = 1
    max_sampling_factor: int = 10_000

    def _score(self, objectives: Dict[str, float], spec: ObjectiveSpec) -> float:
        v = float(objectives[spec.name])
        return v if spec.goal == "max" else -v

    def _sample_valid(self, problem: SearchProblem[ConfigT], n: int) -> List[ConfigT]:
        configurations: List[ConfigT] = []
        total_sample_size = 0

        while len(configurations) < n:
            sampled = self.sampler.sample(n)
            total_sample_size += n

            for c in sampled:
                if problem.validate(c):
                    configurations.append(copy.deepcopy(c))
                    if len(configurations) >= n:
                        break

            if total_sample_size > self.max_sampling_factor * n:
                raise RuntimeError("Cannot sample enough valid configurations (validation always failing?)")

        return configurations

    def _evaluate_batch(self, problem: SearchProblem[ConfigT], configurations: List[ConfigT]):
        if self.workers <= 1:
            out = []
            for c in configurations:
                out.append((c, problem.evaluate(c), problem.meta(c)))
            return out

        with mp.Pool(processes=int(self.workers)) as pool:
            params = [(problem, c) for c in configurations]
            return pool.map(_evaluate_runner, params)

    def optimize(self, problem: SearchProblem[ConfigT]) -> SearchResult[ConfigT]:
        if not problem.objectives:
            raise ValueError("SearchProblem.objectives must not be empty")

        # This optimizer is single-objective by design: it uses the first objective
        # to drive sampler.update(). Multi-objective optimization uses NSGA-II later.
        primary = problem.objectives[0]

        best: Candidate[ConfigT] | None = None
        best_score: float = float("-inf")

        history: List[Dict[str, Any]] = []

        for cycle_idx in range(int(self.cycles)):
            configs = self._sample_valid(problem, int(self.batch_size))

            evaluated = self._evaluate_batch(problem, configs)
            metrics_for_update: List[Tuple[ConfigT, float]] = []

            cycle_best_score = float("-inf")
            cycle_best = None

            for c, objectives, meta in evaluated:
                score = self._score(objectives, primary)
                metrics_for_update.append((c, score))

                if score > cycle_best_score:
                    cycle_best_score = score
                    cycle_best = Candidate(configuration=c, objectives=objectives, metadata=meta or {})

                if score > best_score:
                    best_score = score
                    best = Candidate(configuration=c, objectives=objectives, metadata=meta or {})

            self.sampler.update(metrics_for_update)

            history.append(
                {
                    "cycle": cycle_idx,
                    "primary_objective": primary.name,
                    "cycle_best": (cycle_best.objectives if cycle_best else None),
                    "best_so_far": (best.objectives if best else None),
                }
            )

        if best is None:
            raise RuntimeError("Optimization produced no candidates")

        return SearchResult(objectives=problem.objectives, best=best, pareto_front=[], history=history)


