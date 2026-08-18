from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from mimarsinan.common.best_effort import best_effort
from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.optimizers.budget import problem_budget, seal_ledger
from mimarsinan.search.optimizers.pymoo_bridge import (
    history_rows,
    penalty_objectives,
    seeded_sampling,
    to_minimization,
    user_space_front,
)
from mimarsinan.search.optimizers.search_events import (
    candidates_generated_event,
    emit_search_event,
    generation_complete_event,
    generation_start_event,
    search_complete_event,
)
from mimarsinan.search.problem import CandidateInfeasibleError
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.results import (
    Candidate,
    SearchResult,
    select_minimax_rank,
)

logger = logging.getLogger(__name__)


@dataclass
class NSGA2Optimizer(SearchOptimizer[Dict[str, Any]]):
    pop_size: int = 32
    generations: int = 20
    seed: int = 0
    eliminate_duplicates: bool = True
    verbose: bool = True

    invalid_penalty: float = 1e18

    def optimize(self, problem: EncodedProblem[Dict[str, Any]], reporter=None) -> SearchResult[Dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride] — vector-encoded optimizer requires EncodedProblem
        specs = list(problem.objectives)
        n_obj = len(specs)
        if n_obj == 0:
            raise ValueError("EncodedProblem.objectives must not be empty")

        all_evaluated: List[Tuple[np.ndarray, Dict[str, float], int]] = []
        # Generations are 1-based, as pymoo's ``n_gen`` and the search report's
        # G-badges already are; the callback advances this at each generation
        # BOUNDARY, so a candidate is tagged with the generation that produced it.
        current_gen = [1]
        verdicts: Dict[int, List[int]] = {}

        def tally(gen: int, is_valid: bool) -> None:
            counts = verdicts.setdefault(gen, [0, 0])
            counts[0 if is_valid else 1] += 1

        penalty = float(self.invalid_penalty)

        class _PymooProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=int(problem.n_var),
                    n_obj=int(n_obj),
                    n_ieq_constr=1,
                    xl=np.array(problem.xl, dtype=float),
                    xu=np.array(problem.xu, dtype=float),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                # Only candidate-dependent infeasibility degrades to a penalty
                # row; problem-level breakage propagates and aborts the search.
                try:
                    cfg = problem.decode(np.array(x, dtype=float))

                    # pymoo constraint-domination convention: G <= 0 is feasible, and among infeasible candidates smaller G dominates.
                    cv = float(problem.constraint_violation(cfg))
                    out["G"] = np.array([cv])

                    if cv > 0:
                        obj = penalty_objectives(specs, penalty)
                        all_evaluated.append((x.copy(), obj, current_gen[0]))
                        tally(current_gen[0], False)
                        out["F"] = np.full((n_obj,), penalty, dtype=float)
                        return

                    obj = problem.evaluate(cfg)
                    all_evaluated.append((x.copy(), obj, current_gen[0]))
                    tally(current_gen[0], True)
                    out["F"] = to_minimization(obj, specs)
                except CandidateInfeasibleError as exc:
                    logger.warning(
                        "NSGA2 candidate infeasible (%s: %s) for x=%s; "
                        "recording penalty objectives",
                        type(exc).__name__, exc,
                        np.array(x, dtype=float).tolist(),
                        exc_info=True,
                    )
                    obj = penalty_objectives(specs, penalty)
                    all_evaluated.append((x.copy(), obj, current_gen[0]))
                    tally(current_gen[0], False)
                    out["F"] = np.full((n_obj,), penalty, dtype=float)
                    out["G"] = np.array([1e6])

        # [R6] Generation 1 carries the DECLARED platform when the problem
        # can encode it: the declaration is the known-feasible point, and a
        # shape-constrained space searched from pure noise can reject every
        # offspring (measured: 72/72 on the ViT cell).
        seeds = list(getattr(problem, "seed_vectors", lambda: [])())
        sampling = seeded_sampling(seeds) if seeds else FloatRandomSampling()
        algo = NSGA2(
            pop_size=int(self.pop_size), sampling=sampling,
            eliminate_duplicates=bool(self.eliminate_duplicates),
        )
        termination = get_termination("n_gen", int(self.generations))

        _reporter = reporter
        _specs = specs
        total_gens = int(self.generations)
        pop_size = int(self.pop_size)
        # [TS1] The accountant is the PROBLEM's; a driver only decides where its
        # own boundary is once the distinct budget is spent.
        budget = problem_budget(problem)
        stopped = [False]

        def emit_generation_frames(gen: int, F: np.ndarray) -> None:
            """One start/count/complete triple per generation — never per candidate."""
            valid, failed = verdicts.get(gen, [0, 0])
            emit_search_event(_reporter, generation_start_event(
                gen=gen, total_gens=total_gens,
                phase="initial" if gen == 1 else "evolution",
                objectives=_specs, pop_size=pop_size,
            ))
            emit_search_event(_reporter, candidates_generated_event(
                gen=gen, count=valid + failed,
            ))
            emit_search_event(_reporter, generation_complete_event(
                gen=gen, valid_count=valid, failed_count=failed,
                pareto_objectives=user_space_front(F, _specs),
            ))

        class GenCallback(Callback):
            """The generation BOUNDARY: the budget stop first, then telemetry.

            Only the telemetry is best_effort — reporting is the ONE side concern
            a search may lose without losing the search (the sanctioned seam).
            A lost budget stop would buy evaluations nobody authorized.
            """

            def notify(self, algorithm):
                gen = algorithm.n_gen
                # The generation that just closed keeps its tag; everything
                # evaluated from here on belongs to the next one.
                current_gen[0] = gen + 1
                if budget is not None and budget.exhausted:
                    # Only a stop that DENIED a generation cut the run short:
                    # the last one was ending on its own, and a campaign reads
                    # this flag to tell budget-bound runs from generation-bound.
                    stopped[0] = stopped[0] or gen < total_gens
                    algorithm.termination.terminate()
                    # pymoo updates the criterion BEFORE calling back, so the
                    # forced flag must be re-read here or the run spends one
                    # more generation past the boundary.
                    algorithm.termination.update(algorithm)
                if _reporter is None:
                    return
                front = None
                with best_effort("nsga2 generation front read", logger=logger):
                    front = np.array(algorithm.opt.get("F"))
                if front is None:
                    return
                with best_effort("nsga2 generation metrics report", logger=logger):
                    _reporter("Search generation", gen)
                    for i, spec in enumerate(_specs):
                        v = float(np.min(front[:, i]))
                        val = -v if spec.goal == "max" else v
                        _reporter(f"Search best {spec.name}", val)
                    _reporter("Search Pareto size", len(front))
                with best_effort("nsga2 generation search_event frames", logger=logger):
                    emit_generation_frames(gen, front)

        started = perf_counter()
        res = minimize(
            _PymooProblem(),
            algo,
            termination,
            seed=int(self.seed),
            save_history=True,
            verbose=bool(self.verbose),
            callback=GenCallback(),
        )
        wall_s = perf_counter() - started
        # [TS1] Sealed where the clock stops, so the counts and the wall cover
        # ONE interval: the search, never the front re-read below.
        ledger = seal_ledger(budget, wall_s=wall_s, stopped_at_boundary=stopped[0])

        front_x = [] if res.X is None else list(np.atleast_2d(res.X))
        pareto_x_set: Set[Tuple[float, ...]] = {tuple(x.tolist()) for x in front_x}

        pareto: List[Candidate[Dict[str, Any]]] = []
        for x in front_x:
            cfg = problem.decode(np.array(x, dtype=float))
            if problem.validate(cfg):
                try:
                    obj = problem.evaluate(cfg)
                except CandidateInfeasibleError:
                    obj = penalty_objectives(specs, penalty)
            else:
                obj = penalty_objectives(specs, penalty)
            pareto.append(Candidate(configuration=cfg, objectives=obj, metadata={"x": x.tolist(), "is_pareto": True}))

        all_candidates: List[Candidate[Dict[str, Any]]] = []
        for x, obj, gen in all_evaluated:
            cfg = problem.decode(np.array(x, dtype=float))
            is_pareto = tuple(x.tolist()) in pareto_x_set
            all_candidates.append(Candidate(
                configuration=cfg,
                objectives=obj,
                metadata={"x": x.tolist(), "generation": gen, "is_pareto": is_pareto}
            ))

        best = select_minimax_rank(pareto, specs) or Candidate(configuration={}, objectives={}, metadata={})

        history = history_rows(getattr(res, "history", None) or [], specs)

        total_valid = sum(counts[0] for counts in verdicts.values())
        total_failed = sum(counts[1] for counts in verdicts.values())
        emit_search_event(reporter, search_complete_event(
            total_valid=total_valid,
            total_failed=total_failed,
            final_pareto_size=len(pareto),
        ))

        return SearchResult(
            objectives=specs, best=best, pareto_front=pareto,
            all_candidates=all_candidates, history=history, ledger=ledger,
        )
