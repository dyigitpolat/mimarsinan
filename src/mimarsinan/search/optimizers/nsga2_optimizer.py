from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from mimarsinan.common.best_effort import best_effort
from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.results import Candidate, SearchResult, select_minimax_rank

logger = logging.getLogger(__name__)


@dataclass
class NSGA2Optimizer(SearchOptimizer[Dict[str, Any]]):
    pop_size: int = 32
    generations: int = 20
    seed: int = 0
    eliminate_duplicates: bool = True
    verbose: bool = True

    invalid_penalty: float = 1e18

    def optimize(self, problem: EncodedProblem[Dict[str, Any]], reporter=None) -> SearchResult[Dict[str, Any]]:
        specs = list(problem.objectives)
        n_obj = len(specs)
        if n_obj == 0:
            raise ValueError("EncodedProblem.objectives must not be empty")

        all_evaluated: List[Tuple[np.ndarray, Dict[str, float], int]] = []
        current_gen = [0]

        def to_minimization(obj: Dict[str, float]) -> np.ndarray:
            vals = []
            for spec in specs:
                v = float(obj[spec.name])
                vals.append(-v if spec.goal == "max" else v)
            return np.array(vals, dtype=float)

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
                try:
                    cfg = problem.decode(np.array(x, dtype=float))

                    # pymoo constraint-domination convention: G <= 0 is feasible, and among infeasible candidates smaller G dominates.
                    cv = float(problem.constraint_violation(cfg))
                    out["G"] = np.array([cv])

                    if cv > 0:
                        obj = {s.name: (0.0 if s.goal == "max" else self_outer.invalid_penalty) for s in specs}
                        all_evaluated.append((x.copy(), obj, current_gen[0]))
                        out["F"] = np.full((n_obj,), self_outer.invalid_penalty, dtype=float)
                        return

                    obj = problem.evaluate(cfg)
                    all_evaluated.append((x.copy(), obj, current_gen[0]))
                    out["F"] = to_minimization(obj)
                except Exception as exc:
                    logger.warning(
                        "NSGA2 candidate evaluation failed (%s: %s) for x=%s; "
                        "recording penalty objectives",
                        type(exc).__name__, exc,
                        np.array(x, dtype=float).tolist(),
                        exc_info=True,
                    )
                    obj = {s.name: (0.0 if s.goal == "max" else self_outer.invalid_penalty) for s in specs}
                    all_evaluated.append((x.copy(), obj, current_gen[0]))
                    out["F"] = np.full((n_obj,), self_outer.invalid_penalty, dtype=float)
                    out["G"] = np.array([1e6])

        self_outer = self

        algo = NSGA2(pop_size=int(self.pop_size), eliminate_duplicates=bool(self.eliminate_duplicates))
        termination = get_termination("n_gen", int(self.generations))

        _reporter = reporter
        _specs = specs

        class GenCallback(Callback):
            def notify(self, algorithm):
                current_gen[0] = algorithm.n_gen
                if _reporter is None:
                    return
                with best_effort("nsga2 generation metrics report", logger=logger):
                    _reporter("Search generation", algorithm.n_gen)
                    F = np.array(algorithm.opt.get("F"))
                    for i, spec in enumerate(_specs):
                        v = float(np.min(F[:, i]))
                        val = -v if spec.goal == "max" else v
                        _reporter(f"Search best {spec.name}", val)
                    _reporter("Search Pareto size", len(F))

        res = minimize(
            _PymooProblem(),
            algo,
            termination,
            seed=int(self.seed),
            save_history=True,
            verbose=bool(self.verbose),
            callback=GenCallback(),
        )

        pareto_x_set: Set[Tuple[float, ...]] = set()
        if res.X is not None:
            xs = np.atleast_2d(res.X)
            for x in xs:
                pareto_x_set.add(tuple(x.tolist()))

        pareto: List[Candidate[Dict[str, Any]]] = []
        if res.X is not None:
            xs = np.atleast_2d(res.X)
            for x in xs:
                cfg = problem.decode(np.array(x, dtype=float))
                obj = problem.evaluate(cfg) if problem.validate(cfg) else {s.name: (0.0 if s.goal == "max" else self.invalid_penalty) for s in specs}
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

        history: List[Dict[str, Any]] = []
        if getattr(res, "history", None):
            for gen_idx, h in enumerate(res.history):
                entry: Dict[str, Any] = {"gen": gen_idx}
                with best_effort("nsga2 history best-values extraction", logger=logger):
                    F = np.array(getattr(h, "opt").get("F"))
                    f_min = np.min(F, axis=0)
                    best_vals = {}
                    for i, spec in enumerate(specs):
                        v = float(f_min[i])
                        best_vals[spec.name] = float(-v) if spec.goal == "max" else float(v)
                    entry["best"] = best_vals
                history.append(entry)

        return SearchResult(objectives=specs, best=best, pareto_front=pareto, all_candidates=all_candidates, history=history)


