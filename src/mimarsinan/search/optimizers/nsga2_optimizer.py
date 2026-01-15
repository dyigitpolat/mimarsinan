from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.results import Candidate, SearchResult


@dataclass
class NSGA2Optimizer(SearchOptimizer[Dict[str, Any]]):
    pop_size: int = 32
    generations: int = 20
    seed: int = 0
    eliminate_duplicates: bool = True
    verbose: bool = True

    # Penalty for invalid candidates (used when validate() fails or evaluate() throws)
    invalid_penalty: float = 1e18

    def optimize(self, problem: EncodedProblem[Dict[str, Any]]) -> SearchResult[Dict[str, Any]]:
        specs = list(problem.objectives)
        n_obj = len(specs)
        if n_obj == 0:
            raise ValueError("EncodedProblem.objectives must not be empty")

        # Track all evaluated candidates
        all_evaluated: List[Tuple[np.ndarray, Dict[str, float], int]] = []
        current_gen = [0]  # mutable container for closure

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
                    xl=np.array(problem.xl, dtype=float),
                    xu=np.array(problem.xu, dtype=float),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                # decode + round-to-int policy is implemented by the EncodedProblem
                try:
                    cfg = problem.decode(np.array(x, dtype=float))
                    if not problem.validate(cfg):
                        obj = {s.name: (0.0 if s.goal == "max" else self_outer.invalid_penalty) for s in specs}
                        all_evaluated.append((x.copy(), obj, current_gen[0]))
                        out["F"] = np.full((n_obj,), self_outer.invalid_penalty, dtype=float)
                        return
                    obj = problem.evaluate(cfg)
                    all_evaluated.append((x.copy(), obj, current_gen[0]))
                    out["F"] = to_minimization(obj)
                except Exception:
                    obj = {s.name: (0.0 if s.goal == "max" else self_outer.invalid_penalty) for s in specs}
                    all_evaluated.append((x.copy(), obj, current_gen[0]))
                    out["F"] = np.full((n_obj,), self_outer.invalid_penalty, dtype=float)

        # capture for nested class
        self_outer = self

        algo = NSGA2(pop_size=int(self.pop_size), eliminate_duplicates=bool(self.eliminate_duplicates))
        termination = get_termination("n_gen", int(self.generations))

        # Callback to track generation
        from pymoo.core.callback import Callback
        class GenCallback(Callback):
            def notify(self, algorithm):
                current_gen[0] = algorithm.n_gen

        res = minimize(
            _PymooProblem(),
            algo,
            termination,
            seed=int(self.seed),
            save_history=True,
            verbose=bool(self.verbose),
            callback=GenCallback(),
        )

        # Build pareto front X coordinates as set for quick lookup
        pareto_x_set: Set[Tuple[float, ...]] = set()
        if res.X is not None:
            xs = np.atleast_2d(res.X)
            for x in xs:
                pareto_x_set.add(tuple(x.tolist()))

        # Pymoo returns the nondominated set in res.X / res.F (typically).
        pareto: List[Candidate[Dict[str, Any]]] = []
        if res.X is not None:
            xs = np.atleast_2d(res.X)
            for x in xs:
                cfg = problem.decode(np.array(x, dtype=float))
                obj = problem.evaluate(cfg) if problem.validate(cfg) else {s.name: (0.0 if s.goal == "max" else self.invalid_penalty) for s in specs}
                pareto.append(Candidate(configuration=cfg, objectives=obj, metadata={"x": x.tolist(), "is_pareto": True}))

        # Build all_candidates from tracked evaluations
        all_candidates: List[Candidate[Dict[str, Any]]] = []
        for x, obj, gen in all_evaluated:
            cfg = problem.decode(np.array(x, dtype=float))
            is_pareto = tuple(x.tolist()) in pareto_x_set
            all_candidates.append(Candidate(
                configuration=cfg,
                objectives=obj,
                metadata={"x": x.tolist(), "generation": gen, "is_pareto": is_pareto}
            ))

        # Selection rule: maximize accuracy, then minimize cores, unused/core, params (as per plan).
        def _sort_key(c: Candidate[Dict[str, Any]]):
            acc = float(c.objectives.get("accuracy", 0.0))
            cores = float(c.objectives.get("hard_cores_used", self.invalid_penalty))
            unused = float(c.objectives.get("avg_unused_area_per_core", self.invalid_penalty))
            params = float(c.objectives.get("total_params", self.invalid_penalty))
            return (-acc, cores, unused, params)

        best = min(pareto, key=_sort_key) if pareto else Candidate(configuration={}, objectives={}, metadata={})

        history = []
        if getattr(res, "history", None):
            for gen_idx, h in enumerate(res.history):
                try:
                    F = np.array(getattr(h, "opt").get("F"))
                    f_min = np.min(F, axis=0)
                    best_vals = {}
                    for i, spec in enumerate(specs):
                        v = float(f_min[i])
                        best_vals[spec.name] = float(-v) if spec.goal == "max" else float(v)
                    history.append({"gen": gen_idx, "best": best_vals})
                except Exception:
                    history.append({"gen": gen_idx})

        return SearchResult(objectives=specs, best=best, pareto_front=pareto, all_candidates=all_candidates, history=history)


