"""Everything the NSGA-II driver says in pymoo's vocabulary, said in one place.

pymoo minimizes every objective and knows nothing about a goal-carrying
:class:`ObjectiveSpec`, so a translation layer is unavoidable; keeping it here
leaves the driver itself to describe the SEARCH — sampling, the generation
boundary, and what the run seals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
from pymoo.operators.sampling.rnd import FloatRandomSampling

from mimarsinan.common.best_effort import best_effort
from mimarsinan.search.results import ObjectiveSpec, order_by_minimax_rank

logger = logging.getLogger(__name__)


def seeded_sampling(seeds: Sequence[Any]) -> FloatRandomSampling:
    """Random sampling with the leading rows replaced by the given seeds."""

    class _Seeded(FloatRandomSampling):
        def _do(self, problem, n_samples, **kwargs):
            X = super()._do(problem, n_samples, **kwargs)
            for i, seed in enumerate(seeds[: len(X)]):
                X[i] = seed
            return X

    return _Seeded()


def to_minimization(
    objectives: Dict[str, float], specs: Sequence[ObjectiveSpec],
) -> np.ndarray:
    """One candidate's objectives in pymoo's minimize-everything space."""
    return np.array(
        [
            -float(objectives[spec.name]) if spec.goal == "max"
            else float(objectives[spec.name])
            for spec in specs
        ],
        dtype=float,
    )


def penalty_objectives(
    specs: Sequence[ObjectiveSpec], penalty: float,
) -> Dict[str, float]:
    """The worst score on every axis, in USER space — an infeasible candidate."""
    return {s.name: (0.0 if s.goal == "max" else penalty) for s in specs}


def user_space_front(
    F: np.ndarray, specs: Sequence[ObjectiveSpec],
) -> List[Dict[str, float]]:
    """The front in USER space, incumbent first (the panel renders the head)."""
    rows = [
        {
            spec.name: (-float(v) if spec.goal == "max" else float(v))
            for spec, v in zip(specs, row)
        }
        for row in F
    ]
    return [rows[i] for i in order_by_minimax_rank(rows, specs)]


def history_rows(
    history: Sequence[Any], specs: Sequence[ObjectiveSpec],
) -> List[Dict[str, Any]]:
    """pymoo's per-generation history as the report's own rows.

    ONE generation numbering per SearchResult: this ``gen`` is the same 1-based
    ordinal the candidate tags, the emitted frames, and the LLM backends' own
    history entries carry. A 0-based row here would put the first generation at
    x=0 in the report while its candidates claimed generation 1.
    """
    rows: List[Dict[str, Any]] = []
    for gen_idx, entry in enumerate(history, start=1):
        row: Dict[str, Any] = {"gen": gen_idx}
        with best_effort("nsga2 history best-values extraction", logger=logger):
            f_min = np.min(np.array(getattr(entry, "opt").get("F")), axis=0)
            row["best"] = {
                spec.name: (
                    float(-f_min[i]) if spec.goal == "max" else float(f_min[i])
                )
                for i, spec in enumerate(specs)
            }
        rows.append(row)
    return rows
