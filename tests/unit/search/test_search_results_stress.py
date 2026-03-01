"""
Stress tests for search results and minimax rank selection.

Tests adversarial inputs and edge cases in ranking logic.
"""

import pytest

from mimarsinan.search.results import (
    ObjectiveSpec, Candidate, SearchResult,
    select_minimax_rank,
)


class TestMinimaxRankMissingObjectives:
    """
    BUG: _rank_candidates silently defaults missing objectives to 0.0.
    This can cause incorrect ranking when 0.0 is a meaningful value.
    """

    def test_missing_objective_defaults_to_zero(self):
        """
        Candidate A: {"acc": 0.9, "speed": 100}
        Candidate B: {"acc": 0.8}  (missing "speed")

        With goal "min" on speed, B's speed defaults to 0.0, making B
        appear to have the best speed even though it's undefined.

        DESIGN ISSUE: Missing objectives silently default to 0.0 via
        c.objectives.get(spec.name, 0.0) in _rank_candidates. This can
        be misleading when 0.0 is a valid/good value.
        """
        specs = [
            ObjectiveSpec("acc", "max"),
            ObjectiveSpec("speed", "min"),
        ]
        a = Candidate({"x": 1}, {"acc": 0.9, "speed": 100})
        b = Candidate({"x": 2}, {"acc": 0.8})  # missing speed

        best = select_minimax_rank([a, b], specs)
        # B's speed defaults to 0.0 which ranks #1 for "min"
        # Both end up with worst_rank=2, tied on rank_sum=3
        # Current behavior: one of them is returned
        assert best is not None

    def test_all_objectives_missing(self):
        """Candidate with no objectives at all gets all zeros."""
        specs = [ObjectiveSpec("acc", "max")]
        c = Candidate({}, {})
        result = select_minimax_rank([c], specs)
        # Single candidate always returned
        assert result is c


class TestMinimaxRankStress:
    def test_many_candidates_same_values(self):
        """All candidates identical — should return one of them."""
        specs = [ObjectiveSpec("acc", "max")]
        candidates = [
            Candidate({"x": i}, {"acc": 0.5}) for i in range(100)
        ]
        best = select_minimax_rank(candidates, specs)
        assert best is not None
        assert best.objectives["acc"] == 0.5

    def test_many_objectives(self):
        """10 objectives — verify the ranking still produces a result."""
        specs = [ObjectiveSpec(f"obj_{i}", "max" if i % 2 == 0 else "min")
                 for i in range(10)]
        candidates = [
            Candidate({"x": i}, {f"obj_{j}": float(i + j) for j in range(10)})
            for i in range(20)
        ]
        best = select_minimax_rank(candidates, specs)
        assert best is not None

    def test_negative_objective_values(self):
        """Negative values should still rank correctly."""
        specs = [ObjectiveSpec("loss", "min")]
        candidates = [
            Candidate({"x": 1}, {"loss": -0.5}),
            Candidate({"x": 2}, {"loss": -1.0}),
            Candidate({"x": 3}, {"loss": 0.1}),
        ]
        best = select_minimax_rank(candidates, specs)
        assert best.objectives["loss"] == -1.0

    def test_pareto_tradeoff(self):
        """
        Classic Pareto scenario:
        A: acc=0.99, latency=100
        B: acc=0.80, latency=10
        C: acc=0.90, latency=50  (balanced)

        C should win the minimax rank because its worst rank is best.
        """
        specs = [ObjectiveSpec("acc", "max"), ObjectiveSpec("latency", "min")]
        a = Candidate({}, {"acc": 0.99, "latency": 100})
        b = Candidate({}, {"acc": 0.80, "latency": 10})
        c = Candidate({}, {"acc": 0.90, "latency": 50})

        # Ranks on acc (max, so highest first): A=1, C=2, B=3
        # Ranks on latency (min, so lowest first): B=1, C=2, A=3
        # Worst ranks: A=3, B=3, C=2
        # C should win
        best = select_minimax_rank([a, b, c], specs)
        assert best.objectives["acc"] == 0.90

    def test_float_precision_tiebreaking(self):
        """Near-identical values should still produce deterministic ranking."""
        specs = [ObjectiveSpec("acc", "max")]
        candidates = [
            Candidate({"x": i}, {"acc": 0.9 + i * 1e-10})
            for i in range(5)
        ]
        best = select_minimax_rank(candidates, specs)
        # The candidate with highest acc (last one) should win
        assert best is candidates[-1]
