"""Tests for search results: ObjectiveSpec, Candidate, SearchResult, and ranking."""

import pytest

from mimarsinan.search.results import (
    ObjectiveSpec, Candidate, SearchResult,
    select_minimax_rank,
)


class TestObjectiveSpec:
    def test_creation(self):
        o = ObjectiveSpec(name="accuracy", goal="max")
        assert o.name == "accuracy"
        assert o.goal == "max"

    def test_frozen(self):
        o = ObjectiveSpec(name="loss", goal="min")
        with pytest.raises(AttributeError):
            o.name = "other"


class TestCandidate:
    def test_creation(self):
        c = Candidate(configuration={"lr": 0.01}, objectives={"acc": 0.9})
        assert c.configuration["lr"] == 0.01
        assert c.objectives["acc"] == 0.9

    def test_metadata_defaults_empty(self):
        c = Candidate(configuration={}, objectives={})
        assert c.metadata == {}


class TestSearchResult:
    def test_creation(self):
        specs = [ObjectiveSpec("acc", "max")]
        best = Candidate({"x": 1}, {"acc": 0.95})
        r = SearchResult(objectives=specs, best=best)
        assert r.best.objectives["acc"] == 0.95
        assert r.pareto_front == []

    def test_with_pareto_front(self):
        specs = [ObjectiveSpec("acc", "max"), ObjectiveSpec("size", "min")]
        c1 = Candidate({"x": 1}, {"acc": 0.9, "size": 10})
        c2 = Candidate({"x": 2}, {"acc": 0.85, "size": 5})
        r = SearchResult(objectives=specs, best=c1, pareto_front=[c1, c2])
        assert len(r.pareto_front) == 2


class TestSelectMinimaxRank:
    def test_single_objective_max(self):
        specs = [ObjectiveSpec("acc", "max")]
        candidates = [
            Candidate({"x": 1}, {"acc": 0.8}),
            Candidate({"x": 2}, {"acc": 0.95}),
            Candidate({"x": 3}, {"acc": 0.7}),
        ]
        best = select_minimax_rank(candidates, specs)
        assert best.objectives["acc"] == 0.95

    def test_single_objective_min(self):
        specs = [ObjectiveSpec("loss", "min")]
        candidates = [
            Candidate({"x": 1}, {"loss": 0.5}),
            Candidate({"x": 2}, {"loss": 0.1}),
        ]
        best = select_minimax_rank(candidates, specs)
        assert best.objectives["loss"] == 0.1

    def test_multi_objective_minimax_rank(self):
        specs = [ObjectiveSpec("acc", "max"), ObjectiveSpec("params", "min")]
        candidates = [
            Candidate({"x": 1}, {"acc": 0.9, "params": 100}),
            Candidate({"x": 2}, {"acc": 0.8, "params": 50}),
            Candidate({"x": 3}, {"acc": 0.95, "params": 200}),
        ]
        best = select_minimax_rank(candidates, specs)
        assert best is not None

    def test_empty_candidates_returns_none(self):
        specs = [ObjectiveSpec("acc", "max")]
        result = select_minimax_rank([], specs)
        assert result is None

    def test_single_candidate(self):
        specs = [ObjectiveSpec("acc", "max")]
        c = Candidate({"x": 1}, {"acc": 0.9})
        best = select_minimax_rank([c], specs)
        assert best is c

    def test_tie_broken_by_rank_sum(self):
        specs = [ObjectiveSpec("a", "max"), ObjectiveSpec("b", "max")]
        c1 = Candidate({"x": 1}, {"a": 0.9, "b": 0.5})
        c2 = Candidate({"x": 2}, {"a": 0.5, "b": 0.9})
        c3 = Candidate({"x": 3}, {"a": 0.8, "b": 0.8})
        best = select_minimax_rank([c1, c2, c3], specs)
        assert best.configuration["x"] == 3
