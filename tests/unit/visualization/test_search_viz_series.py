"""Unit tests for the shared search-viz series helpers."""

import math

from mimarsinan.visualization.search_viz.series import (
    PENALTY_CUTOFF,
    best_metric_series,
    finite_pairs,
    goal_by_metric,
    nan_gapped,
    pareto_metric_series,
)


class TestGoalByMetric:
    def test_maps_named_objectives_to_goals(self) -> None:
        result = {"objectives": [
            {"name": "accuracy", "goal": "max"},
            {"name": "cores", "goal": "min"},
        ]}
        assert goal_by_metric(result) == {"accuracy": "max", "cores": "min"}

    def test_skips_unnamed_and_non_dict_objectives(self) -> None:
        result = {"objectives": [
            {"goal": "max"},
            {"name": None, "goal": "min"},
            "not-a-dict",
            {"name": "cores", "goal": "min"},
        ]}
        assert goal_by_metric(result) == {"cores": "min"}

    def test_empty_or_missing_objectives(self) -> None:
        assert goal_by_metric({}) == {}
        assert goal_by_metric({"objectives": None}) == {}

    def test_preserves_objective_order(self) -> None:
        result = {"objectives": [{"name": n, "goal": "max"} for n in "cab"]}
        assert list(goal_by_metric(result).keys()) == ["c", "a", "b"]


class TestBestMetricSeries:
    def test_extracts_floats_and_none_for_missing(self) -> None:
        bests = [{"acc": 0.5}, {}, {"acc": "0.75"}, {"acc": "bad"}]
        assert best_metric_series(bests, "acc") == [0.5, None, 0.75, None]


class TestParetoMetricSeries:
    def test_penalty_and_missing_become_none(self) -> None:
        pareto = [
            {"objectives": {"acc": 0.9}},
            {"objectives": {"acc": PENALTY_CUTOFF * 2}},
            {"objectives": {}},
            "not-a-dict",
            {"objectives": None},
        ]
        assert pareto_metric_series(pareto, "acc") == [0.9, None, None, None, None]


class TestFinitePairs:
    def test_drops_pairs_with_any_none(self) -> None:
        xs, ys = finite_pairs([1.0, None, 3.0, 4.0], [10.0, 20.0, None, 40.0])
        assert xs == [1.0, 4.0]
        assert ys == [10.0, 40.0]

    def test_empty_input(self) -> None:
        assert finite_pairs([], []) == ([], [])


class TestNanGapped:
    def test_none_becomes_nan_values_pass_through(self) -> None:
        out = nan_gapped([1.0, None, 2.5])
        assert out[0] == 1.0
        assert math.isnan(out[1])
        assert out[2] == 2.5
        assert all(isinstance(v, float) for v in out)
