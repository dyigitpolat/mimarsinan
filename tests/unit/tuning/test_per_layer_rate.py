"""Tests for the optional per-perceptron rate schedule.

Default behaviour: scalar rate applies uniformly to every perceptron.
When ``config["per_layer_rate_schedule"] = True``, the same scalar ``rate``
from the orchestration loop maps to a *per-perceptron* schedule; each
perceptron can lag or lead, but the endpoint invariant holds — when the
scalar reaches 1.0, every perceptron's effective rate is 1.0 as well.
"""

import pytest

from mimarsinan.tuning.per_layer_schedule import (
    LinearPerLayerSchedule,
    build_per_layer_schedule,
    uniform_rate_fn,
)


class _StubPerceptron:
    def __init__(self, name):
        self.name = name


def _perceptrons(names):
    return [_StubPerceptron(n) for n in names]


class TestUniformRate:
    def test_uniform_applies_same_rate_to_all(self):
        ps = _perceptrons(["a", "b", "c"])
        fn = uniform_rate_fn(0.5)
        assert [fn(p) for p in ps] == [0.5, 0.5, 0.5]

    def test_uniform_endpoint(self):
        ps = _perceptrons(["a", "b", "c"])
        fn = uniform_rate_fn(1.0)
        assert [fn(p) for p in ps] == [1.0, 1.0, 1.0]


class TestLinearPerLayerSchedule:
    def test_respects_sensitivity_ordering_at_middle_rate(self):
        ps = _perceptrons(["low", "mid", "high"])
        # sensitivity: higher sensitivity -> slower adaptation (lower per-layer rate)
        sensitivity = {"low": 0.0, "mid": 0.5, "high": 1.0}
        schedule = LinearPerLayerSchedule(ps, sensitivity)
        r_fn = schedule.rate_fn(0.5)
        lows = r_fn(ps[0])
        mids = r_fn(ps[1])
        highs = r_fn(ps[2])
        assert lows >= mids >= highs

    def test_endpoint_invariant_reaches_one(self):
        ps = _perceptrons(["a", "b", "c"])
        sensitivity = {"a": 0.0, "b": 0.5, "c": 1.0}
        schedule = LinearPerLayerSchedule(ps, sensitivity)
        r_fn = schedule.rate_fn(1.0)
        for p in ps:
            assert r_fn(p) == pytest.approx(1.0), (
                f"At scalar rate=1.0 every perceptron must reach 1.0 "
                f"({p.name} -> {r_fn(p)})"
            )

    def test_start_invariant_zero_means_zero(self):
        ps = _perceptrons(["a", "b", "c"])
        sensitivity = {"a": 0.0, "b": 0.5, "c": 1.0}
        schedule = LinearPerLayerSchedule(ps, sensitivity)
        r_fn = schedule.rate_fn(0.0)
        for p in ps:
            assert r_fn(p) == pytest.approx(0.0)

    def test_clamped_into_unit_interval(self):
        ps = _perceptrons(["a", "b"])
        sensitivity = {"a": 0.0, "b": 1.0}
        schedule = LinearPerLayerSchedule(ps, sensitivity)
        r_fn = schedule.rate_fn(0.5)
        for p in ps:
            assert 0.0 <= r_fn(p) <= 1.0


class TestBuildPerLayerSchedule:
    def test_default_returns_uniform(self):
        ps = _perceptrons(["a", "b"])
        fn_factory = build_per_layer_schedule(config={}, perceptrons=ps, sensitivities=None)
        r_fn = fn_factory(0.7)
        for p in ps:
            assert r_fn(p) == pytest.approx(0.7)

    def test_explicit_opt_in_uses_per_layer(self):
        ps = _perceptrons(["easy", "hard"])
        sensitivities = {"easy": 0.0, "hard": 1.0}
        fn_factory = build_per_layer_schedule(
            config={"per_layer_rate_schedule": True},
            perceptrons=ps,
            sensitivities=sensitivities,
        )
        r_fn = fn_factory(0.5)
        assert r_fn(ps[0]) >= r_fn(ps[1])
        # Endpoint invariant preserved
        r_fn_one = fn_factory(1.0)
        for p in ps:
            assert r_fn_one(p) == pytest.approx(1.0)

    def test_opt_in_without_sensitivities_falls_back_to_uniform(self):
        ps = _perceptrons(["a", "b"])
        fn_factory = build_per_layer_schedule(
            config={"per_layer_rate_schedule": True},
            perceptrons=ps,
            sensitivities=None,
        )
        r_fn = fn_factory(0.5)
        for p in ps:
            assert r_fn(p) == pytest.approx(0.5)
