"""Tests for validation-sized target decay and floor/cap behavior."""

import math

from mimarsinan.tuning.adaptation_target_adjuster import (
    AdaptationTargetAdjuster,
    target_decay_from_validation_samples,
)


def test_target_decay_clamped_and_monotonic():
    d_small = target_decay_from_validation_samples(100)
    d_large = target_decay_from_validation_samples(100_000)
    assert 0.95 <= d_small <= 0.999
    assert 0.95 <= d_large <= 0.999
    assert d_large >= d_small


def test_from_pipeline_matches_formula():
    from conftest import MockPipeline, default_config

    pipe = MockPipeline(config=default_config())
    adj = AdaptationTargetAdjuster.from_pipeline(0.9, pipe)
    dp = pipe.data_provider_factory.create()
    n = dp.get_validation_set_size()
    expected = target_decay_from_validation_samples(n)
    assert math.isclose(adj.decay, expected)


def test_floor_prevents_unbounded_decay():
    """Target must never drop below original * floor_ratio, even after many misses."""
    adj = AdaptationTargetAdjuster(0.95, decay=0.98, floor_ratio=0.90)
    for _ in range(100):
        adj.update_target(0.0)
    assert adj.get_target() >= 0.95 * 0.90 - 1e-9


def test_cap_prevents_overshoot_above_original():
    """Target must never grow above the original value, even after many hits."""
    adj = AdaptationTargetAdjuster(0.95, decay=0.98, floor_ratio=0.90)
    for _ in range(100):
        adj.update_target(1.0)
    assert adj.get_target() <= 0.95 + 1e-9


def test_growth_recovers_toward_original():
    """After decay, hitting the target should grow it back toward original."""
    adj = AdaptationTargetAdjuster(0.95, decay=0.98, floor_ratio=0.90)
    for _ in range(5):
        adj.update_target(0.0)
    decayed = adj.get_target()
    assert decayed < 0.95
    for _ in range(50):
        adj.update_target(1.0)
    assert adj.get_target() > decayed
    assert adj.get_target() <= 0.95 + 1e-9


def test_default_floor_ratio():
    """Default floor_ratio is 0.90."""
    adj = AdaptationTargetAdjuster(1.0)
    assert math.isclose(adj.floor, 0.90)


def test_from_pipeline_derives_floor_from_degradation_tolerance():
    from conftest import MockPipeline, default_config

    cfg = default_config()
    cfg["degradation_tolerance"] = 0.05
    pipe = MockPipeline(config=cfg)
    adj = AdaptationTargetAdjuster.from_pipeline(0.9, pipe)
    assert math.isclose(adj.floor, 0.9 * (1.0 - 0.05))
