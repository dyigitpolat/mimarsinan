"""The pipeline endpoint-wall ledger: one wall budget shared by armed endpoints."""

from __future__ import annotations

WALL_CACHE_KEY = "__mbh_endpoint_wall_consumed"


def _cache_write(cache, value: float) -> None:
    # PipelineCache exposes ``add``; the test MockPipeline cache is a plain dict.
    add = getattr(cache, "add", None)
    if add is not None:
        add(WALL_CACHE_KEY, float(value))
    else:
        cache[WALL_CACHE_KEY] = float(value)


def consumed(pipeline) -> float:
    """Endpoint wall seconds consumed so far in this run (0.0 when none)."""
    value = pipeline.cache.get(WALL_CACHE_KEY)
    return 0.0 if value is None else float(value)


def consume(pipeline, seconds: float) -> float:
    """Add an armed endpoint stage's measured wall seconds; returns the total."""
    total = consumed(pipeline) + max(0.0, float(seconds))
    _cache_write(pipeline.cache, total)
    return total


def remaining(pipeline, total_budget: float) -> float:
    """Wall seconds the run's endpoint budget still affords (floored at 0)."""
    return max(0.0, float(total_budget) - consumed(pipeline))
