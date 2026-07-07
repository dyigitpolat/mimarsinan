"""The pipeline endpoint-step ledger: one training-step budget shared by armed endpoints."""

from __future__ import annotations

STEPS_CACHE_KEY = "__mbh_endpoint_steps_consumed"


def _cache_write(cache, value: int) -> None:
    # PipelineCache exposes ``add``; the test MockPipeline cache is a plain dict.
    add = getattr(cache, "add", None)
    if add is not None:
        add(STEPS_CACHE_KEY, int(value))
    else:
        cache[STEPS_CACHE_KEY] = int(value)


def consumed(pipeline) -> int:
    """Endpoint training steps consumed so far in this run (0 when none)."""
    value = pipeline.cache.get(STEPS_CACHE_KEY)
    return 0 if value is None else int(value)


def consume(pipeline, steps: int) -> int:
    """Add an armed endpoint stage's trained step count; returns the total."""
    total = consumed(pipeline) + max(0, int(steps))
    _cache_write(pipeline.cache, total)
    return total


def remaining(pipeline, total_budget: int) -> int:
    """Training steps the run's endpoint budget still affords (floored at 0)."""
    return max(0, int(total_budget) - consumed(pipeline))
