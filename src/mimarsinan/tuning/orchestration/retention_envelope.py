"""The pipeline retention-envelope SSOT: the incoming model's own clean accuracy.

The first seeded pipeline metric is the pretrained / float envelope — the best a
conversion can retain. Recorded ONCE (never a ratchet), read by the endpoint-
recovery target so an absolute floor can never demand more than the model can
give: ``target = max(highwater, min(floor, envelope))``. Resume-safe (a
run-scoped cache key, reset by a fresh run and kept by an explicit resume).
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration.run_ledger import (
    RETENTION_ENVELOPE_CACHE_KEY as RETENTION_ENVELOPE_CACHE_KEY,
    cache_write,
)


def peek(pipeline) -> float | None:
    """The recorded envelope, or None when nothing has been seeded yet."""
    value = pipeline.cache.get(RETENTION_ENVELOPE_CACHE_KEY)
    return None if value is None else float(value)


def seed(pipeline, envelope: float) -> float | None:
    """Record the incoming-model envelope, WRITE-ONCE. A second seed (a later,
    possibly higher or lower metric) never overwrites the fixed envelope;
    non-positive values are ignored (unseeded metrics). Returns the recorded
    envelope, or None if nothing has been recorded."""
    current = peek(pipeline)
    if current is not None:
        return current
    value = float(envelope)
    if value <= 0.0:
        return None
    cache_write(pipeline.cache, RETENTION_ENVELOPE_CACHE_KEY, value)
    return value
