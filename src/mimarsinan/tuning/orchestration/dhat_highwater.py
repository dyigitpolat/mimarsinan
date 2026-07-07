"""The pipeline D-hat high-water SSOT: the running max deployed full-transform accuracy."""

from __future__ import annotations

from mimarsinan.tuning.orchestration.run_ledger import (
    HIGHWATER_CACHE_KEY as HIGHWATER_CACHE_KEY,
    cache_write,
)


def peek(pipeline) -> float | None:
    """The current high-water mark, or None when no gate probe has written one."""
    value = pipeline.cache.get(HIGHWATER_CACHE_KEY)
    return None if value is None else float(value)


def observe(pipeline, dhat: float) -> float:
    """Ratchet the high-water mark with one deployed full-transform read.

    Written by the gate's probes (and the endpoint stage's exit read); the mark
    only ever rises. Returns the mark after the observation.
    """
    current = peek(pipeline)
    value = float(dhat)
    if current is None or value > current:
        cache_write(pipeline.cache, HIGHWATER_CACHE_KEY, value)
        return value
    return current


def require(pipeline) -> float:
    """The high-water mark, failing loud when absent.

    The endpoint-recovery target anchors here, never on a damaged local
    baseline; absence means no D-hat-gated ladder ran before the endpoint —
    a pipeline wiring defect, not a degradable condition.
    """
    value = peek(pipeline)
    if value is None:
        raise RuntimeError(
            "The pipeline D-hat high-water mark is absent: no [MBH-GATE] probe "
            "has run before this endpoint-recovery stage. The gated fast ladder "
            "writes it; check the tuner ordering / optimization driver."
        )
    return value
