"""The run-scoped ledger SSOT: the run-scoped cache keys and their one lifecycle."""

from __future__ import annotations

HIGHWATER_CACHE_KEY = "__mbh_dhat_highwater"
ENDPOINT_STEPS_CACHE_KEY = "__mbh_endpoint_steps_consumed"

RUN_SCOPED_KEYS = (HIGHWATER_CACHE_KEY, ENDPOINT_STEPS_CACHE_KEY)
"""Every run-scoped ledger key: reset by a fresh ``pipeline.run()``, kept by an
explicit resume, and snapshot/restored around each independent conversion draw."""


def cache_write(cache, key: str, value) -> None:
    """Write through ``add`` when the cache exposes it (PipelineCache), else item-set."""
    add = getattr(cache, "add", None)
    if add is not None:
        add(key, value)
    else:
        cache[key] = value


def cache_remove(cache, key: str) -> None:
    """Remove through ``remove`` when the cache exposes it, else ``pop``; idempotent."""
    remove = getattr(cache, "remove", None)
    if remove is not None:
        remove(key)
    else:
        cache.pop(key, None)


def reset(cache) -> None:
    """Clear every run-scoped ledger (a fresh run must not inherit a previous
    attempt's consumption from a reused cache directory)."""
    for key in RUN_SCOPED_KEYS:
        cache_remove(cache, key)


def snapshot(cache) -> dict:
    """The current run-scoped ledger state (``None`` marks an absent key)."""
    return {key: cache.get(key) for key in RUN_SCOPED_KEYS}


def restore(cache, snapshot: dict) -> None:
    """Restore a :func:`snapshot`, removing keys the snapshot holds as absent."""
    for key, value in snapshot.items():
        if value is None:
            cache_remove(cache, key)
        else:
            cache_write(cache, key, value)
