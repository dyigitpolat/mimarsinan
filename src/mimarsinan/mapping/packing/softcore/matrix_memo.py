"""Content-keyed memo for resolved core weight grids.

Placement descriptors make ``resolve_core_matrix`` a pure function of the
core's content key, so equal keys may share ONE array. On the real vehicle
4,925 cores resolve to 27 distinct grids: without this memo every consumer
(GUI walk, uploads, exporters) re-materialized 92.9 GB per sweep.

The memo is bounded (LRU by resolved bytes) and every eviction is announced —
a silent cap would read as "sharing works" while the cost quietly returned.
Payload identity is re-checked on every hit, because content keys embed
payload ``id``s and a freed array's id can be re-used.

Contract: a resolved grid MAY be shared, so callers must not mutate it. This
is not new — the exact-fit plain path already returned the shared payload.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

from mimarsinan.mapping.packing.softcore.matrix_placement import (
    core_matrix_content_key,
    core_matrix_payloads,
    resolve_core_matrix,
    same_core_matrix_payloads,
)

__all__ = [
    "matrix_memo_stats",
    "memoized_resolve_core_matrix",
    "reset_matrix_memo",
]

_DEFAULT_BUDGET_BYTES = 2 * 1024 ** 3

_entries: "OrderedDict[Any, tuple]" = OrderedDict()
_budget: int = _DEFAULT_BUDGET_BYTES
_bytes: int = 0
_stats: Dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}


def reset_matrix_memo(*, budget_bytes: "int | None" = None) -> None:
    """Drop every cached grid and restore the budget (tests, long runs).

    A reset restores EVERYTHING, budget included: leaving a previously narrowed
    budget in place made one caller's temporary cap silently permanent for the
    rest of the process, and every later resolution stopped sharing while
    reading as if it did. Pass ``budget_bytes`` to set a different one.
    """
    global _budget, _bytes
    _entries.clear()
    _bytes = 0
    for key in _stats:
        _stats[key] = 0
    _budget = _DEFAULT_BUDGET_BYTES if budget_bytes is None else int(budget_bytes)


def matrix_memo_stats() -> Dict[str, int]:
    return {**_stats, "entries": len(_entries), "bytes": _bytes}


def _evict_to_budget() -> None:
    global _bytes
    while _entries and _bytes > _budget:
        _key, (_payloads, array) = _entries.popitem(last=False)
        _bytes -= int(getattr(array, "nbytes", 0))
        _stats["evictions"] += 1
        if _stats["evictions"] in (1, 10, 100) or _stats["evictions"] % 1000 == 0:
            print(
                f"[MatrixMemo] evicted {_stats['evictions']} grid(s): resolved "
                f"working set exceeds {_budget / 1e9:.2f} GB — cores are "
                f"re-materializing; raise the budget to restore sharing.",
                flush=True,
            )


def memoized_resolve_core_matrix(
    owned, placements, axons_per_core: int, neurons_per_core: int, owner: str,
):
    """``resolve_core_matrix`` with content-keyed sharing."""
    if owned is not None:
        return owned                      # already a stable shared object
    if not placements:
        return resolve_core_matrix(
            owned, placements, axons_per_core, neurons_per_core, owner,
        )
    key = core_matrix_content_key(
        owned, placements, axons_per_core, neurons_per_core,
    )
    payloads = core_matrix_payloads(owned, placements)
    hit = _entries.get(key)
    if hit is not None and same_core_matrix_payloads(hit[0], payloads):
        _entries.move_to_end(key)
        _stats["hits"] += 1
        return hit[1]
    array = resolve_core_matrix(
        owned, placements, axons_per_core, neurons_per_core, owner,
    )
    _stats["misses"] += 1
    global _bytes
    if hit is not None:
        _bytes -= int(getattr(hit[1], "nbytes", 0))
    _entries[key] = (payloads, array)
    _entries.move_to_end(key)
    _bytes += int(getattr(array, "nbytes", 0))
    _evict_to_budget()
    return array
