"""Shared upstream-neural ID walks for latency engines."""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol


class _SourceLike(Protocol):
    def is_off(self) -> bool: ...
    node_id: int


def iter_upstream_neural_ids(
    sources: Iterable[_SourceLike],
    *,
    skip_off: bool = True,
) -> Iterator[int]:
    """Yield distinct upstream ``node_id`` values from flat source lists."""
    seen: set[int] = set()
    for src in sources:
        if skip_off and src.is_off():
            continue
        nid = int(src.node_id)
        if nid >= 0 and nid not in seen:
            seen.add(nid)
            yield nid
