"""Per-instance derived-value memos for unhashable hosts (id-keyed, finalizer-evicted)."""

from __future__ import annotations

import weakref
from typing import Any, Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class InstanceMemo(Generic[T]):
    """One derived value per live host instance.

    Keyed by id(host): IR/segment dataclasses are unhashable (eq=True); the
    finalizer evicts the entry during the host's dealloc, before id reuse.
    """

    def __init__(self) -> None:
        self._memo: Dict[int, T] = {}

    def get(self, host: Any, build: Callable[[Any], T]) -> T:
        key = id(host)
        if key in self._memo:
            return self._memo[key]
        value = build(host)
        self._memo[key] = value
        weakref.finalize(host, self._memo.pop, key, None)
        return value

    def __len__(self) -> int:
        return len(self._memo)


_DEFAULT: InstanceMemo[Any] = InstanceMemo()


def instance_memo(host: Any, build: Callable[[Any], T]) -> T:
    """Module-shared convenience memo; prefer a dedicated ``InstanceMemo`` per kind."""
    return _DEFAULT.get(host, build)


def memo_size() -> int:
    """Number of live entries in the shared memo (test observability)."""
    return len(_DEFAULT)
