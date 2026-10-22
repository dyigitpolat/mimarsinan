"""The greedy engine's unplaced softcores when it picks through the pick index: removal in O(1)."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class RemainingSoftcores:
    """The caller's softcore list behind ``remove`` (the first equal element, as ``list.remove``), ``append`` and
    ``len``; ``finish`` writes back the list those calls would have left.

    Equal elements share one hash (the hash contract), so the earliest live slot of an element's equality class is
    the element ``list.remove`` matches. A list holding an unhashable element is used as it is.
    """

    def __init__(self, items: List[Any]) -> None:
        self._items = items
        self._slots: Optional[List[Any]] = []
        self._alive: List[bool] = []
        self._classes: Dict[Any, Deque[int]] = {}
        self._removed = 0
        try:
            for item in items:
                self._push(item)
        except TypeError:
            self._slots = None

    def _push(self, item: Any) -> None:
        assert self._slots is not None
        self._classes.setdefault(item, deque()).append(len(self._slots))
        self._slots.append(item)
        self._alive.append(True)

    def __len__(self) -> int:
        return len(self._items) if self._slots is None else len(self._slots) - self._removed

    def remove(self, item: Any) -> None:
        if self._slots is None:
            self._items.remove(item)
            return
        slots = self._classes.get(item)
        if not slots:
            raise ValueError("list.remove(x): x not in list")
        self._alive[slots.popleft()] = False
        self._removed += 1

    def append(self, item: Any) -> None:
        if self._slots is not None:
            try:
                hash(item)
            except TypeError:
                self.finish()
                self._slots = None
        if self._slots is None:
            self._items.append(item)
        else:
            self._push(item)

    def finish(self) -> None:
        if self._slots is not None:
            self._items[:] = [item for item, alive in zip(self._slots, self._alive) if alive]
