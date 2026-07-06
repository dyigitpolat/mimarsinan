"""Incremental argmax index reproducing ``pick_best_softcore`` exactly in O(log n)."""

from __future__ import annotations

import heapq
import itertools
from typing import Generic, Iterable, List, Tuple

from mimarsinan.mapping.packing.canonical import SoftT


class PickBestIndex(Generic[SoftT]):
    """Exact-equivalent accelerator for ``pick_best_softcore``'s two max() scans.

    Heap order is (-count, seq): equal counts resolve to the smallest insertion
    sequence, which is max()'s first-occurrence tie-break because list order is
    insertion order (removals preserve relative order, splits append at the end).
    Each heap entry carries its core (a strong reference), so an id() cannot be
    reused while any heap still holds a lazily-deleted entry for it.
    """

    def __init__(self, cores: Iterable[SoftT] = ()) -> None:
        self._seq = itertools.count()
        self._alive: dict[int, SoftT] = {}
        self._by_input: List[Tuple[int, int, int, SoftT]] = []
        self._by_output: List[Tuple[int, int, int, SoftT]] = []
        for core in cores:
            self.add(core)

    def add(self, core: SoftT) -> None:
        seq = next(self._seq)
        key = id(core)
        self._alive[key] = core
        heapq.heappush(self._by_input, (-int(core.get_input_count()), seq, key, core))
        heapq.heappush(self._by_output, (-int(core.get_output_count()), seq, key, core))

    def discard(self, core: SoftT) -> None:
        self._alive.pop(id(core), None)

    def _peek(self, heap: List[Tuple[int, int, int, SoftT]]) -> SoftT:
        while heap:
            entry = heap[0]
            if self._alive.get(entry[2]) is entry[3]:
                return entry[3]
            heapq.heappop(heap)
        raise ValueError("unmapped_cores is empty")

    def pick(self) -> SoftT:
        """Replicates ``pick_best_softcore`` bit-for-bit, including tie-breaks."""
        core_a = self._peek(self._by_input)
        core_b = self._peek(self._by_output)
        if core_a.get_input_count() > core_b.get_output_count():
            return core_a
        return core_b
