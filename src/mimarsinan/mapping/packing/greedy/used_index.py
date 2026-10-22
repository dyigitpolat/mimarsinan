"""The greedy engine's used cores by residency class, with an exact state index for the canonical predicate."""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Dict, List, Optional, Tuple

from mimarsinan.mapping.packing.canonical import CANONICAL_CORE_READS


def _hard_tg(hc: Any) -> Optional[int]:
    tg = getattr(hc, "residency_class_id", None)
    return int(tg) if tg is not None else None


def core_state(hc: Any) -> Tuple:
    """Everything the canonical predicate, the bankless best-fit cost and the split into a used core read of a core
    (``CANONICAL_CORE_READS`` and its shape): two cores of one state answer every softcore without a bank alike."""
    return tuple(getattr(hc, name, None) for name in CANONICAL_CORE_READS) + (
        int(hc.get_input_count()), int(hc.get_output_count()))


class UsedBuckets:
    """Used-core indices by the residency class each core holds, in the order cores enter a class.

    ``scan_order(tg)`` is the engine's candidate order (the class's bucket, then the unclassed bucket). With
    ``indexed`` it holds, per bucket, one representative per core state (the state's earliest core): the
    canonical predicate and the best-fit cost read only the state, so a scan over the representatives in candidate
    order selects exactly the core a scan over every core selects.
    """

    def __init__(self, used: List[Any], *, indexed: bool) -> None:
        self.used = used
        self.by_tg: Dict[Optional[int], List[int]] = {}
        self._tg_of: Dict[int, Optional[int]] = {}
        self._indexed = indexed
        self._seq = itertools.count()
        self._entry: Dict[int, Tuple[Optional[int], Tuple, int]] = {}
        self._states: Dict[Optional[int], Dict[Tuple, List[Tuple[int, int]]]] = {}
        self._live: Dict[Tuple[Optional[int], Tuple], int] = {}
        for idx in range(len(used)):
            self.register(idx)

    def register(self, idx: int) -> None:
        tg = _hard_tg(self.used[idx])
        self.by_tg.setdefault(tg, []).append(idx)
        self._tg_of[idx] = tg
        if self._indexed:
            self._enter(idx, tg, next(self._seq))

    def reclass(self, idx: int) -> None:
        """After a placement: a core whose class changed moves to the end of its new class's bucket."""
        tg, old = _hard_tg(self.used[idx]), self._tg_of.get(idx)
        if idx not in self._tg_of:
            return
        if tg != old:
            self.by_tg[old].remove(idx)
            self.by_tg.setdefault(tg, []).append(idx)
            self._tg_of[idx] = tg
        self.restate(idx, new_seq=tg != old)

    def restate(self, idx: int, *, new_seq: bool = False) -> None:
        """After a placement that keeps the core in its bucket (its state alone changed)."""
        if not self._indexed or idx not in self._entry:
            return
        tg, _, seq = self._entry[idx]
        self._leave(idx)
        self._enter(idx, self._tg_of[idx], next(self._seq) if new_seq else seq)

    def _enter(self, idx: int, tg: Optional[int], seq: int) -> None:
        state = core_state(self.used[idx])
        self._entry[idx] = (tg, state, seq)
        heapq.heappush(self._states.setdefault(tg, {}).setdefault(state, []), (seq, idx))
        self._live[(tg, state)] = self._live.get((tg, state), 0) + 1

    def _leave(self, idx: int) -> None:
        tg, state, _ = self._entry.pop(idx)
        self._live[(tg, state)] -= 1
        if not self._live[(tg, state)]:
            del self._live[(tg, state)]
            del self._states[tg][state]

    def _first(self, tg: Optional[int], state: Tuple) -> Tuple[int, int]:
        heap = self._states[tg][state]
        while self._entry.get(heap[0][1]) != (tg, state, heap[0][0]):
            heapq.heappop(heap)
        return heap[0]

    def scan_order(self, core_tg: Optional[int], *, every_core: bool) -> List[int]:
        """Candidate indices: the class's bucket, then the unclassed one (once when the softcore holds no class)."""
        keys = [core_tg, None] if core_tg is not None else [None]
        if every_core or not self._indexed:
            return [idx for key in keys for idx in self.by_tg.get(key, ())]
        out: List[int] = []
        for key in keys:
            firsts = sorted(self._first(key, state) for state in list(self._states.get(key, {})))
            out.extend(idx for _, idx in firsts)
        return out
