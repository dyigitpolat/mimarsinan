"""The greedy engine's used cores by residency class, with an exact state index for the canonical predicate."""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Dict, List, Optional, Tuple

from mimarsinan.mapping.packing.canonical import CANONICAL_CORE_READS
from mimarsinan.mapping.packing.placement_cost import resident_bank_ids


def _hard_tg(hc: Any) -> Optional[int]:
    tg = getattr(hc, "residency_class_id", None)
    return int(tg) if tg is not None else None


#: The state index's group of every core (beside one group per weight bank): a key no bank id equals.
_EVERY = object()


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
    order selects exactly the core a scan over every core selects. [W04] It holds the same per weight bank among the
    cores that host the bank (``bank_hosts``): a bank-affine softcore's cost on a host reads the state alone too.
    """

    def __init__(self, used: List[Any], *, indexed: bool) -> None:
        self.used = used
        self.by_tg: Dict[Optional[int], List[int]] = {}
        self._tg_of: Dict[int, Optional[int]] = {}
        self._indexed = indexed
        self._seq = itertools.count()
        #: idx -> (class, state, seq, banks hosted at entry); every placement re-enters its core, so banks are current
        self._entry: Dict[int, Tuple[Optional[int], Tuple, int, frozenset]] = {}
        #: (group, class) -> state -> heap of (seq, idx); a group is ``_EVERY`` or one weight bank
        self._states: Dict[Tuple[Any, Optional[int]], Dict[Tuple, List[Tuple[int, int]]]] = {}
        self._live: Dict[Tuple[Any, Optional[int], Tuple], int] = {}
        for idx in range(len(used)):
            self.register(idx)

    @property
    def indexed(self) -> bool:
        return self._indexed

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
        seq = self._entry[idx][2]
        self._leave(idx)
        self._enter(idx, self._tg_of[idx], next(self._seq) if new_seq else seq)

    def _enter(self, idx: int, tg: Optional[int], seq: int) -> None:
        state = core_state(self.used[idx])
        banks = resident_bank_ids(self.used[idx])
        self._entry[idx] = (tg, state, seq, banks)
        for group in (_EVERY, *banks):
            heapq.heappush(self._states.setdefault((group, tg), {}).setdefault(state, []), (seq, idx))
            self._live[(group, tg, state)] = self._live.get((group, tg, state), 0) + 1

    def _leave(self, idx: int) -> None:
        tg, state, _, banks = self._entry.pop(idx)
        for group in (_EVERY, *banks):
            self._live[(group, tg, state)] -= 1
            if not self._live[(group, tg, state)]:
                del self._live[(group, tg, state)]
                del self._states[(group, tg)][state]

    def _holds(self, item: Tuple[int, int], group: Any, tg: Optional[int], state: Tuple) -> bool:
        entry = self._entry.get(item[1])
        return entry is not None and entry[:3] == (tg, state, item[0]) and (group is _EVERY or group in entry[3])

    def _first(self, group: Any, tg: Optional[int], state: Tuple) -> Tuple[int, int]:
        heap = self._states[(group, tg)][state]
        while not self._holds(heap[0], group, tg, state):
            heapq.heappop(heap)
        return heap[0]

    def _representatives(self, group: Any, core_tg: Optional[int]) -> List[int]:
        out: List[int] = []
        for key in ([core_tg, None] if core_tg is not None else [None]):
            firsts = sorted(self._first(group, key, state) for state in list(self._states.get((group, key), {})))
            out.extend(idx for _, idx in firsts)
        return out

    def scan_order(self, core_tg: Optional[int], *, every_core: bool) -> List[int]:
        """Candidate indices: the class's bucket, then the unclassed one (once when the softcore holds no class)."""
        if every_core or not self._indexed:
            keys = [core_tg, None] if core_tg is not None else [None]
            return [idx for key in keys for idx in self.by_tg.get(key, ())]
        return self._representatives(_EVERY, core_tg)

    def bank_hosts(self, core_tg: Optional[int], bank: Any) -> List[int]:
        """[W04] Indexed only: among the candidate buckets' cores that host ``bank``, one per state (its earliest), in
        candidate order; a scan over them selects the host a scan over every host selects."""
        return self._representatives(bank, core_tg)
