"""``pack_layout``'s verdict on a pool of one core shape, decomposed by residency class: exact, and cheap to extend.

On a pool of one core shape (A, N) the greedy engine's choices for a softcore read only the cores of its residency
class: a layout core takes the class of its first softcore, the candidate scans (best fit, split into a used core)
read the class's cores only, and every fresh core is the same core. The engine's order restricted to one class is
the order the class alone gives: without neuron splitting it pops the softcores sorted by max(axons, neurons), with
it the pick takes the softcore of greatest (max(a, n), n == max(a, n), list position) and a split appends its
remainder behind every softcore of its class. So each class's run on an unbounded pool consumes a fixed number of
pool cores, its demand, and a run on ``count`` cores follows the unbounded run until the step whose demand the pool
cannot meet, where it raises. A batch therefore packs iff no class's run raises and the class demands sum to at most
``count``.

A class's demand is a function of its type sequence: (axons, neurons, latency, bank, grouped) per softcore in list
order, the latencies and the banks relabelled by first occurrence (the engine compares them only for equality and
None: a bank-affine softcore prefers a core of its class holding its bank), and whether the softcore belongs to a
coalescing group (never neuron-split). [W04] A wide softcore is one unit, fused whole as the hard-core packer fuses
it. It is measured by ``pack_layout`` itself on an unbounded pool and memoized, per packer, by the sequence's
run-length encoding. A class of one type repeated k times reads a prefix table: the run of k softcores is the first k
of the run of K >= k (until a neuron split, whose remainder the k-run would place earlier), so one run of K gives
every demand below it.
"""

from __future__ import annotations

import functools
import numbers
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout import layout_packer
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutPackingResult, LayoutSoftCoreSpec

#: The demand of a class whose run raises on an unbounded pool (above any core count).
UNPLACEABLE = 1 << 62
#: The core count of the unbounded pool a class's demand is measured on.
_UNBOUNDED = 1 << 40
#: Memoized class demands kept per pool before the memo is cleared (type tables are kept).
_MEMO_LIMIT = 1 << 16

#: One softcore of a class's type sequence: (axons, neurons, latency label, bank label, in a coalescing group).
SpecType = Tuple[int, int, Optional[int], Optional[int], bool]
#: A class's type sequence, run-length encoded.
Runs = Tuple[Tuple[SpecType, int], ...]


@dataclass(frozen=True)
class _PrefixTable:
    """``values[k]``: the demand of k softcores of one type, for every k the measured run decides."""

    values: Tuple[int, ...]
    growable: bool


def _integral(value) -> bool:
    return type(value) is int or isinstance(value, numbers.Integral)


def _admitted(sc) -> bool:
    """A plain layout spec of integral counts and class: every attribute the engine reads is one this module keys."""
    return (type(sc) is LayoutSoftCoreSpec and _integral(sc.input_count) and _integral(sc.output_count)
            and int(sc.input_count) >= 0 and int(sc.output_count) >= 0 and _integral(sc.residency_class_id)
            and (sc.latency_tag is None or _integral(sc.latency_tag))
            and (sc.bank_id is None or _integral(sc.bank_id))
            and (sc.coalescing_group_id is None or _integral(sc.coalescing_group_id)))


@dataclass
class _ClassRuns:
    runs: Runs = ()
    labels: Tuple[int, ...] = ()
    demand: int = 0
    banks: Tuple[int, ...] = ()

    def extended(self, items: Sequence[LayoutSoftCoreSpec]) -> Tuple[Runs, Tuple[int, ...], Tuple[int, ...]]:
        runs, labels, banks = list(self.runs), list(self.labels), list(self.banks)
        for sc in items:
            label = bank = None
            if sc.latency_tag is not None:
                latency = int(sc.latency_tag)
                if latency not in labels:
                    labels.append(latency)
                label = labels.index(latency)
            if sc.bank_id is not None:
                if int(sc.bank_id) not in banks:
                    banks.append(int(sc.bank_id))
                bank = banks.index(int(sc.bank_id))
            t = (int(sc.input_count), int(sc.output_count), label, bank, sc.coalescing_group_id is not None)
            if runs and runs[-1][0] == t:
                runs[-1] = (t, runs[-1][1] + 1)
            else:
                runs.append((t, 1))
        return tuple(runs), tuple(labels), tuple(banks)


def _by_class(batch: Sequence[LayoutSoftCoreSpec]) -> Dict[int, List[LayoutSoftCoreSpec]]:
    groups: Dict[int, List[LayoutSoftCoreSpec]] = {}
    for sc in batch:
        groups.setdefault(int(sc.residency_class_id), []).append(sc)
    return groups


class ClassDemand:
    """The class demands on one pool shape (A, N) under one set of packing flags and one packer, memoized."""

    def __init__(self, shape: Tuple[int, int], widest_axons: int, *, allow_coalescing: bool, allow_splitting: bool,
                 pack: Callable[..., LayoutPackingResult]) -> None:
        self.shape = shape
        self._pack = pack
        self.allow_coalescing = allow_coalescing
        self.allow_splitting = allow_splitting
        # The widest declared type, cores or none, stays declared as on the pool (it holds no core to take).
        self._types = [LayoutHardCoreType(shape[0], shape[1], _UNBOUNDED),
                       LayoutHardCoreType(widest_axons, shape[1], 0)]
        self._memo: Dict[Runs, int] = {}
        self._tables: Dict[SpecType, _PrefixTable] = {}

    def _run(self, runs: Runs, *, placements: bool):
        specs = [LayoutSoftCoreSpec(a, n, residency_class_id=0, latency_tag=lat, bank_id=bank,
                                    coalescing_group_id=0 if grouped else None)
                 for (a, n, lat, bank, grouped), k in runs for _ in range(k)]
        return self._pack(softcores=specs, core_types=self._types, allow_neuron_splitting=self.allow_splitting,
                          allow_coalescing=self.allow_coalescing, collect_placements=placements)

    def _consumed(self, snapshots) -> List[int]:
        """Pool cores per used core: a fused core consumed its components."""
        return [int(s.axons_per_core) // self.shape[0] for s in snapshots]

    def _table(self, t: SpecType, k: int) -> _PrefixTable:
        table = self._tables.get(t)
        if table is not None and (k < len(table.values) or not table.growable):
            return table
        size = max(k, 2 * (len(table.values) - 1) if table is not None else 0, 32)
        result = self._run(((t, size),), placements=True)
        table = self._prefix(result, size, 1) if result.feasible else _PrefixTable((0,), False)
        self._tables[t] = table
        return table

    def _prefix(self, result, size: int, q: int) -> _PrefixTable:
        """Demands of the first k softcores of a run of ``size``: the cores its first k * q steps opened, for every k
        before the first softcore a neuron split reached (a split's remainder is placed after every softcore)."""
        placements = result.placements or ()
        split_origins = [origin for origin, _ in placements[size * q:]]
        valid = min(split_origins) if split_origins else size
        consumed = self._consumed(result.used_core_snapshots)
        cumulative, total = [], 0
        for c in consumed:
            total += c
            cumulative.append(total)
        values, top = [0], -1
        for k in range(1, valid + 1):
            for _, core in placements[(k - 1) * q:k * q]:
                top = max(top, core)
            values.append(cumulative[top] if top >= 0 else 0)
        return _PrefixTable(values=tuple(values), growable=valid == size)

    def demand(self, runs: Runs) -> int:
        """Pool cores the run of this type sequence consumes on an unbounded pool (``UNPLACEABLE`` if it raises)."""
        if not runs:
            return 0
        if len(runs) == 1:
            t, k = runs[0]
            table = self._table(t, k)
            if k < len(table.values):
                return table.values[k]
        value = self._memo.get(runs)
        if value is None:
            if len(self._memo) >= _MEMO_LIMIT:
                self._memo.clear()
            result = self._run(runs, placements=False)
            value = sum(self._consumed(result.used_core_snapshots)) if result.feasible else UNPLACEABLE
            self._memo[runs] = value
        return value


@functools.lru_cache(maxsize=64)
def _shared(shape: Tuple[int, int], widest_axons: int, allow_coalescing: bool, allow_splitting: bool,
            pack: Callable[..., LayoutPackingResult]) -> ClassDemand:
    """One memo per pool shape, flags and packer (a replaced ``pack_layout`` measures afresh)."""
    return ClassDemand(shape, widest_axons, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                       pack=pack)


class PoolDemand:
    """``pack_layout``'s exact verdict on one declared pool of one core shape, by class demands."""

    def __init__(self, classes: ClassDemand, count: int) -> None:
        self.classes = classes
        self.count = count

    @classmethod
    def for_pool(cls, softcores: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
                 allow_coalescing: bool, allow_splitting: bool) -> Optional["PoolDemand"]:
        """None unless the pool holds one core shape and every softcore is a distinct plain layout spec (an object
        listed twice is one softcore placed twice once ``pack_layout`` copies the batch)."""
        shapes = {(int(t.max_axons), int(t.max_neurons)) for t in core_types if int(t.count) > 0}
        if len(shapes) != 1:
            return None
        (A, N), = shapes
        if A <= 0 or N <= 0 or len({id(sc) for sc in softcores}) != len(softcores):
            return None
        if not all(_admitted(sc) for sc in softcores):
            return None
        count = sum(int(t.count) for t in core_types if int(t.count) > 0)
        widest = max(int(t.max_axons) for t in core_types)
        return cls(_shared((A, N), widest, bool(allow_coalescing), bool(allow_splitting), layout_packer.pack_layout),
                   count)

    def feasible(self, batch: Sequence[LayoutSoftCoreSpec]) -> bool:
        """``pack_layout(batch).feasible`` on this pool."""
        total = 0
        for items in _by_class(batch).values():
            total += self.classes.demand(_ClassRuns().extended(items)[0])
            if total > self.count:
                return False
        return True

    def running(self) -> "RunningDemand":
        return RunningDemand(self)


class RunningDemand:
    """The class demands of a growing batch (the split's running sub-segment), extended a bundle at a time.

    ``take`` and ``reset`` only queue softcores (a certified extension needs no demand); ``extends`` brings the
    demands up to date, then answers for the batch plus a bundle and keeps the bundle when it packs.
    """

    def __init__(self, pool: PoolDemand) -> None:
        self.pool = pool
        self._classes: Dict[int, _ClassRuns] = {}
        self._total = 0
        self._pending: List[LayoutSoftCoreSpec] = []

    def reset(self, batch: Sequence[LayoutSoftCoreSpec]) -> None:
        self._classes, self._total, self._pending = {}, 0, list(batch)

    def take(self, batch: Sequence[LayoutSoftCoreSpec]) -> None:
        self._pending.extend(batch)

    def _staged(self, batch: Sequence[LayoutSoftCoreSpec]) -> Tuple[Dict[int, _ClassRuns], int]:
        staged: Dict[int, _ClassRuns] = {}
        total = self._total
        for c, items in _by_class(batch).items():
            old = self._classes.get(c, _ClassRuns())
            runs, labels, banks = old.extended(items)
            new = _ClassRuns(runs, labels, self.pool.classes.demand(runs), banks)
            total += new.demand - old.demand
            staged[c] = new
        return staged, total

    def extends(self, bundle: Sequence[LayoutSoftCoreSpec]) -> bool:
        """``pack_layout(batch + bundle).feasible``; when true, the bundle joins the batch."""
        if self._pending:
            staged, self._total = self._staged(self._pending)
            self._classes.update(staged)
            self._pending = []
        staged, total = self._staged(bundle)
        if total > self.pool.count:
            return False
        self._classes.update(staged)
        self._total = total
        return True
