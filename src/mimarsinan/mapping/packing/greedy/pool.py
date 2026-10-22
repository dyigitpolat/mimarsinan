"""The greedy engine's unused cores: the caller's list, or cores built on demand from declared types."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Generic, List, Sequence, Tuple, TypeVar

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal

HardT = TypeVar("HardT")
TypeKey = Tuple[int, int]


class CoreListedTwice(ToolchainRefusal, ValueError):
    """A list of unused cores that holds one core object more than once: one core the engine could open twice."""


class CorePool(Generic[HardT]):
    """Unused cores grouped by type key, each group in the pool's list order.

    The engine reads a group's last cores, takes cores from its tail (a fresh core, a fusion) or its head (a split
    into an unused core), and asks for the list position of a group's first core; on a list of distinct cores
    every answer equals the one the caller's list gives, so the pool never changes a decision, only what a decision
    costs. A list holding one core object twice is refused by name (``CoreListedTwice``).
    """

    def keys(self) -> List[TypeKey]:
        """Type keys still holding a core, in the order the list first holds them."""
        raise NotImplementedError

    def count(self, key: TypeKey) -> int:
        raise NotImplementedError

    def total(self) -> int:
        raise NotImplementedError

    def last(self, key: TypeKey, n: int = 1) -> List[HardT]:
        """The group's last ``n`` cores, in list order."""
        raise NotImplementedError

    def take_last(self, key: TypeKey, n: int = 1) -> List[HardT]:
        raise NotImplementedError

    def first_position(self, key: TypeKey) -> int:
        raise NotImplementedError

    def take_first(self, key: TypeKey) -> HardT:
        raise NotImplementedError

    def finish(self) -> None:
        """Bring the backing list to the state the engine's removals leave (a no-op without one)."""


class ListCorePool(CorePool[HardT]):
    """The caller's list of unused cores, taken from in O(1); ``finish`` removes the taken cores in one pass."""

    def __init__(self, items: List[HardT], key_of: Callable[[HardT], TypeKey]) -> None:
        if len(set(map(id, items))) != len(items):
            seen = Counter(map(id, items))
            raise CoreListedTwice(
                f"greedy packing: a core object is listed {max(seen.values())} times in the unused-core list "
                f"({len(items)} entries, {len(seen)} cores); each unused core must be listed once")
        self._items = items
        self._groups: Dict[TypeKey, List[Tuple[int, HardT]]] = {}
        self._heads: Dict[TypeKey, int] = {}
        for position, hc in enumerate(items):
            self._groups.setdefault(key_of(hc), []).append((position, hc))
        self._heads = {key: 0 for key in self._groups}
        self._taken: set = set()
        self._total = len(items)

    def keys(self) -> List[TypeKey]:
        return [key for key in self._groups if self.count(key)]

    def count(self, key: TypeKey) -> int:
        group = self._groups.get(key)
        return 0 if group is None else len(group) - self._heads[key]

    def total(self) -> int:
        return self._total

    def last(self, key: TypeKey, n: int = 1) -> List[HardT]:
        return [hc for _, hc in self._groups[key][len(self._groups[key]) - n:]]

    def take_last(self, key: TypeKey, n: int = 1) -> List[HardT]:
        group = self._groups[key]
        taken = group[len(group) - n:]
        del group[len(group) - n:]
        return self._took([hc for _, hc in taken])

    def first_position(self, key: TypeKey) -> int:
        return self._groups[key][self._heads[key]][0]

    def take_first(self, key: TypeKey) -> HardT:
        hc = self._groups[key][self._heads[key]][1]
        self._heads[key] += 1
        return self._took([hc])[0]

    def _took(self, cores: List[HardT]) -> List[HardT]:
        self._taken.update(id(hc) for hc in cores)
        self._total -= len(cores)
        return cores

    def finish(self) -> None:
        """Remove the taken cores from the list, keeping the order of the rest."""
        if not self._taken:
            return
        self._items[:] = [hc for hc in self._items if id(hc) not in self._taken]
        self._taken.clear()


class LazyCorePool(CorePool[HardT]):
    """Declared core types, each core built only when the engine reads or takes it.

    List positions follow the order a list of every declared core would hold them in (type by type, ``count``
    cores each), so position ties break as they would on that list.
    """

    def __init__(self, types: Sequence[Tuple[TypeKey, int]], build: Callable[[TypeKey], HardT]) -> None:
        self._build = build
        self._ranges: Dict[TypeKey, List[List[int]]] = {}
        offset = 0
        for key, count in types:
            if count > 0:
                self._ranges.setdefault(key, []).append([offset, offset + count])
            offset += max(0, count)
        self._built: Dict[int, HardT] = {}
        self._total = sum(hi - lo for ranges in self._ranges.values() for lo, hi in ranges)

    def keys(self) -> List[TypeKey]:
        return [key for key, ranges in self._ranges.items() if ranges]

    def count(self, key: TypeKey) -> int:
        return sum(hi - lo for lo, hi in self._ranges.get(key, ()))

    def total(self) -> int:
        return self._total

    def _core(self, key: TypeKey, position: int) -> HardT:
        if position not in self._built:
            self._built[position] = self._build(key)
        return self._built[position]

    def _tail_positions(self, key: TypeKey, n: int) -> List[int]:
        out: List[int] = []
        for lo, hi in reversed(self._ranges[key]):
            take = min(n - len(out), hi - lo)
            out.extend(range(hi - take, hi)[::-1])
            if len(out) == n:
                break
        return out[::-1]

    def last(self, key: TypeKey, n: int = 1) -> List[HardT]:
        return [self._core(key, p) for p in self._tail_positions(key, n)]

    def take_last(self, key: TypeKey, n: int = 1) -> List[HardT]:
        cores = self.last(key, n)
        ranges, left = self._ranges[key], n
        while left:
            take = min(left, ranges[-1][1] - ranges[-1][0])
            ranges[-1][1] -= take
            left -= take
            if ranges[-1][0] == ranges[-1][1]:
                ranges.pop()
        return self._took(cores)

    def first_position(self, key: TypeKey) -> int:
        return self._ranges[key][0][0]

    def take_first(self, key: TypeKey) -> HardT:
        ranges = self._ranges[key]
        core = self._core(key, ranges[0][0])
        ranges[0][0] += 1
        if ranges[0][0] == ranges[0][1]:
            ranges.pop(0)
        return self._took([core])[0]

    def _took(self, cores: List[HardT]) -> List[HardT]:
        self._total -= len(cores)
        return cores
