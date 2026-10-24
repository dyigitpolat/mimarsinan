"""Columnar axon sources (coverage W9): a core's per-axon spike sources as three columns, and the soft-to-hard neuron
map held per placement.

A ``SpikeSource`` per axon costs about 104 B (a slotted object and its neuron int); the identity value program of a
large network holds one per core axon, and its hard-core mapping held a dict entry and two tuples per neuron. Here the
sources are three columns (17 B per axon) and the neuron map one record per placement. Every value is the one the
object forms hold: a source reads back as the ``SpikeSource`` it stands for, the spans are those
``compress_spike_sources`` gives, a remap is the per-source dict lookup's, a missing key is the same ``KeyError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Iterable, Iterator, List, Tuple, overload

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.support.spike_source_spans import (
    SPIKE_SOURCES_DENSE_TAG,
    SPIKE_SOURCES_SPANS_TAG,
    SpikeSourceSpan,
    _SPANS_MIN_AVG_RUN_LENGTH,
)

INPUT, OFF, ON = 1, 2, 4
"""Flag bits, as the packed pickle encodes them (``_encode_spike_sources_dense``)."""


class ColumnSpikeSource(SpikeSource):
    """One entry of ``SpikeSourceColumns``, read only: a write raises (columns are rewritten whole), so a writer of
    single sources fails by name instead of losing its write; a copy is an ordinary ``SpikeSource``."""

    __slots__ = ()

    def __setattr__(self, name, value):
        raise TypeError(f"a columnar axon source is read only (set {name!r}): rewrite its SpikeSourceColumns")

    def __reduce__(self):
        return (SpikeSource, (self.core_, self.neuron_, self.is_input_, self.is_off_, self.is_always_on_))


def _entry(core: int, neuron: int, flags: int) -> ColumnSpikeSource:
    entry = object.__new__(ColumnSpikeSource)
    for name, value in (("is_input_", bool(flags & INPUT)), ("is_off_", bool(flags & OFF)),
                        ("is_always_on_", bool(flags & ON)), ("core_", core), ("neuron_", neuron)):
        object.__setattr__(entry, name, value)
    return entry


class SpikeSourceColumns(Sequence[SpikeSource]):
    """Spike sources as columns: ``core`` and ``neuron`` (int64), ``flags`` (uint8, ``INPUT``/``OFF``/``ON``). Read
    like a list of ``SpikeSource``: its length, an index (a read-only ``ColumnSpikeSource``), a slice or an index array
    (columns), iteration; ``sources[i] = source`` writes one entry."""

    __slots__ = ("core", "neuron", "flags")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, core, neuron, flags) -> None:
        self.core = np.asarray(core, dtype=np.int64)
        self.neuron = np.asarray(neuron, dtype=np.int64)
        self.flags = np.asarray(flags, dtype=np.uint8)
        if not self.core.shape == self.neuron.shape == self.flags.shape or self.core.ndim != 1:
            raise ValueError("SpikeSourceColumns: three one-dimensional columns of one length")

    @classmethod
    def from_ir_sources(cls, sources: "Sequence[Any] | np.ndarray") -> "SpikeSourceColumns":
        """The columns ``ir_source_to_spike_source`` gives source by source: off (-1) and always-on (-3) at neuron 0,
        a network input (-2) at its index, a node's output at its id and index."""
        count = len(sources)
        ids = np.fromiter((source.node_id for source in sources), dtype=np.int64, count=count)
        index = np.fromiter((source.index for source in sources), dtype=np.int64, count=count)
        flags = np.select([ids == -1, ids == -2, ids == -3], [OFF, INPUT, ON], 0).astype(np.uint8)
        return cls(ids, np.where((ids == -1) | (ids == -3), 0, index), flags)

    @classmethod
    def of(cls, sources: Iterable[SpikeSource]) -> "SpikeSourceColumns":
        if isinstance(sources, SpikeSourceColumns):
            return sources
        listed = list(sources)
        return cls([int(s.core_) for s in listed], [int(s.neuron_) for s in listed],
                   [(INPUT if s.is_input_ else 0) | (OFF if s.is_off_ else 0) | (ON if s.is_always_on_ else 0)
                    for s in listed])

    @classmethod
    def concatenate(cls, parts: Sequence["SpikeSourceColumns"]) -> "SpikeSourceColumns":
        return cls(np.concatenate([p.core for p in parts]), np.concatenate([p.neuron for p in parts]),
                   np.concatenate([p.flags for p in parts]))

    def __len__(self) -> int:
        return int(self.core.size)

    @overload
    def __getitem__(self, key: int) -> ColumnSpikeSource: ...

    @overload
    def __getitem__(self, key: slice) -> "SpikeSourceColumns": ...

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return _entry(int(self.core[key]), int(self.neuron[key]), int(self.flags[key]))
        return self.take(key)

    def take(self, key) -> "SpikeSourceColumns":
        """The entries at ``key`` (a slice or an index array), a copy."""
        return SpikeSourceColumns(self.core[key].copy(), self.neuron[key].copy(), self.flags[key].copy())

    def copy(self) -> "SpikeSourceColumns":
        return self.take(slice(None))

    def __setitem__(self, key: int, source: SpikeSource) -> None:
        one = SpikeSourceColumns.of([source])
        self.core[key], self.neuron[key], self.flags[key] = one.core[0], one.neuron[0], one.flags[0]

    def __iter__(self) -> Iterator[ColumnSpikeSource]:
        for core, neuron, flags in zip(self.core.tolist(), self.neuron.tolist(), self.flags.tolist()):
            yield _entry(core, neuron, flags)

    # A segment's output sources stand where an object array of them stood: ``flatten``, ``tolist``, ``np.asarray``.
    @property
    def shape(self) -> Tuple[int]:
        return (len(self),)

    def flatten(self) -> "SpikeSourceColumns":
        return self

    def tolist(self) -> list:
        return list(self)

    def __array__(self, dtype=None, copy=None):
        array = np.empty(len(self), dtype=object)
        array[:] = self.tolist()
        return array if dtype is None else array.astype(dtype)

    def assign(self, other: "SpikeSourceColumns") -> None:
        """Become ``other`` in place (a caller's sequence rewritten, as ``list.clear`` and ``extend`` rewrote it)."""
        self.core, self.neuron, self.flags = other.core, other.neuron, other.flags

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpikeSourceColumns):
            return NotImplemented
        return all(np.array_equal(a, b) for a, b in ((self.core, other.core), (self.neuron, other.neuron),
                                                      (self.flags, other.flags)))

    def source_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(core, neuron, off, input, always on)``, one array each (``ChipLatency`` reads them by duck typing)."""
        flags = self.flags
        return self.core, self.neuron, (flags & OFF) > 0, (flags & INPUT) > 0, (flags & ON) > 0

    def kinds(self) -> np.ndarray:
        """``compress_spike_sources``'s classification: 0 off, 1 input, 2 on, 3 core (off before input before on)."""
        flags = self.flags
        return np.select([(flags & OFF) > 0, (flags & INPUT) > 0, (flags & ON) > 0], [0, 1, 2], 3)

    def spans(self) -> List[SpikeSourceSpan]:
        """``compress_spike_sources`` of these sources: runs of one kind, an input or core run of one source at
        stride 1."""
        kinds = self.kinds()
        src = np.select([kinds == 0, kinds == 1, kinds == 2], [-1, -2, -3], self.core)
        start = np.where((kinds == 1) | (kinds == 3), self.neuron, 0)
        if not kinds.size:
            return []
        indexed = (kinds[1:] == 1) | (kinds[1:] == 3)
        breaks = (kinds[1:] != kinds[:-1]) | (indexed & ((src[1:] != src[:-1]) | (start[1:] != start[:-1] + 1)))
        heads = np.concatenate([[0], np.flatnonzero(breaks) + 1])
        ends = np.concatenate([heads[1:], [kinds.size]])
        names = ("off", "input", "on", "core")
        return [SpikeSourceSpan(kind=names[int(kinds[h])], src_core=int(src[h]), src_start=int(start[h]),
                                length=int(e - h), dst_start=int(h)) for h, e in zip(heads.tolist(), ends.tolist())]

    compressed_spans = spans

    def packed_payload(self) -> tuple:
        """``encode_spike_sources_packed``'s payload: spans when every source round-trips through them
        (``span_round_trip_exact``) and they compress, the dense columns otherwise."""
        kinds, core, neuron, flags = self.kinds(), self.core, self.neuron, self.flags
        exact = np.select([kinds == 0, kinds == 1, kinds == 2],
                          [(core == -1) & (neuron == 0) & ((flags & (INPUT | ON)) == 0),
                           (core == -2) & ((flags & ON) == 0), (core == -3) & (neuron == 0)], True)
        if bool(exact.all()):
            spans = self.spans()
            if len(spans) * _SPANS_MIN_AVG_RUN_LENGTH <= len(self):
                return (SPIKE_SOURCES_SPANS_TAG, tuple(spans))
        return (SPIKE_SOURCES_DENSE_TAG, core.astype(np.int32), neuron.astype(np.int32), flags.copy())

    def remapped(self, lookup: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                 ) -> "SpikeSourceColumns":
        """Every core entry ``(core, neuron)`` through ``lookup`` (``(cores, neurons) -> (cores, neurons, kept)``); an
        entry it does not keep becomes off (core -1, neuron 0); every other entry is unchanged."""
        core, neuron, flags = self.core.copy(), self.neuron.copy(), self.flags.copy()
        at = np.flatnonzero(self.kinds() == 3)
        new_core, new_neuron, kept = lookup(core[at], neuron[at])
        core[at] = np.where(kept, new_core, -1)
        neuron[at] = np.where(kept, new_neuron, 0)
        flags[at] = np.where(kept, flags[at], OFF)
        return SpikeSourceColumns(core, neuron, flags)

    def padded(self, length: int) -> "SpikeSourceColumns":
        """These sources, then off sources up to ``length``."""
        extra = max(0, int(length) - len(self))
        if not extra:
            return self
        return SpikeSourceColumns.concatenate([self, SpikeSourceColumns(np.full(extra, -1), np.zeros(extra),
                                                                        np.full(extra, OFF))])


def dict_lookup(maps: Mapping) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A ``remapped`` lookup through ``{core: {neuron: new neuron}}`` (compaction's reindex maps): a core without a
    map, or a neuron its map does not hold, is not kept; the core is unchanged."""

    def lookup(cores: np.ndarray, neurons: np.ndarray):
        new, kept = np.zeros_like(neurons), np.zeros(cores.shape, dtype=bool)
        for core in np.unique(cores).tolist():
            at = np.flatnonzero(cores == core)
            remap = maps.get(core)
            if remap is None:
                continue
            values = [remap.get(n) for n in neurons[at].tolist()]
            hit = np.array([v is not None for v in values], dtype=bool)
            new[at[hit]] = [v for v in values if v is not None]
            kept[at] = hit
        return cores, new, kept

    return lookup


class PlacementNeuronMap(Mapping):
    """``(soft core id, neuron of the original soft core) -> (hard core index, neuron)`` of every placed soft core
    neuron, held as one record per placement (``place``) instead of one dict entry per neuron. The placements of one
    soft core (its fragments) cover disjoint neuron ranges; an overlapping placement raises."""

    __slots__ = ("_by_core", "_order", "_count")

    def __init__(self) -> None:
        self._by_core: dict = {}
        self._order: list = []
        self._count = 0

    def place(self, soft_core: int, first: int, count: int, hard_core: int, hard_first: int) -> None:
        records = self._by_core.setdefault(int(soft_core), [])
        for start, size, _hc, _hf in records:
            if first < start + size and start < first + count:
                raise RuntimeError(f"PlacementNeuronMap: soft core {soft_core} neurons {first}..{first + count} "
                                   f"placed twice (an earlier placement holds {start}..{start + size})")
        record = (int(first), int(count), int(hard_core), int(hard_first))
        records.append(record)
        self._order.append((int(soft_core), record))
        self._count += int(count)

    def __getitem__(self, key):
        try:
            soft_core, neuron = key
        except (TypeError, ValueError):
            raise KeyError(key) from None
        for start, size, hard_core, hard_first in self._by_core.get(soft_core, ()):
            if start <= neuron < start + size:
                return (hard_core, hard_first + neuron - start)
        raise KeyError(key)

    def __iter__(self):
        for soft_core, (start, size, _hc, _hf) in self._order:
            for neuron in range(start, start + size):
                yield (soft_core, neuron)

    def __len__(self) -> int:
        return self._count

    def lookup(self, cores: np.ndarray, neurons: np.ndarray):
        """A ``remapped`` lookup: every entry mapped (``(hard core, neuron)``), or the ``KeyError`` of the first entry
        no placement holds, as the per-source dict lookup raises it."""
        new_core, new_neuron = np.zeros_like(cores), np.zeros_like(neurons)
        found = np.zeros(cores.shape, dtype=bool)
        for core in np.unique(cores).tolist():
            at = np.flatnonzero(cores == core)
            for start, size, hard_core, hard_first in self._by_core.get(core, ()):
                inside = at[(neurons[at] >= start) & (neurons[at] < start + size)]
                new_core[inside], new_neuron[inside] = hard_core, hard_first + neurons[inside] - start
                found[inside] = True
        if not found.all():
            first = int(np.flatnonzero(~found)[0])
            raise KeyError((int(cores[first]), int(neurons[first])))
        return new_core, new_neuron, found
