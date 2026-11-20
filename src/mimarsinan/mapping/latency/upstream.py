"""Shared upstream walks for the latency engines, and [U-LIVE] the one liveness predicate every scheduling rule reads.

An axon is live for scheduling exactly where it can deliver charge to a member of the scheduled unit: it is not off,
it reads neither the unit's input (the segment input, at cycle 0 of every window) nor the always-on line (every
cycle), and its weight row holds a nonzero weight in a column the deployed core keeps (an IR node's compaction masks
apply; a bound operand bank's placeholder, NaN until its payload is written, counts as nonzero). Per neuron, its entry
is nonzero. ``live_axons`` is that predicate. Every rule that decides which axons count for a schedule reads it here,
through the adapters ``core_reads`` (a hard core) and ``ir_reads`` / ``ir_live_reads`` (an IR node): ChipLatency's sweep, its
shift alignment, its consumer map and its per-neuron delays, IRLatency, the relay planner, the window rule, the pass
projection's fused and own latencies and delivery delays, the static check's identity timing and the program check's
core facts. ChipLatency's direct signals of the output sources read ``direct_signals``. The finite list of those
sites is held by a test over src's syntax trees (tests/unit/mapping/test_live_sites.py)."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, NamedTuple, Optional, Protocol

import numpy as np


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


def charged_rows(matrix: Any, n_axons: int, columns: Optional[np.ndarray] = None) -> np.ndarray:
    """Per axon, whether its weight row holds a nonzero weight (in ``columns`` where given, the columns a core keeps;
    a boolean matrix is the nonzero mask, read without a second comparison).
    Rows past ``n_axons`` are a hard core's padding and are no axon, and an axon past the matrix's rows holds no
    weight. A matrix that is absent, or one with a nonzero row past the axons, cannot be aligned with its sources, and
    every row reads charged (the conservative reading: a schedule never drops what it cannot read)."""
    n = int(n_axons)
    if matrix is None:
        return np.ones(n, dtype=bool)
    m = np.asarray(matrix)
    if m.ndim != 2:
        return np.ones(n, dtype=bool)
    nonzero = m if m.dtype == bool else m != 0  # a boolean matrix is read as the nonzero mask itself
    if nonzero.shape[0] > n and bool(nonzero[n:].any()):
        return np.ones(n, dtype=bool)
    if columns is not None:
        nonzero = nonzero[:, np.asarray(columns, dtype=np.int64)]
    out = np.zeros(n, dtype=bool)
    k = min(n, int(nonzero.shape[0]))
    if nonzero.shape[1] and k:
        out[:k] = nonzero[:k].any(axis=1)
    return out


def live_axons(matrix: Any, off: np.ndarray, outside: np.ndarray, columns: Optional[np.ndarray] = None) -> np.ndarray:
    """THE predicate: per axon, live for scheduling = not ``off``, not ``outside`` the scheduled unit (its input or the
    always-on line) and its row charged (``charged_rows``)."""
    off = np.asarray(off, dtype=bool)
    outside = np.asarray(outside, dtype=bool)
    return charged_rows(matrix, len(off), columns) & ~off & ~outside


def source_arrays(sources) -> Any:
    """``(core, neuron, off, input, always on)`` of a hard core's axon sources, one array each: a columnar sequence's
    own (``SpikeSourceColumns.source_arrays``, coverage W9), else read source by source (a source without a core
    index reads no core)."""
    own = getattr(sources, "source_arrays", None)
    if callable(own):
        return own()
    listed = list(sources)
    count = len(listed)

    def column(read, dtype):
        return np.fromiter((read(s) for s in listed), dtype=dtype, count=count)

    return (column(lambda s: int(getattr(s, "core_", -1)), np.int64), column(lambda s: int(s.neuron_), np.int64),
            column(lambda s: bool(getattr(s, "is_off_", False)), bool),
            column(lambda s: bool(getattr(s, "is_input_", False)), bool),
            column(lambda s: bool(getattr(s, "is_always_on_", False)), bool))


def direct_signals(sources) -> np.ndarray:
    """Per source, a direct signal by the predicate's marks: off, a stage input or the always-on line (no core of the
    stage delivers it, so it adds no delay)."""
    _core, _neuron, off, is_input, on = source_arrays(sources)
    return np.asarray(off, dtype=bool) | np.asarray(is_input, dtype=bool) | np.asarray(on, dtype=bool)


class CoreReads(NamedTuple):
    """A hard core's axons as a schedule reads them: per axon its producer core (-1 where it reads no core of the
    stage: off, input or always-on) and neuron, whether it reads a stage input, ``live`` (the predicate; a stage input
    counts only where ``carried`` says an earlier pass of the scheduled unit produces it), and per (row, neuron) the
    entries that deliver charge (``entries``: nonzero and not off, the matrix's padding rows included as read)."""

    core: np.ndarray
    neuron: np.ndarray
    stage_input: np.ndarray
    live: np.ndarray
    entries: Optional[np.ndarray]


def core_reads(core: Any, *, carried: Optional[Callable[[int], bool]] = None, matrix: Any = None) -> CoreReads:
    """``CoreReads`` of one hard core under the predicate (``matrix``: its matrix when the caller holds it, else read
    once here; a core with no matrix to read reads every row charged); ``carried``, given a stage-input axon's input neuron, says whether an earlier pass of the scheduled
    unit produces it (the pass projection's fused segment), every other stage input being outside the unit."""
    src, neuron, off, is_input, on = source_arrays(core.axon_sources)
    off, is_input, on = (np.asarray(a, dtype=bool) for a in (off, is_input, on))
    if matrix is None:
        read = getattr(core, "get_core_matrix", None)
        matrix = read() if callable(read) else None
    nonzero = None if matrix is None else np.asarray(matrix) != 0
    if nonzero is not None and nonzero.ndim != 2:
        nonzero = None
    inner = np.zeros(len(off), dtype=bool)
    if carried is not None:
        for a in np.flatnonzero(is_input & ~off & ~on):
            inner[a] = bool(carried(int(neuron[a])))
    src = np.asarray(src, dtype=np.int64)
    # a source with no core of the stage behind it (a negative core index) is a direct signal, flagged or not
    live = live_axons(nonzero, off, (is_input & ~inner) | on | ((src < 0) & ~is_input))
    if nonzero is None:  # no weights to read: every row charged, no per-neuron entry known
        nonzero = np.zeros((0, 0), dtype=bool)
    rows = min(nonzero.shape[0], len(off))
    nonzero[:rows][off[:rows]] = False
    member = ~(off | is_input | on) & (src >= 0)
    return CoreReads(np.where(member, src, -1), np.asarray(neuron, dtype=np.int64),
                     is_input & ~off & ~on, live, nonzero)


def _kept(node: Any, n_axons: int, n_neurons: int) -> "tuple[np.ndarray, Optional[np.ndarray]]":
    """The rows (a mask) and columns (indices, None for every column) an IR node's compaction keeps where its masks
    describe its matrix (``compaction.compaction_masks_apply``: both masks at its shape), else every row and column."""
    rows, cols = getattr(node, "pruned_row_mask", None), getattr(node, "pruned_col_mask", None)
    if rows is None or cols is None or len(rows) != n_axons or len(cols) != n_neurons:
        return np.ones(int(n_axons), dtype=bool), None
    keep_cols = None if not any(cols) else np.asarray([c for c in range(n_neurons) if not cols[c]], dtype=np.int64)
    return ~np.asarray(rows, dtype=bool), keep_cols


def ir_live_reads(node: Any, ir_graph: Any = None, members: Optional[Any] = None) -> np.ndarray:
    """Per flat input source of an IR node: the predicate (``ir_reads``' second array)."""
    return ir_reads(node, ir_graph, members)[1]


def ir_reads(node: Any, ir_graph: Any = None, members: Optional[Any] = None) -> "tuple[np.ndarray, np.ndarray]":
    """``(node id per flat input source, the predicate per source)`` of an IR node. Off is no ``IRSource`` or node id
    -1; outside is the raw input (-2), the always-on line (-3) or, given ``members`` (a container of node ids, or an
    array indexed by node id whose entry is at or above 0 for a member), a node the scheduled unit does not hold. A neural core's row is charged where it holds a nonzero
    weight in a column its compaction keeps and the row is kept; a host operation has no weight row, so each of its
    reads is charged."""
    flat = node.input_sources.flatten()
    n = len(flat)
    listed = [int(getattr(s, "node_id", -1)) for s in flat]
    ids = np.asarray(listed, dtype=np.int64).reshape(n)
    off = ids == -1
    outside = ids < -1
    if isinstance(members, np.ndarray):  # a table by node id, a member's entry at or above 0: one vectorized read
        known = (ids >= 0) & (ids < len(members))
        member = np.zeros(n, dtype=bool)
        member[known] = members[ids[known]] >= 0
        outside |= (ids >= 0) & ~member
    elif members is not None:
        outside |= (ids >= 0) & ~np.fromiter((i in members for i in listed), dtype=bool, count=n)
    get = getattr(node, "get_core_matrix", None)
    if get is None:
        return ids, ~off & ~outside
    try:
        matrix = get(ir_graph)
    except (ValueError, KeyError):  # a node whose matrix cannot be resolved: every row reads charged
        matrix = None
    m = None if matrix is None else np.asarray(matrix)
    width = int(m.shape[1]) if m is not None and m.ndim == 2 else 0
    rows, columns = _kept(node, n, width)
    return ids, live_axons(m, off, outside, columns) & rows
