"""[W15] A built program as its executors read it: per executed unit (a stage, or each level unit of a re-timed stage)
its input slices, and per hard core the used rectangle, the IR neuron each used column is (its placement record), and
the source each used axon row reads, as one integer code per row.

The weights are the ones the HCM uploads (the bank block on the bank fast path, ``executed_reads``). A row's code
names its source in the program-independent space both programs share: ``(node, index)`` of an IR node's
buffer (the network input -2, a compacted neural column, a host output), the always-on axon, or nothing (an off axon).
An in-unit core source is coded through the placement of the neuron it reads; a stage input through the slice that
holds its position. A source a program cannot name (an unused neuron, a position no slice holds) gets a code no other
program can produce, so it never matches. Read-only: nothing here writes to the program."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import enumerate_execution_stages
from mimarsinan.mapping.packing.executed_reads import executed_weights
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources

OFF = np.int64(-1)
#: Row kinds.
K_OFF, K_ON, K_INPUT, K_CORE, K_BAD = 0, 1, 2, 3, 4


def code(node: int, index) -> Any:
    """One int64 per source ``(node, index)``; ``node`` >= -16 (the reserved ids below 0 included)."""
    return (np.int64(int(node) + 16) << np.int64(32)) | np.asarray(index, dtype=np.int64)


def decode(value: int) -> Tuple[int, int]:
    return int(value >> 32) - 16, int(value & 0xFFFFFFFF)


ON = code(-3, 0)
BIAS = code(-4, 0)


@dataclass
class Placed:
    """One placement record: the IR node, its compacted columns ``[col0, col0 + cols)`` at the core's columns
    ``[neuron_offset, neuron_offset + cols)``."""

    unit: int
    core: int
    node: int
    col0: int
    cols: int
    neuron_offset: int
    record: Dict[str, Any]


@dataclass
class CoreView:
    """A hard core as executed: the used rectangle, per used column its (node, column), per used row its code,
    kind and auxiliary index (the in-unit source core, or the stage-input slice), and its core mapping and index
    (its executed weights)."""

    core: Any
    used_axons: int
    used_neurons: int
    node_of: np.ndarray
    col_of: np.ndarray
    codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    kinds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    aux: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    placements: List[Placed] = field(default_factory=list)
    mapping: Any = None
    index: int = 0

    def weights(self) -> np.ndarray:
        """The weight block the HCM uploads for this core (``executed_reads.executed_weights``: the bank block on the
        bank fast path, else the grid), recomputed per call so a sweep holds one core's block at a time."""
        return executed_weights(self.mapping, self.index)


@dataclass
class UnitView:
    """An executed unit: its stage, its program stage, the schedule segment, whether it is a level unit, its input
    slices ``(node, offset, size)`` and its cores."""

    index: int
    program_index: int
    stage: Any
    program_stage: Any
    segment: Optional[int]
    level: bool
    slices: List[Tuple[int, int, int]]
    cores: List[CoreView]


@dataclass
class ProgramView:
    units: List[UnitView]
    hosts: List[Tuple[int, int, Any]]
    placements: Dict[int, List[Placed]]
    unplaced: List[Tuple[int, int, int]]


def _used(core) -> Tuple[int, int]:
    return (max(int(core.axons_per_core) - int(core.available_axons), 0),
            max(int(core.neurons_per_core) - int(core.available_neurons), 0))


def _placements(unit: int, h: int, core_view: CoreView, records) -> None:
    node_of, col_of = core_view.node_of, core_view.col_of
    for record in records:
        start, count = int(record["neuron_offset"]), int(record["neurons"])
        base = int((record.get("neuron_range_in_original") or (0, count))[0])
        node = int(record["ir_node_id"])
        core_view.placements.append(Placed(unit, h, node, base, count, start, record))
        lo, hi = max(start, 0), min(start + count, core_view.used_neurons)
        if hi > lo:
            fresh = node_of[lo:hi] < 0
            node_of[lo:hi] = np.where(fresh, node, -2)          # -2: two placements claim one column
            col_of[lo:hi] = np.where(fresh, base + np.arange(lo - start, hi - start), -1)


def _slice_tables(slices) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max((o + s for _, o, s in slices), default=0)
    node = np.full(width, -1, dtype=np.int64)
    base = np.zeros(width, dtype=np.int64)
    which = np.full(width, -1, dtype=np.int64)
    for k, (n, o, s) in enumerate(slices):
        node[o:o + s], base[o:o + s], which[o:o + s] = n, np.arange(s), k
    return node, base, which


def _rows(view: UnitView, cv: CoreView, foreign: int, tables) -> None:
    """The row codes of one core (``foreign``: this program's reserved node id for an unnamable source; ``tables``:
    the unit's input slices per stage-input position)."""
    ua = cv.used_axons
    codes = np.full(ua, OFF, dtype=np.int64)
    kinds = np.zeros(ua, dtype=np.int8)
    aux = np.full(ua, -1, dtype=np.int64)
    s_node, s_base, s_which = tables
    for span in compress_spike_sources(cv.core.axon_sources[:ua]):
        d0, d1 = int(span.dst_start), min(int(span.dst_end), ua)
        if d1 <= d0:
            continue
        if span.kind == "off":
            continue
        if span.kind == "on":
            codes[d0:d1], kinds[d0:d1] = ON, K_ON
            continue
        idx = np.arange(int(span.src_start), int(span.src_start) + (d1 - d0), dtype=np.int64)
        if span.kind == "input":
            ok = idx < len(s_node)
            pos = np.where(ok, idx, 0)
            ok &= s_which[pos] >= 0
            codes[d0:d1] = np.where(ok, (s_node[pos] + 16) << 32 | s_base[pos], code(foreign, idx))
            kinds[d0:d1] = np.where(ok, K_INPUT, K_BAD)
            aux[d0:d1] = np.where(ok, s_which[pos], -1)
            continue
        src = int(span.src_core)
        if not 0 <= src < len(view.cores):
            codes[d0:d1], kinds[d0:d1] = code(foreign, idx), K_BAD
            continue
        sv = view.cores[src]
        ok = idx < sv.used_neurons
        pos = np.where(ok, idx, 0)
        nodes = sv.node_of[pos] if sv.used_neurons else np.full(len(idx), -1)
        ok &= nodes >= 0
        cols = sv.col_of[pos] if sv.used_neurons else np.zeros(len(idx), dtype=np.int64)
        codes[d0:d1] = np.where(ok, (nodes + 16) << 32 | np.maximum(cols, 0), code(foreign, idx))
        kinds[d0:d1] = np.where(ok, K_CORE, K_BAD)
        aux[d0:d1] = src
    cv.codes, cv.kinds, cv.aux = codes, kinds, aux


def build_view(program: Any, *, foreign: int) -> ProgramView:
    """The view of ``program``'s executed units (``enumerate_execution_stages``, the stage loop's own order)."""
    units: List[UnitView] = []
    hosts: List[Tuple[int, int, Any]] = []
    placements: Dict[int, List[Placed]] = {}
    unplaced: List[Tuple[int, int, int]] = []
    index_of = {id(s): i for i, s in enumerate(program.stages)}
    for index, program_stage, stage in enumerate_execution_stages(program.stages):
        p_index = index_of[id(program_stage)]
        if stage.kind != "neural" or stage.hard_core_mapping is None:
            hosts.append((index, p_index, stage))
            continue
        hcm = stage.hard_core_mapping
        records = list(getattr(hcm, "soft_core_placements_per_hard_core", None) or [])
        view = UnitView(index, p_index, stage, program_stage, program_stage.schedule_segment_index,
                        stage is not program_stage,
                        [(int(s.node_id), int(s.offset), int(s.size)) for s in stage.input_map], [])
        for h, core in enumerate(hcm.cores):
            ua, un = _used(core)
            cv = CoreView(core, ua, un, np.full(un, -1, dtype=np.int64), np.full(un, -1, dtype=np.int64),
                          mapping=hcm, index=h)
            _placements(len(units), h, cv, records[h] if h < len(records) else [])
            for p in cv.placements:
                placements.setdefault(p.node, []).append(p)
            unplaced.extend((len(units), h, int(k)) for k in np.flatnonzero(cv.node_of < 0))
            view.cores.append(cv)
        tables = _slice_tables(view.slices)
        for cv in view.cores:
            _rows(view, cv, foreign, tables)
        units.append(view)
    return ProgramView(units, hosts, placements, unplaced)


@dataclass
class Canonical:
    """A neuron block's fan-in summed per source: sorted unique codes, and per code a row of weights."""

    codes: np.ndarray
    weights: np.ndarray


def canonical(codes: np.ndarray, block: np.ndarray, bias: Optional[np.ndarray]) -> Canonical:
    """Rows summed per source code (off rows dropped), the hardware bias as one more row (code ``BIAS``)."""
    keep = codes != OFF
    rows, data = codes[keep], np.asarray(block, dtype=np.float64)[keep]
    if bias is not None:
        rows = np.concatenate([rows, [BIAS]])
        data = np.concatenate([data, np.asarray(bias, dtype=np.float64).reshape(1, -1)], axis=0)
    if not len(rows):
        return Canonical(np.zeros(0, dtype=np.int64), np.zeros((0, block.shape[1])))
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    starts = np.flatnonzero(np.r_[True, sorted_rows[1:] != sorted_rows[:-1]])
    return Canonical(sorted_rows[starts], np.add.reduceat(data[order], starts, axis=0))


def restrict(can: Canonical, lo: int, hi: int) -> Canonical:
    """The columns ``[lo, hi)`` of a canonical block, its all-zero rows there dropped."""
    weights = can.weights[:, lo:hi]
    live = np.any(weights != 0, axis=1)
    return Canonical(can.codes[live], weights[live])


def same(a: Canonical, b: Canonical) -> bool:
    """Equal fan-ins: the same sources with the same weights (both restricted, so zero rows are absent)."""
    return bool(np.array_equal(a.codes, b.codes) and np.array_equal(a.weights, b.weights))


def first_difference(a: Canonical, b: Canonical) -> Optional[Tuple[int, int, float, float]]:
    """``(code, column, a's weight, b's weight)`` of the first difference between two canonical blocks of one width,
    None when equal (a code one side lacks counts as weight 0 there)."""
    union = np.union1d(a.codes, b.codes)
    width = a.weights.shape[1]
    da, db = np.zeros((len(union), width)), np.zeros((len(union), width))
    da[np.searchsorted(union, a.codes)] = a.weights
    db[np.searchsorted(union, b.codes)] = b.weights
    diff = np.argwhere(da != db)
    if not len(diff):
        return None
    r, c = (int(v) for v in diff[0])
    return int(union[r]), c, float(da[r, c]), float(db[r, c])
