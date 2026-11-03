"""Grids of a lane program (WI-5e): per lane owner an unsigned, offset or symmetric output grid; per core the grid
its accumulator rows read and, per column, its fused operand's grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch

from mimarsinan.mapping.export.law_grids import GridUnion, GroupKey, ReadoutLawGridError
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.lanes.ir.fusion import LANE_OWNER, OPERAND_ROWS
from mimarsinan.models.nn.activations.integer_law import grid_levels, round_half_away
from mimarsinan.models.nn.activations.lane_quantization import quantize_to_lane_grid, unsigned_levels


@dataclass(frozen=True)
class LaneGrid:
    """A lane owner's output grid: ``k Δ - offset``, codes ``[0, 2^b - 1]`` when unsigned, else ``[-L, L]``."""

    range: float
    offset: float = 0.0
    unsigned: bool = False

    def codes(self, bits: int) -> Tuple[int, int]:
        return (0, unsigned_levels(bits)) if self.unsigned else (-grid_levels(bits), grid_levels(bits))

    def step(self, bits: int) -> float:
        return float(self.range) / self.codes(bits)[1]

    def zero_value(self, bits: int) -> float:
        """The value this grid writes for 0, by the lane's own rounding (ties away) and code clamp: 0.0 exactly
        where value 0 is a code (``offset`` a whole number of steps inside the codes)."""
        zero = torch.zeros(1, dtype=torch.float64)
        if self.unsigned:
            return float(quantize_to_lane_grid(zero, self.range, bits, self.offset)[0])
        low, high = self.codes(bits)
        step = self.step(bits)
        code = int(torch.clamp(round_half_away(torch.tensor(self.offset / step, dtype=torch.float64)), low, high))
        return code * step - float(self.offset)


@dataclass(frozen=True)
class GridRead:
    """A grid as a reader decodes it: step, offset, the largest code magnitude and its declared codes ``(low, high)``
    (a fused operand's term ``M_1 q`` spans ``M_1 low .. M_1 high``: its producer's law clamps to them)."""

    step: float
    offset: float
    levels: int
    codes: Tuple[int, int]


@dataclass(frozen=True)
class CoreLaneGrid:
    """One core's grids: the accumulator rows' read, the output grid (step, offset, codes), per column the
    operand's read (``None`` where the column has no operand)."""

    inputs: GridRead
    output_step: float
    output_offset: float
    lo: int
    hi: int
    operands: Tuple[Optional[GridRead], ...]


def owner_of(core: NeuralCore) -> int:
    owner = core.perceptron_index if core.perceptron_index is not None else getattr(core, LANE_OWNER, None)
    if owner is None:
        raise ReadoutLawGridError(f"core {core.name!r} belongs to no perceptron and no lane: its output grid is "
                                  f"undeclared")
    return int(owner)


def _operand_rows(core: NeuralCore) -> np.ndarray:
    rows = getattr(core, OPERAND_ROWS, None)
    return np.full(core.get_output_count(), -1, dtype=np.int64) if rows is None else np.asarray(rows)


def data_rows(core: NeuralCore) -> np.ndarray:
    """Mask of the accumulator's data rows: not off, not always-on, not pruned, not an operand row."""
    sources = core.input_sources.flatten()
    mask = np.array([not (s.is_off() or s.is_always_on()) for s in sources], dtype=bool)
    pruned = getattr(core, "pruned_row_mask", None)
    if pruned is not None and len(pruned) == len(sources):
        mask &= ~np.asarray(pruned, dtype=bool)
    rows = _operand_rows(core)
    mask[rows[rows >= 0]] = False
    return mask


def _producers(core: NeuralCore, rows: Iterable[int], nodes: Mapping[int, object]) -> Tuple[set, bool]:
    sources = core.input_sources.flatten()
    owners, host = set(), False
    for row in rows:
        node = nodes.get(int(sources[row].node_id)) if sources[row].node_id >= 0 else None
        if isinstance(node, NeuralCore):
            owners.add(owner_of(node))
        else:
            host = True
    return owners, host


def _host_read(core: NeuralCore) -> GridRead:
    grid = getattr(core, "boundary_grid", None)
    if grid is None or not grid.armed:
        raise ReadoutLawGridError(f"core {core.name!r} reads host-sourced axons but carries no armed boundary grid")
    low, high = grid.code_range if hasattr(grid, "code_range") else (-grid.levels, grid.levels)
    return GridRead(float(grid.step), float(grid.entry_offset()), max(abs(int(low)), abs(int(high))),
                    (int(low), int(high)))


class PrunedOffsetGridError(ReadoutLawGridError):
    """A pruned lane program with an offset lane grid whose value 0 is no code: pruning drops the rows that read a
    pruned column as if they read 0, while the column's law writes the code nearest its offset."""


_PRUNING_MASKS = ("pre_pruning_row_mask", "pre_pruning_col_mask", "pruned_row_mask", "pruned_col_mask")


def _pruned(core: NeuralCore) -> bool:
    return any(mask is not None and any(bool(v) for v in mask)
               for mask in (getattr(core, name, None) for name in _PRUNING_MASKS))


def refuse_pruned_offset_lanes(cores, owners: Mapping[int, LaneGrid], bits: int) -> None:
    """Refuse by name a pruned lane program beside a lane grid whose value 0 is no code (the grid's own snap of 0,
    :meth:`LaneGrid.zero_value`, is not exactly 0: an offset that is no whole number of steps, or whose code lies
    outside ``0..2^b - 1``): its pruned columns write a nonzero value the pruned readers dropped. Entry grids are
    exact (their rows are never pruned, ``offset_grid_exempt_rows``)."""
    if not any(_pruned(core) for core in cores):
        return
    for owner, grid in sorted(owners.items()):
        if grid.offset == 0.0:
            continue
        zero = grid.zero_value(bits)
        if zero != 0.0:
            raise PrunedOffsetGridError(
                f"lane owner {owner} writes an offset grid (offset {grid.offset:g}, step {grid.step(bits):g}, codes "
                f"{grid.codes(bits)}) whose value 0 is no code (0 snaps to {zero:g}), in a pruned program: pruning "
                f"drops the rows reading a pruned column as 0, which that column's law writes as that value; prune "
                f"without offset lanes")


def assign_lane_grids(ir_graph: IRGraph, owners: Mapping[int, LaneGrid], bits: int) -> Dict[int, CoreLaneGrid]:
    """``{core id: CoreLaneGrid}``. Owners read together by one core's accumulator share the widest range of their
    union (a concatenation join) and must agree on offset and sign; a core mixing host and core rows is refused."""
    nodes = {int(n.id): n for n in ir_graph.nodes}
    cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    union = GridUnion()
    reads = {}
    for core in cores:
        union.find_key((owner_of(core), 0))  # N2's grid key of an ungrouped owner: (owner, 0)
        producers, host = _producers(core, np.flatnonzero(data_rows(core)).tolist(), nodes)
        if producers and host:
            raise ReadoutLawGridError(f"core {core.name!r} reads both host-sourced and core-sourced axons; one DAC "
                                      f"grid per core")
        union.join_keys([(int(p), 0) for p in producers])
        reads[int(core.id)] = (producers, host)
    missing = sorted({owner_of(c) for c in cores} - set(owners))
    if missing:
        raise ReadoutLawGridError(f"lane owners {missing} carry cores but no readout range")
    joined: Dict[GroupKey, LaneGrid] = {}
    for owner, grid in owners.items():
        root = union.find_key((int(owner), 0))
        seen = joined.get(root)
        if seen is not None and (seen.offset, seen.unsigned) != (grid.offset, grid.unsigned):
            raise ReadoutLawGridError(f"lane owners {owner} and a joined owner read by one core have different grid "
                                      f"kinds (offset or sign): one DAC grid per core")
        joined[root] = grid if seen is None or grid.range > seen.range else seen
    unified = {int(o): LaneGrid(joined[union.find_key((int(o), 0))].range, g.offset, g.unsigned)
               for o, g in owners.items()}
    refuse_pruned_offset_lanes(cores, unified, bits)

    reads_by_owner: Dict[int, GridRead] = {}

    def read(owner: int) -> GridRead:
        hit = reads_by_owner.get(owner)
        if hit is None:
            grid = unified[owner]
            low, high = grid.codes(bits)
            hit = reads_by_owner[owner] = GridRead(grid.step(bits), grid.offset, max(abs(low), abs(high)),
                                                   (low, high))
        return hit

    out: Dict[int, CoreLaneGrid] = {}
    for core in cores:
        producers, host = reads[int(core.id)]
        inputs = (read(next(iter(producers))) if producers else _host_read(core) if host
                  else GridRead(1.0, 0.0, grid_levels(bits), (-grid_levels(bits), grid_levels(bits))))
        sources: list = list(core.input_sources.flatten())
        operands = tuple(None if row < 0 or sources[row].is_off()  # an off operand (a pruned column) adds 0
                         else read(owner_of(nodes[int(sources[row].node_id)]))  # type: ignore[arg-type]
                         for row in _operand_rows(core).tolist())
        grid = unified[owner_of(core)]
        low, high = grid.codes(bits)
        out[int(core.id)] = CoreLaneGrid(inputs, grid.step(bits), grid.offset, low, high, operands)
    return out
