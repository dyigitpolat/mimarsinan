"""Input and output grids of law-carrying cores: one DAC grid per core, producer grids unified (E282.S)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_stamp import core_channels
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.integer_law import grid_levels

#: A grid's key: (perceptron index, consumer group); a perceptron without groups is group 0.
GroupKey = Tuple[int, int]


class ReadoutLawGridError(ToolchainRefusal, ValueError):
    """A core whose data axons do not lie on one declared grid, so its accumulator has no integer form."""


@dataclass(frozen=True)
class CoreGrid:
    """The grids one core's law reads and writes. A core whose columns span several output groups (N2 consumer
    groups) states each column's step and levels; ``output_levels`` None is the ``activation_bits`` grid."""

    input_step: float
    input_levels: int
    output_step: float
    output_range: float
    output_levels: Optional[int] = None
    column_steps: Optional[Tuple[float, ...]] = None
    column_levels: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class OutputGroups:
    """A perceptron's consumer groups: per output channel its group, per group its range and positive levels."""

    channels: np.ndarray
    ranges: Tuple[float, ...]
    levels: Tuple[int, ...]


def data_sources(core: NeuralCore) -> Iterable:
    """The core's data-axon sources (off, always-on and pruned rows excluded)."""
    sources = core.input_sources.flatten()
    pruned = getattr(core, "pruned_row_mask", None)
    mask = list(pruned) if pruned is not None and len(pruned) == len(sources) else [False] * len(sources)
    for row, source in enumerate(sources):
        if source.is_off() or source.is_always_on() or mask[row]:
            continue
        yield source


class _Groups:
    """Group lookups over the perceptrons' consumer groups (every other perceptron is one group, 0)."""

    def __init__(self, groups: Mapping[int, OutputGroups]) -> None:
        self.groups = groups
        self._channels: Dict[int, np.ndarray] = {}

    def of_column(self, core: NeuralCore, column: int) -> GroupKey:
        index = int(core.perceptron_index)  # type: ignore[arg-type]
        found = self.groups.get(index)
        if found is None:
            return (index, 0)
        channels = self._channels.get(int(core.id))
        if channels is None:
            channels = self._channels[int(core.id)] = core_channels(core)
        return (index, int(found.channels[int(channels[column])]))

    def of_columns(self, core: NeuralCore) -> Tuple[GroupKey, ...]:
        index = int(core.perceptron_index)  # type: ignore[arg-type]
        if index not in self.groups:
            return ((index, 0),)
        return tuple(self.of_column(core, j) for j in range(int(core.get_output_count())))


def _producers(core: NeuralCore, nodes_by_id: Mapping[int, object], groups: _Groups) -> Tuple[Set[GroupKey], bool]:
    """(grid keys of producer cores, whether any data axon is host- or input-sourced)."""
    producers: Set[GroupKey] = set()
    host = False
    for source in data_sources(core):
        node = nodes_by_id.get(int(source.node_id)) if source.node_id >= 0 else None
        if isinstance(node, NeuralCore):
            if node.perceptron_index is None:
                raise ReadoutLawGridError(
                    f"core {core.name!r} reads core {node.name!r}, which belongs to no perceptron: its output "
                    f"grid is undeclared"
                )
            producers.add(groups.of_column(node, int(source.index)))
        else:
            host = True
    return producers, host


class _Union:
    """Union-find over grid keys."""

    def __init__(self) -> None:
        self.parent: Dict[GroupKey, GroupKey] = {}

    def find_key(self, x: GroupKey) -> GroupKey:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join_keys(self, items: Iterable[GroupKey]) -> None:
        roots = [self.find_key(i) for i in items]
        for root in roots[1:]:
            self.parent[root] = roots[0]

    def keys(self) -> Iterable[GroupKey]:
        return list(self.parent)


def assign_core_grids(
    ir_graph, perceptron_ranges: Mapping[int, float], activation_bits: int, *,
    groups: Optional[Mapping[int, OutputGroups]] = None,
) -> Tuple[Dict[int, CoreGrid], Dict]:
    """``({core id: CoreGrid}, {grid key: output step})``, the keys perceptron indices when no perceptron has
    consumer groups (G2's surface) and ``(perceptron, group)`` pairs otherwise.

    Every output grid spans its calibrated (or declared) range on its levels; grids read by one core share the
    widest range of their union (concatenation joins), so each core reads one grid. A core whose data axons are
    host- or input-sourced reads its armed boundary grid; a core mixing both sources, a host-fed core without a
    grid, or a union of grids of different widths is refused by name.
    """
    grouping = _Groups(dict(groups or {}))
    nodes_by_id = {int(n.id): n for n in ir_graph.nodes}
    cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    union = _Union()
    reads = {}
    columns: Dict[int, Tuple[GroupKey, ...]] = {}
    for core in cores:
        if core.perceptron_index is None:
            raise ReadoutLawGridError(f"core {core.name!r} belongs to no perceptron: its output grid is undeclared")
        columns[int(core.id)] = grouping.of_columns(core)
        for key in columns[int(core.id)]:
            union.find_key(key)
        producers, host = _producers(core, nodes_by_id, grouping)
        if producers and host:
            raise ReadoutLawGridError(
                f"core {core.name!r} reads both host-sourced and core-sourced axons; one DAC grid per core"
            )
        grid = getattr(core, "boundary_grid", None)
        if host and (grid is None or not grid.armed):
            raise ReadoutLawGridError(
                f"core {core.name!r} reads host-sourced axons but carries no armed boundary grid"
            )
        union.join_keys(producers)
        reads[int(core.id)] = (producers, host)
    owners = {key[0] for keys in columns.values() for key in keys}
    missing = sorted(owners - set(perceptron_ranges) - set(grouping.groups))
    if missing:
        raise ReadoutLawGridError(f"perceptrons {missing} carry cores but no calibrated readout range")
    levels = grid_levels(activation_bits)

    def range_of(key: GroupKey) -> float:
        found = grouping.groups.get(key[0])
        return float(found.ranges[key[1]]) if found is not None else float(perceptron_ranges[key[0]])

    def levels_of(key: GroupKey) -> int:
        found = grouping.groups.get(key[0])
        return int(found.levels[key[1]]) if found is not None else levels

    keys = set(union.keys()) | {(int(i), 0) for i in perceptron_ranges if int(i) not in grouping.groups}
    widest: Dict[GroupKey, float] = {}
    width: Dict[GroupKey, int] = {}
    for key in sorted(keys):
        root = union.find_key(key)
        widest[root] = max(widest.get(root, 0.0), range_of(key))
        if width.setdefault(root, levels_of(key)) != levels_of(key):
            raise ReadoutLawGridError(f"one core reads output grids of {width[root]} and {levels_of(key)} levels "
                                      f"(perceptron {key[0]}, group {key[1]}): one DAC grid per core")
    steps = {key: widest[union.find_key(key)] / width[union.find_key(key)] for key in keys}
    grids: Dict[int, CoreGrid] = {}
    for core in cores:
        producers, host = reads[int(core.id)]
        grid = core.boundary_grid
        if producers:
            first = next(iter(producers))
            in_step, in_levels = steps[first], width[union.find_key(first)]
        elif host and grid is not None:
            in_step, in_levels = float(grid.step), int(grid.levels)
        else:
            in_step, in_levels = 1.0, levels
        out = columns[int(core.id)]
        roots = [union.find_key(key) for key in out]
        column_steps = tuple(steps[key] for key in out)
        column_levels = tuple(width[root] for root in roots)
        uniform = len(set(zip(column_steps, column_levels))) == 1
        grids[int(core.id)] = CoreGrid(
            input_step=in_step, input_levels=in_levels, output_step=column_steps[0], output_range=widest[roots[0]],
            output_levels=None if column_levels[0] == levels else column_levels[0],
            column_steps=None if uniform else column_steps, column_levels=None if uniform else column_levels,
        )
    if not grouping.groups:
        return grids, {key[0]: step for key, step in steps.items()}
    return grids, steps
