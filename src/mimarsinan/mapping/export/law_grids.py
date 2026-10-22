"""Input and output grids of law-carrying cores: one DAC grid per core, producer grids unified (E282.S)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Set, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.integer_law import grid_levels


class ReadoutLawGridError(ToolchainRefusal, ValueError):
    """A core whose data axons do not lie on one declared grid, so its accumulator has no integer form."""


@dataclass(frozen=True)
class CoreGrid:
    """The grids one core's law reads and writes."""

    input_step: float
    input_levels: int
    output_step: float
    output_range: float


def data_sources(core: NeuralCore) -> Iterable:
    """The core's data-axon sources (off, always-on and pruned rows excluded)."""
    sources = core.input_sources.flatten()
    pruned = getattr(core, "pruned_row_mask", None)
    mask = list(pruned) if pruned is not None and len(pruned) == len(sources) else [False] * len(sources)
    for row, source in enumerate(sources):
        if source.is_off() or source.is_always_on() or mask[row]:
            continue
        yield source


def _producers(core: NeuralCore, nodes_by_id: Mapping[int, object]) -> Tuple[Set[int], bool]:
    """(perceptron indices of producer cores, whether any data axon is host- or input-sourced)."""
    producers: Set[int] = set()
    host = False
    for source in data_sources(core):
        node = nodes_by_id.get(int(source.node_id)) if source.node_id >= 0 else None
        if isinstance(node, NeuralCore):
            if node.perceptron_index is None:
                raise ReadoutLawGridError(
                    f"core {core.name!r} reads core {node.name!r}, which belongs to no perceptron: its output "
                    f"grid is undeclared"
                )
            producers.add(int(node.perceptron_index))
        else:
            host = True
    return producers, host


class _Union:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, items: Iterable[int]) -> None:
        roots = [self.find(i) for i in items]
        for root in roots[1:]:
            self.parent[root] = roots[0]


def assign_core_grids(
    ir_graph, perceptron_ranges: Mapping[int, float], activation_bits: int,
) -> Tuple[Dict[int, CoreGrid], Dict[int, float]]:
    """``({core id: CoreGrid}, {perceptron index: output step})``.

    Every perceptron's output grid spans its calibrated (or declared) range on the ``activation_bits``
    grid; perceptrons read by one core share the widest range of their union (concatenation joins), so
    each core reads one grid. A core whose data axons are host- or input-sourced reads its armed
    boundary grid; a core mixing both sources, or a host-fed core without a grid, is refused by name.
    """
    nodes_by_id = {int(n.id): n for n in ir_graph.nodes}
    cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    union = _Union()
    reads = {}
    owner: Dict[int, int] = {}
    for core in cores:
        if core.perceptron_index is None:
            raise ReadoutLawGridError(f"core {core.name!r} belongs to no perceptron: its output grid is undeclared")
        owner[int(core.id)] = int(core.perceptron_index)
        union.find(owner[int(core.id)])
        producers, host = _producers(core, nodes_by_id)
        if producers and host:
            raise ReadoutLawGridError(
                f"core {core.name!r} reads both host-sourced and core-sourced axons; one DAC grid per core"
            )
        grid = getattr(core, "boundary_grid", None)
        if host and (grid is None or not grid.armed):
            raise ReadoutLawGridError(
                f"core {core.name!r} reads host-sourced axons but carries no armed boundary grid"
            )
        union.join(producers)
        reads[int(core.id)] = (producers, host)
    missing = sorted(set(owner.values()) - set(perceptron_ranges))
    if missing:
        raise ReadoutLawGridError(f"perceptrons {missing} carry cores but no calibrated readout range")
    widest: Dict[int, float] = {}
    for index, value in perceptron_ranges.items():
        root = union.find(int(index))
        widest[root] = max(widest.get(root, 0.0), float(value))
    levels = grid_levels(activation_bits)
    steps = {int(i): widest[union.find(int(i))] / levels for i in perceptron_ranges}
    grids: Dict[int, CoreGrid] = {}
    for core in cores:
        producers, host = reads[int(core.id)]
        index = owner[int(core.id)]
        grid = core.boundary_grid
        if producers:
            in_step, in_levels = steps[next(iter(producers))], levels
        elif host and grid is not None:
            in_step, in_levels = float(grid.step), int(grid.levels)
        else:
            in_step, in_levels = 1.0, levels
        grids[int(core.id)] = CoreGrid(
            input_step=in_step, input_levels=in_levels,
            output_step=steps[index], output_range=widest[union.find(index)],
        )
    return grids, steps
