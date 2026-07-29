"""[W6b] The measured facts behind both elimination views, read two ways.

One dataclass pair (:class:`InstanceFacts`, :class:`BankFacts`) carries every
number the report needs, and two readers produce it:

- :func:`facts_from_pruning_result` — the FIRST-CLASS path, run at the
  soft-core mapping seam on the still-uncompacted IR with one propagation
  arm's kill sets. Every mapped softcore is present at its full pre-elimination
  geometry, so the denominator is unambiguous and identical across arms;
- :func:`facts_from_masks` — the RECONSTRUCTION path, for a stored
  post-pruning IR, reading the elimination masks the graph retained.

Both apply the SAME rule a bank-backed core is compacted under (W3c): the
row/column kill set an instance realizes is the BANK's, sliced into the
instance's column window — a column starved in one instance's view but alive
in another's is not removed from the shared physical structure and therefore
is not removed from that instance's crossbar either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Any, Iterable, Sequence

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
    GlobalPruningResult,
)
from mimarsinan.mapping.softcore_elimination.types import (
    GEOMETRY_AS_STORED,
    GEOMETRY_PRE_ELIMINATION,
)


class SoftcoreEliminationError(RuntimeError):
    """The IR cannot be read into a softcore-elimination measurement."""


@dataclass(frozen=True)
class InstanceFacts:
    """One mapped softcore: its crossbar geometry and what died in it."""

    name: str
    node_id: int
    axons: int
    neurons: int
    rows_eliminated: int
    cols_eliminated: int
    weight_bank_id: int | None

    @property
    def cells(self) -> int:
        return self.axons * self.neurons

    @property
    def surviving(self) -> int:
        return (
            (self.axons - self.rows_eliminated)
            * (self.neurons - self.cols_eliminated)
        )


@dataclass(frozen=True)
class BankFacts:
    """One shared WeightBank: physical geometry and its intersection kills."""

    bank_id: int
    axons: int
    neurons: int
    rows_eliminated: int
    cols_eliminated: int
    sharers: tuple[str, ...]

    @property
    def cells(self) -> int:
        return self.axons * self.neurons

    @property
    def surviving(self) -> int:
        return (
            (self.axons - self.rows_eliminated)
            * (self.neurons - self.cols_eliminated)
        )


@dataclass(frozen=True)
class SoftcoreFacts:
    instances: tuple[InstanceFacts, ...]
    banks: tuple[BankFacts, ...]


def _neural_cores(graph: IRGraph) -> list[NeuralCore]:
    return [n for n in graph.nodes if isinstance(n, NeuralCore)]


def _column_window(node: NeuralCore, bank_neurons: int) -> tuple[int, int]:
    if node.weight_row_slice is None:
        return 0, bank_neurons
    start, end = node.weight_row_slice
    return int(start), int(end)


def _count_in_range(indices: Iterable[int], low: int, high: int) -> int:
    return sum(1 for i in indices if low <= int(i) < high)


def facts_from_pruning_result(
    graph: IRGraph, result: GlobalPruningResult
) -> SoftcoreFacts:
    """Measure one arm's kill sets against the graph's mapped softcores."""
    banks = dict(getattr(graph, "weight_banks", {}) or {})
    cores = _neural_cores(graph)
    sharer_names: dict[int, list[str]] = {bank_id: [] for bank_id in banks}
    for node in cores:
        bank_id = getattr(node, "weight_bank_id", None)
        if bank_id in sharer_names:
            sharer_names[bank_id].append(str(node.name))

    instances: list[InstanceFacts] = []
    for node in cores:
        axons, neurons = node.get_core_matrix(graph).shape
        bank_id = getattr(node, "weight_bank_id", None)
        if bank_id is not None and bank_id in banks:
            start, end = _column_window(node, banks[bank_id].core_matrix.shape[1])
            rows = _count_in_range(
                result.pruned_rows_per_bank.get(bank_id, set()), 0, axons
            )
            cols = _count_in_range(
                result.pruned_cols_per_bank.get(bank_id, set()), start, end
            )
        else:
            rows = _count_in_range(
                result.pruned_rows_per_node.get(node.id, set()), 0, axons
            )
            cols = _count_in_range(
                result.pruned_cols_per_node.get(node.id, set()), 0, neurons
            )
        instances.append(InstanceFacts(
            name=str(node.name), node_id=int(node.id),
            axons=int(axons), neurons=int(neurons),
            rows_eliminated=rows, cols_eliminated=cols,
            weight_bank_id=bank_id if bank_id in banks else None,
        ))

    bank_facts = tuple(
        BankFacts(
            bank_id=int(bank_id),
            axons=int(bank.core_matrix.shape[0]),
            neurons=int(bank.core_matrix.shape[1]),
            rows_eliminated=_count_in_range(
                result.pruned_rows_per_bank.get(bank_id, set()),
                0, bank.core_matrix.shape[0],
            ),
            cols_eliminated=_count_in_range(
                result.pruned_cols_per_bank.get(bank_id, set()),
                0, bank.core_matrix.shape[1],
            ),
            sharers=tuple(sorted(sharer_names[bank_id])),
        )
        for bank_id, bank in banks.items()
    )
    return SoftcoreFacts(instances=tuple(instances), banks=bank_facts)


def _masks_of(node: NeuralCore, geometry: str) -> tuple[Any, Any]:
    """The row/col masks that describe this core under ``geometry``."""
    if geometry == GEOMETRY_PRE_ELIMINATION:
        rows = (
            node.pre_pruning_row_mask
            if node.pre_pruning_row_mask is not None
            else node.pruned_row_mask
        )
        cols = (
            node.pre_pruning_col_mask
            if node.pre_pruning_col_mask is not None
            else node.pruned_col_mask
        )
        return rows, cols
    if geometry == GEOMETRY_AS_STORED:
        return node.pruned_row_mask, node.pruned_col_mask
    raise SoftcoreEliminationError(
        f"unknown geometry {geometry!r}; expected "
        f"{GEOMETRY_PRE_ELIMINATION!r} or {GEOMETRY_AS_STORED!r}"
    )


def facts_from_masks(
    graph: IRGraph, *, geometry: str = GEOMETRY_PRE_ELIMINATION
) -> SoftcoreFacts:
    """Reconstruct the measurement from a stored graph's retained masks.

    ``geometry="pre_elimination"`` (default) measures against each core's
    PRE-elimination crossbar — the honest denominator, and the one the
    soft-core seam emits, recovered from ``pre_pruning_*_mask`` where IR
    compaction has already shrunk an owned matrix. ``geometry="as_stored"``
    measures against the geometry the node currently carries, which for an
    already-compacted owned core silently drops the rows it lost.
    """
    instances: list[InstanceFacts] = []
    banks = dict(getattr(graph, "weight_banks", {}) or {})
    for node in _neural_cores(graph):
        row_mask, col_mask = _masks_of(node, geometry)
        if row_mask is None or col_mask is None:
            axons, neurons = node.get_core_matrix(graph).shape
            rows = cols = 0
        else:
            axons, neurons = len(row_mask), len(col_mask)
            rows = sum(1 for v in row_mask if v)
            cols = sum(1 for v in col_mask if v)
        bank_id = getattr(node, "weight_bank_id", None)
        instances.append(InstanceFacts(
            name=str(node.name), node_id=int(node.id),
            axons=int(axons), neurons=int(neurons),
            rows_eliminated=int(rows), cols_eliminated=int(cols),
            weight_bank_id=bank_id if bank_id in banks else None,
        ))
    return SoftcoreFacts(
        instances=tuple(instances),
        banks=tuple(_bank_facts_from_masks(graph, banks)),
    )


def _bank_facts_from_masks(graph: IRGraph, banks: dict) -> list[BankFacts]:
    """Intersect the sharing instances' masks back into bank coordinates."""
    out: list[BankFacts] = []
    cores = _neural_cores(graph)
    for bank_id, bank in banks.items():
        n_axons, n_neurons = bank.core_matrix.shape
        sharers = [
            n for n in cores
            if getattr(n, "weight_bank_id", None) == bank_id
        ]
        if not sharers:
            # No surviving instance references this storage at all.
            out.append(BankFacts(
                bank_id=int(bank_id), axons=int(n_axons), neurons=int(n_neurons),
                rows_eliminated=int(n_axons), cols_eliminated=int(n_neurons),
                sharers=(),
            ))
            continue
        rows = _intersect_row_masks(sharers, n_axons, bank_id)
        cols = _dead_in_every_view(sharers, n_neurons, bank_id)
        out.append(BankFacts(
            bank_id=int(bank_id), axons=int(n_axons), neurons=int(n_neurons),
            rows_eliminated=len(rows), cols_eliminated=len(cols),
            sharers=tuple(sorted(str(n.name) for n in sharers)),
        ))
    return out


def _intersect_row_masks(
    sharers: Sequence[NeuralCore], n_axons: int, bank_id: int
) -> AbstractSet[int]:
    """A bank row is eliminated only when dead for EVERY sharing instance."""
    intersection: set[int] | None = None
    for node in sharers:
        mask = node.pruned_row_mask
        if mask is None:
            return set()
        if len(mask) != n_axons:
            raise SoftcoreEliminationError(
                f"bank {bank_id}: core {node.name!r} carries a row mask of "
                f"length {len(mask)} against {n_axons} bank axons; the masks "
                "must stay in bank coordinates for the intersection rule."
            )
        dead = {i for i, v in enumerate(mask) if v}
        intersection = dead if intersection is None else (intersection & dead)
    return intersection or set()


def _dead_in_every_view(
    sharers: Sequence[NeuralCore], n_neurons: int, bank_id: int
) -> AbstractSet[int]:
    """A bank column is eliminated only when dead in the view of every
    instance whose column window covers it; uncovered columns are unmapped."""
    covered = [False] * n_neurons
    dead_everywhere = [True] * n_neurons
    for node in sharers:
        start, end = _column_window(node, n_neurons)
        mask = node.pruned_col_mask
        if mask is None:
            mask = [False] * (end - start)
        if len(mask) != end - start:
            raise SoftcoreEliminationError(
                f"bank {bank_id}: core {node.name!r} carries a column mask of "
                f"length {len(mask)} against a window of {end - start} "
                "columns; the masks must match the instance's bank window."
            )
        for j in range(start, end):
            covered[j] = True
            if not mask[j - start]:
                dead_everywhere[j] = False
    return {j for j in range(n_neurons) if covered[j] and dead_everywhere[j]}
