"""[W6b] The measured facts behind both elimination views, and the LIVE reader.

One dataclass pair (:class:`InstanceFacts`, :class:`BankFacts`) carries every
number the report needs, and two readers produce it:

- :func:`facts_from_pruning_result` (here) — the FIRST-CLASS path, run at the
  soft-core mapping seam on the still-uncompacted IR with one propagation
  arm's kill sets. Every mapped softcore is present at its full pre-elimination
  geometry, so the denominator is unambiguous and identical across arms;
- :func:`..mask_facts.facts_from_masks` — the RECONSTRUCTION path, for a
  stored post-pruning IR, reading the elimination masks the graph retained.

Both apply the SAME rule a bank-backed core is compacted under (W3c): the
row/column kill set an instance realizes is the BANK's, sliced into the
instance's column window — a column starved in one instance's view but alive
in another's is not removed from the shared physical structure and therefore
is not removed from that instance's crossbar either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
    GlobalPruningResult,
)
from mimarsinan.mapping.softcore_elimination.identity import (
    MappedLayerKey,
    mapped_layer_key,
)


class SoftcoreEliminationError(RuntimeError):
    """The IR cannot be read into a softcore-elimination measurement."""


@dataclass(frozen=True)
class InstanceFacts:
    """One mapped softcore: its crossbar geometry and what died in it.

    ``layer_key`` is the STRUCTURAL mapped-layer identity this instance is
    bucketed under; ``name`` is carried for the display label only.
    """

    name: str
    node_id: int
    axons: int
    neurons: int
    rows_eliminated: int
    cols_eliminated: int
    weight_bank_id: int | None
    layer_key: MappedLayerKey
    #: the instance's own structural coordinates, as digit tuples — the values
    #: a mapper writes into the positional suffix of its name. Display only.
    coordinates: tuple[tuple[int, ...], ...] = ()

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
    #: the STRUCTURAL mapped layers this storage is shared by.
    sharer_keys: tuple[MappedLayerKey, ...] = ()

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


def neural_cores(graph: IRGraph) -> list[NeuralCore]:
    return [n for n in graph.nodes if isinstance(n, NeuralCore)]


def column_window(node: NeuralCore, bank_neurons: int) -> tuple[int, int]:
    if node.weight_row_slice is None:
        return 0, bank_neurons
    start, end = node.weight_row_slice
    return int(start), int(end)


def _count_in_range(indices: Iterable[int], low: int, high: int) -> int:
    return sum(1 for i in indices if low <= int(i) < high)


def instance_coordinates(node: NeuralCore) -> tuple[tuple[int, ...], ...]:
    """The structural coordinates a mapper encodes into an instance's name."""
    out: list[tuple[int, ...]] = []
    column = getattr(node, "perceptron_output_column", None)
    if column is not None:
        out.append((int(column),))
    for attr in (
        "perceptron_output_slice", "perceptron_input_slice", "weight_row_slice"
    ):
        window = getattr(node, attr, None)
        if window is not None:
            out.append((int(window[0]), int(window[1])))
    return tuple(dict.fromkeys(out))


def facts_from_pruning_result(
    graph: IRGraph, result: GlobalPruningResult
) -> SoftcoreFacts:
    """Measure one arm's kill sets against the graph's mapped softcores."""
    banks = dict(getattr(graph, "weight_banks", {}) or {})
    cores = neural_cores(graph)
    sharer_names: dict[int, list[str]] = {bank_id: [] for bank_id in banks}
    sharer_keys: dict[int, list[MappedLayerKey]] = {
        bank_id: [] for bank_id in banks
    }
    for node in cores:
        bank_id = getattr(node, "weight_bank_id", None)
        if bank_id in sharer_names:
            sharer_names[bank_id].append(str(node.name))
            sharer_keys[bank_id].append(mapped_layer_key(node, graph))

    instances: list[InstanceFacts] = []
    for node in cores:
        axons, neurons = node.get_core_matrix(graph).shape
        bank_id = getattr(node, "weight_bank_id", None)
        if bank_id is not None and bank_id in banks:
            start, end = column_window(node, banks[bank_id].core_matrix.shape[1])
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
            layer_key=mapped_layer_key(node, graph),
            coordinates=instance_coordinates(node),
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
            sharer_keys=tuple(dict.fromkeys(sharer_keys[bank_id])),
        )
        for bank_id, bank in banks.items()
    )
    return SoftcoreFacts(instances=tuple(instances), banks=bank_facts)
