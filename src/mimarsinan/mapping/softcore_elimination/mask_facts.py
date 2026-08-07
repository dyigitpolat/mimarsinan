"""[W6b] The RETROSPECTIVE reader: softcore facts from a stored graph's masks.

The soft-core seam measures a live pruning run (:mod:`..facts`); this module
answers the same questions about an IR that has already been pruned and
stored, using the elimination masks the graph retained. Only the realized
elimination is recoverable — the weaker arms left no trace — and cores the
liveness pass deleted are gone from the denominator entirely.

The bank view applies the W3c intersection rule directly to those masks: a
bank row dies only when it is dead for EVERY sharing instance, a bank column
only when it is dead in the view of every instance whose window covers it.
"""

from __future__ import annotations

from typing import AbstractSet, Any, Sequence

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.softcore_elimination.facts import (
    BankFacts,
    InstanceFacts,
    SoftcoreEliminationError,
    SoftcoreFacts,
    column_window,
    instance_coordinates,
    neural_cores,
)
from mimarsinan.mapping.softcore_elimination.identity import mapped_layer_key
from mimarsinan.mapping.softcore_elimination.types import (
    GEOMETRY_AS_STORED,
    GEOMETRY_PRE_ELIMINATION,
)


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
    for node in neural_cores(graph):
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
            layer_key=mapped_layer_key(node, graph),
            coordinates=instance_coordinates(node),
        ))
    return SoftcoreFacts(
        instances=tuple(instances),
        banks=tuple(_bank_facts_from_masks(graph, banks)),
    )


def _bank_facts_from_masks(graph: IRGraph, banks: dict) -> list[BankFacts]:
    """Intersect the sharing instances' masks back into bank coordinates."""
    out: list[BankFacts] = []
    cores = neural_cores(graph)
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
                sharers=(), sharer_keys=(),
            ))
            continue
        rows = _intersect_row_masks(sharers, n_axons, bank_id)
        cols = _dead_in_every_view(sharers, n_neurons, bank_id)
        out.append(BankFacts(
            bank_id=int(bank_id), axons=int(n_axons), neurons=int(n_neurons),
            rows_eliminated=len(rows), cols_eliminated=len(cols),
            sharers=tuple(sorted(str(n.name) for n in sharers)),
            sharer_keys=tuple(dict.fromkeys(
                mapped_layer_key(n, graph) for n in sharers
            )),
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
        start, end = column_window(node, n_neurons)
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
