"""Seed admission, reference zeroing, and the shared-bank union rule check."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.certificate.errors import (
    CascadeCertificateError,
    CascadeCertificatePreconditionError,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult

SeedMasks = Dict[int, Tuple[Sequence[bool], Sequence[bool]]]


def check_shared_bank_union_rule(
    ir_graph: IRGraph, result: GlobalPruningResult
) -> int:
    """A bank row/column may be eliminated only if dead for ALL sharing instances.

    Returns the number of eliminated bank columns verified; raises on violation.
    """
    checked = 0
    banks = getattr(ir_graph, "weight_banks", None) or {}
    for bank_id, bank in banks.items():
        nodes = [
            n for n in ir_graph.nodes
            if isinstance(n, NeuralCore)
            and getattr(n, "weight_bank_id", None) == bank_id
        ]
        n_neurons = bank.core_matrix.shape[1]
        for col in sorted(result.pruned_cols_per_bank.get(bank_id, set())):
            checked += 1
            for node in nodes:
                start, end = node.weight_row_slice or (0, n_neurons)
                if not start <= col < end:
                    continue
                if (col - start) not in result.pruned_cols_per_node.get(
                    node.id, set()
                ):
                    raise CascadeCertificateError(
                        f"shared-bank union rule violated: bank {bank_id} "
                        f"column {col} was eliminated but sharing node "
                        f"{node.id} ({node.name!r}) still holds its local "
                        f"column {col - start} live."
                    )
        for row in sorted(result.pruned_rows_per_bank.get(bank_id, set())):
            for node in nodes:
                if row not in result.pruned_rows_per_node.get(node.id, set()):
                    raise CascadeCertificateError(
                        f"shared-bank union rule violated: bank {bank_id} "
                        f"row {row} was eliminated but sharing node "
                        f"{node.id} ({node.name!r}) still holds it live."
                    )
    return checked


def refuse_unusable_seeds(
    ir_graph: IRGraph,
    initial_pruned_per_node: SeedMasks | None,
    initial_pruned_per_bank: SeedMasks | None,
) -> None:
    """Seeds the cascade would silently ignore are refused loudly instead."""
    node_by_id = {
        n.id: n for n in ir_graph.nodes if isinstance(n, NeuralCore)
    }
    for nid in (initial_pruned_per_node or {}):
        node = node_by_id.get(nid)
        if node is None:
            raise CascadeCertificatePreconditionError(
                f"initial_pruned_per_node targets unknown NeuralCore id={nid}."
            )
        if node.core_matrix is None:
            raise CascadeCertificatePreconditionError(
                f"initial_pruned_per_node targets bank-backed NeuralCore "
                f"id={nid}; the cascade silently ignores such seeds — pass "
                "initial_pruned_per_bank instead."
            )
    banks = getattr(ir_graph, "weight_banks", None) or {}
    for bid, (rows, cols) in (initial_pruned_per_bank or {}).items():
        bank = banks.get(bid)
        if bank is None:
            raise CascadeCertificatePreconditionError(
                f"initial_pruned_per_bank targets unknown WeightBank id={bid}."
            )
        n_axons, n_neurons = bank.core_matrix.shape
        if len(rows) != n_axons or len(cols) != n_neurons:
            raise CascadeCertificatePreconditionError(
                f"initial_pruned_per_bank masks for bank {bid} have shape "
                f"({len(rows)}, {len(cols)}) but the bank is "
                f"({n_axons}, {n_neurons}); the cascade silently drops "
                "mismatched bank masks."
            )


def _bank_exemptions(
    ir_graph: IRGraph, bank_id: int, exempt_rows, exempt_cols
) -> Tuple[set, set]:
    banks = getattr(ir_graph, "weight_banks", None) or {}
    n_neurons = banks[bank_id].core_matrix.shape[1]
    rows: set = set()
    cols: set = set()
    for node in ir_graph.nodes:
        if (
            not isinstance(node, NeuralCore)
            or getattr(node, "weight_bank_id", None) != bank_id
        ):
            continue
        start, _end = node.weight_row_slice or (0, n_neurons)
        rows |= set(exempt_rows.get(node.id, frozenset()))
        cols |= {start + j for j in exempt_cols.get(node.id, frozenset())}
    return rows, cols


def apply_admitted_seed_zeroing(
    reference: IRGraph,
    seed_per_node: Dict[int, Tuple[set, set]],
    seed_per_bank: Dict[int, Tuple[set, set]],
    exempt_rows,
    exempt_cols,
) -> None:
    """Zero exactly the seeds the cascade admits (boundary policy mirrored)."""
    node_by_id = {
        n.id: n for n in reference.nodes if isinstance(n, NeuralCore)
    }
    for nid, (rows, cols) in seed_per_node.items():
        node = node_by_id[nid]
        matrix = node.core_matrix
        assert matrix is not None, (
            f"admitted per-node seed targets NeuralCore id={nid} without an "
            "owned core_matrix; refuse_unusable_seeds must run first."
        )
        rows_adm = sorted(set(rows) - set(exempt_rows.get(nid, frozenset())))
        cols_adm = sorted(set(cols) - set(exempt_cols.get(nid, frozenset())))
        if rows_adm:
            matrix[rows_adm, :] = 0.0
        if cols_adm:
            matrix[:, cols_adm] = 0.0
            bias = getattr(node, "hardware_bias", None)
            if bias is not None:
                bias[cols_adm] = 0.0
    banks = getattr(reference, "weight_banks", None) or {}
    for bid, (rows, cols) in seed_per_bank.items():
        ex_rows, ex_cols = _bank_exemptions(
            reference, bid, exempt_rows, exempt_cols
        )
        bank = banks[bid]
        rows_adm = sorted(set(rows) - ex_rows)
        cols_adm = sorted(set(cols) - ex_cols)
        if rows_adm:
            bank.core_matrix[rows_adm, :] = 0.0
        if cols_adm:
            bank.core_matrix[:, cols_adm] = 0.0
            bias = getattr(bank, "hardware_bias", None)
            if bias is not None:
                bias[cols_adm] = 0.0
