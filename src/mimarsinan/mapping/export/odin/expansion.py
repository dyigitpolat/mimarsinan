"""Row-pair expansion: a signed logical crossbar becomes physical rows with row signs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from mimarsinan.chip_simulation.soma_axes import PER_AXON_SIGN
from mimarsinan.mapping.export.odin.feasibility import check_weight_magnitudes
from mimarsinan.mapping.platform.event_order import (
    bias_tail_slots,
    canonical_slot_order,
    excitatory_row,
    inhibitory_row,
)


class RowPairExpansionError(ValueError):
    """The logical grid is not a shape the row-pair expansion can consume."""


@dataclass(frozen=True)
class PhysicalRow:
    """One physical crossbar row: a logical slot's excitatory or inhibitory half."""

    row: int
    slot: int
    inhibitory: bool
    always_on: bool
    magnitudes: np.ndarray

    @property
    def emits(self) -> bool:
        """All-zero rows are no-ops under the pair lemma, so nothing is injected."""
        return bool(self.magnitudes.any())


@dataclass(frozen=True)
class RowPairExpansion:
    """The physical rows of one core, in ascending physical-row order."""

    rows: Tuple[PhysicalRow, ...]
    bias_slots: Tuple[int, ...]

    @property
    def syn_sign_bits(self) -> Tuple[bool, ...]:
        return tuple(row.inhibitory for row in self.rows)

    def emitting_rows_for_slot(self, slot: int) -> Tuple[int, ...]:
        """The physical rows a single occurrence of ``slot`` must actually drive."""
        pair = (excitatory_row(slot), inhibitory_row(slot))
        return tuple(row for row in pair if self.rows[row].emits)


def expand_row_pairs(
    matrix: Any, *, n_bias_rows: int, weight_bits: int, core_index: int = 0
) -> RowPairExpansion:
    """Split each signed logical slot into its ``(2a, 2a+1)`` exc/inh physical pair.

    Allocates FRESH arrays: ``HardCore.get_core_matrix()`` hands back a shared
    memoized grid that callers must never mutate.
    """
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise RowPairExpansionError(
            f"the logical crossbar must be 2-D (slots x neurons), got shape "
            f"{values.shape}")
    rounded = np.rint(values.astype(np.float64))
    if not np.allclose(values.astype(np.float64), rounded, atol=0.0):
        raise RowPairExpansionError(
            "the logical crossbar must be integral before packing; quantization "
            "delivers integer weights and a silent truncation would change the "
            "deployed physics")
    signed = rounded.astype(np.int64)
    check_weight_magnitudes(
        signed, weight_bits=weight_bits,
        weight_sign_granularity=PER_AXON_SIGN, core_index=core_index,
    )
    n_slots = int(signed.shape[0])
    bias = tuple(bias_tail_slots(n_slots, int(n_bias_rows)))
    rows = []
    for slot in canonical_slot_order(n_slots):
        column = signed[slot]
        rows.append(PhysicalRow(
            row=excitatory_row(slot), slot=slot, inhibitory=False,
            always_on=slot in bias,
            magnitudes=np.maximum(column, 0).astype(np.int64),
        ))
        rows.append(PhysicalRow(
            row=inhibitory_row(slot), slot=slot, inhibitory=True,
            always_on=slot in bias,
            magnitudes=np.maximum(-column, 0).astype(np.int64),
        ))
    return RowPairExpansion(rows=tuple(rows), bias_slots=bias)
