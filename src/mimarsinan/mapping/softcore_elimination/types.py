"""[W6b] The two labelled elimination views, as flat serializable records.

The method targets the weights that OCCUPY CROSSBAR SOFTCORES, and that
target admits two different honest denominators:

- :class:`GroupElimination` — the AS-MAPPED (per-crossbar occupancy) view.
  Every mapped softcore instance counts its own ``a x n`` cells, so a weight
  matrix instantiated on 65 crossbars contributes 65 times. This is the MAC
  sites / crossbar area question, and it is the paper's headline row.
- :class:`StorageElimination` — the PHYSICAL (weight storage) view. Shared
  structure is counted ONCE, and a bank row/column is eliminated only when it
  is dead for EVERY sharing instance (the W3c intersection rule). This is the
  weight-memory question and it is strictly the more conservative one.

Both are kept in the record, each labelled; neither substitutes for the other.
In both views a cell is eliminated when its row OR its column is eliminated,
so ``surviving = (R - r) * (C - c)`` per instance / per storage unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# How the as-mapped crossbar geometry (the DENOMINATOR) is established.
# "pre_elimination": every core measured against the crossbar it needed
#   BEFORE elimination — uniform, and what the soft-core seam emits.
# "as_stored": measured against the geometry the stored IR still carries,
#   which for an already-compacted owned matrix silently drops the rows it
#   lost; retrospective only.
GEOMETRY_PRE_ELIMINATION = "pre_elimination"
GEOMETRY_AS_STORED = "as_stored"

SOFTCORE_ELIMINATION_RECORD_FILENAME = "softcore_elimination.json"
SOFTCORE_ELIMINATION_TABLE_FILENAME = "softcore_elimination.md"

# Group label for storage whose sharing instances span several groups, and for
# storage no surviving instance references any more.
SHARED_GROUP_LABEL = "<shared>"
UNMAPPED_GROUP_LABEL = "<unmapped>"
TOTAL_GROUP_LABEL = "TOTAL"


def _fraction(part: int, whole: int) -> float:
    return (part / whole) if whole else 0.0


@dataclass(frozen=True)
class GroupElimination:
    """As-mapped weight cells of one softcore group (or of the whole program).

    ``rows_eliminated`` / ``cols_eliminated`` are SUMMED over the group's
    instances; ``axons`` / ``neurons`` are the shared crossbar dimensions and
    are None when the group's instances do not share one geometry.
    """

    group: str
    instances: int
    axons: int | None
    neurons: int | None
    rows_eliminated: int
    cols_eliminated: int
    cells: int
    surviving: int
    layers: tuple[str, ...] = ()

    @property
    def eliminated(self) -> int:
        return self.cells - self.surviving

    @property
    def eliminated_fraction(self) -> float:
        return _fraction(self.eliminated, self.cells)

    @property
    def rows_eliminated_per_instance(self) -> float:
        return _fraction(self.rows_eliminated, self.instances)

    @property
    def cols_eliminated_per_instance(self) -> float:
        return _fraction(self.cols_eliminated, self.instances)

    @property
    def dimensions(self) -> str:
        if self.axons is None or self.neurons is None:
            return "mixed"
        return f"{self.axons}x{self.neurons}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "instances": self.instances,
            "axons": self.axons,
            "neurons": self.neurons,
            "rows_eliminated": self.rows_eliminated,
            "cols_eliminated": self.cols_eliminated,
            "rows_eliminated_per_instance": self.rows_eliminated_per_instance,
            "cols_eliminated_per_instance": self.cols_eliminated_per_instance,
            "cells": self.cells,
            "surviving": self.surviving,
            "eliminated": self.eliminated,
            "eliminated_fraction": self.eliminated_fraction,
            "layers": list(self.layers),
        }


def aggregate_groups(
    groups: "tuple[GroupElimination, ...]", *, label: str = TOTAL_GROUP_LABEL
) -> GroupElimination:
    """Sum groups into one row; dimensions survive only if all groups agree."""
    dims = {(g.axons, g.neurons) for g in groups}
    axons, neurons = dims.pop() if len(dims) == 1 else (None, None)
    layers: list[str] = []
    for g in groups:
        layers.extend(g.layers)
    return GroupElimination(
        group=label,
        instances=sum(g.instances for g in groups),
        axons=axons,
        neurons=neurons,
        rows_eliminated=sum(g.rows_eliminated for g in groups),
        cols_eliminated=sum(g.cols_eliminated for g in groups),
        cells=sum(g.cells for g in groups),
        surviving=sum(g.surviving for g in groups),
        layers=tuple(sorted(layers)),
    )


@dataclass(frozen=True)
class StorageElimination:
    """Distinct physical weight storage of one group: shared banks counted
    once (intersection rule) plus every unshared owned crossbar matrix."""

    group: str
    banks: int
    bank_cells_before: int
    bank_cells_after: int
    owned_matrices: int
    owned_cells_before: int
    owned_cells_after: int

    @property
    def units(self) -> int:
        return self.banks + self.owned_matrices

    @property
    def cells_before(self) -> int:
        return self.bank_cells_before + self.owned_cells_before

    @property
    def cells_after(self) -> int:
        return self.bank_cells_after + self.owned_cells_after

    @property
    def eliminated_fraction(self) -> float:
        return _fraction(
            self.cells_before - self.cells_after, self.cells_before
        )

    @property
    def bank_eliminated_fraction(self) -> float:
        return _fraction(
            self.bank_cells_before - self.bank_cells_after,
            self.bank_cells_before,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "units": self.units,
            "banks": self.banks,
            "bank_cells_before": self.bank_cells_before,
            "bank_cells_after": self.bank_cells_after,
            "bank_eliminated_fraction": self.bank_eliminated_fraction,
            "owned_matrices": self.owned_matrices,
            "owned_cells_before": self.owned_cells_before,
            "owned_cells_after": self.owned_cells_after,
            "cells_before": self.cells_before,
            "cells_after": self.cells_after,
            "eliminated_fraction": self.eliminated_fraction,
        }


def aggregate_storage(
    storages: "tuple[StorageElimination, ...]", *, label: str = TOTAL_GROUP_LABEL
) -> StorageElimination:
    """Sum physical-storage rows; every unit is already counted once."""
    return StorageElimination(
        group=label,
        banks=sum(s.banks for s in storages),
        bank_cells_before=sum(s.bank_cells_before for s in storages),
        bank_cells_after=sum(s.bank_cells_after for s in storages),
        owned_matrices=sum(s.owned_matrices for s in storages),
        owned_cells_before=sum(s.owned_cells_before for s in storages),
        owned_cells_after=sum(s.owned_cells_after for s in storages),
    )
