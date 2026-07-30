"""[W6b] The per-run softcore-elimination record: views, arms, serialization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from mimarsinan.mapping.softcore_elimination.types import (
    GEOMETRY_PRE_ELIMINATION,
    SOFTCORE_ELIMINATION_RECORD_FILENAME,
    GroupElimination,
    StorageElimination,
)


@dataclass(frozen=True)
class EliminationView:
    """One propagation arm's answer to both questions.

    ``groups`` are the paper-table rows (repeat indices collapsed), ``layers``
    the same measurement at mapped-layer granularity, ``storage`` the physical
    weight-storage view of the same groups.
    """

    arm: str
    groups: tuple[GroupElimination, ...]
    layers: tuple[GroupElimination, ...]
    total: GroupElimination
    storage: tuple[StorageElimination, ...]
    storage_total: StorageElimination

    def to_dict(self, *, include_layers: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arm": self.arm,
            "groups": [g.to_dict() for g in self.groups],
            "total": self.total.to_dict(),
            "storage": [s.to_dict() for s in self.storage],
            "storage_total": self.storage_total.to_dict(),
        }
        if include_layers:
            out["layers"] = [g.to_dict() for g in self.layers]
        return out


@dataclass(frozen=True)
class SoftcoreEliminationReport:
    """Weight-cell elimination in the mapped softcores of one deployment.

    ``views`` holds one :class:`EliminationView` per propagation arm that was
    actually run (masked / closure / cascade), so the same table doubles as
    the C1 evidence: the closure column minus the masked column is what
    seed-group coupling bought, and the cascade column minus the closure
    column is emergent propagation, PER SOFTCORE GROUP.
    """

    deployed_arm: str
    views: Mapping[str, EliminationView]
    geometry: str = GEOMETRY_PRE_ELIMINATION

    @property
    def view(self) -> EliminationView:
        return self.views[self.deployed_arm]

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self.views)

    @property
    def total(self) -> GroupElimination:
        return self.view.total

    @property
    def storage_total(self) -> StorageElimination:
        return self.view.storage_total

    def group(self, name: str) -> GroupElimination:
        """The deployed arm's row for one group; fails loud when absent."""
        for g in self.view.groups:
            if g.group == name:
                return g
        raise KeyError(
            f"no softcore group {name!r} in this report; groups are "
            f"{[g.group for g in self.view.groups]}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Flat record: the deployed arm inline, the weaker arms beside it."""
        record = self.view.to_dict()
        record.update({
            "deployed_arm": self.deployed_arm,
            "geometry": self.geometry,
            "arms": list(self.arms),
            "per_arm": {
                arm: view.to_dict(include_layers=False)
                for arm, view in self.views.items()
            },
        })
        return record


def write_softcore_elimination_record(
    report: SoftcoreEliminationReport, run_directory: str
) -> str:
    """Serialize the record as JSON into the run directory; returns the path."""
    os.makedirs(run_directory, exist_ok=True)
    path = os.path.join(run_directory, SOFTCORE_ELIMINATION_RECORD_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


def summarize_softcore_elimination(report: SoftcoreEliminationReport) -> str:
    """One-line run log, mirroring the crossbar-utilization report's shape."""
    total = report.total
    storage = report.storage_total
    arms = " ".join(
        f"{arm}={report.views[arm].total.eliminated_fraction:.4f}"
        for arm in report.arms
    )
    return (
        f"[SoftcoreElimination] arm={report.deployed_arm} "
        f"groups={len(report.view.groups)} instances={total.instances} "
        f"cells={total.cells} surviving={total.surviving} "
        f"eliminated={total.eliminated_fraction:.4f} "
        f"bank_cells={storage.bank_cells_before}->{storage.bank_cells_after} "
        f"({storage.bank_eliminated_fraction:.4f}) "
        f"per_arm[{arms}]"
    )
