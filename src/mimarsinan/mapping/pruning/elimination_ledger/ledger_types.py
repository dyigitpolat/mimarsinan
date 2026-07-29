"""Elimination-ledger record types: per-kill attribution and depth (W3)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

KILL_CAUSE_SEED = "seed"
KILL_CAUSE_CLOSURE_COUPLING = "closure_coupling"
KILL_CAUSE_EMERGENT_PROPAGATION = "emergent_propagation"
KILL_CAUSE_LIVENESS_DEAD = "liveness_dead"
# [W4b-2] an axon row whose line was a known CONSTANT and whose
# contribution was folded onto the core's constant carrier.
KILL_CAUSE_CONSTANT_FOLD = "constant_fold"

ELIMINATION_LEDGER_RECORD_FILENAME = "elimination_ledger.json"


class EliminationLedgerError(RuntimeError):
    """The ledger failed to reconcile against the production kill sets."""


@dataclass(frozen=True)
class EliminationCounts:
    """Rows/columns killed, split by attribution category."""

    seed_rows: int = 0
    closure_rows: int = 0
    emergent_rows: int = 0
    liveness_rows: int = 0
    constant_rows: int = 0
    seed_cols: int = 0
    closure_cols: int = 0
    emergent_cols: int = 0
    liveness_cols: int = 0

    @property
    def total_rows(self) -> int:
        return (
            self.seed_rows + self.closure_rows + self.emergent_rows
            + self.liveness_rows + self.constant_rows
        )

    @property
    def total_cols(self) -> int:
        return (
            self.seed_cols + self.closure_cols
            + self.emergent_cols + self.liveness_cols
        )


@dataclass(frozen=True)
class NodeEliminationRecord:
    """One NeuralCore's eliminations; depths cover fixpoint kills only
    (seed = 0; a kill caused by depth-d structure is d+1).

    ``counts.constant_rows`` [W4b-2] carves the CONSTANT-FOLD kills out of the
    arm-difference buckets so the categories stay a partition of the kill set.
    """

    node_id: int
    name: str
    n_axons: int
    n_neurons: int
    counts: EliminationCounts
    row_depths: Mapping[int, int]
    col_depths: Mapping[int, int]


@dataclass(frozen=True)
class BankEliminationRecord:
    """One shared WeightBank's eliminations in bank coordinates."""

    bank_id: int
    n_axons: int
    n_neurons: int
    counts: EliminationCounts


@dataclass(frozen=True)
class EliminationLedger:
    """Attribution of every eliminated row/column of one pruning run.

    Totals reconcile against the compacted shapes: for a surviving owned core
    the compacted shape is (n_axons - total_rows, n_neurons - total_cols),
    with the BIAS_ONLY single-placeholder-row exception.
    """

    mode: str
    per_node: tuple
    per_bank: tuple
    cores_deleted: int
    bias_only_collapses: int
    fixpoint_iterations: int

    def _sum(self, field: str) -> int:
        return (
            sum(getattr(r.counts, field) for r in self.per_node)
            + sum(getattr(r.counts, field) for r in self.per_bank)
        )

    @property
    def seed_rows(self) -> int:
        return self._sum("seed_rows")

    @property
    def seed_cols(self) -> int:
        return self._sum("seed_cols")

    @property
    def closure_coupling_rows(self) -> int:
        return self._sum("closure_rows")

    @property
    def closure_coupling_cols(self) -> int:
        return self._sum("closure_cols")

    @property
    def emergent_propagation_rows(self) -> int:
        return self._sum("emergent_rows")

    @property
    def emergent_propagation_cols(self) -> int:
        return self._sum("emergent_cols")

    @property
    def constant_fold_rows(self) -> int:
        return self._sum("constant_rows")

    @property
    def liveness_dead_rows(self) -> int:
        return self._sum("liveness_rows")

    @property
    def liveness_dead_cols(self) -> int:
        return self._sum("liveness_cols")

    def _all_depths(self) -> list:
        out: list = []
        for r in self.per_node:
            out.extend(r.row_depths.values())
            out.extend(r.col_depths.values())
        return out

    @property
    def max_propagation_depth(self) -> int:
        depths = self._all_depths()
        return max(depths) if depths else 0

    @property
    def mean_propagation_depth(self) -> float:
        """Mean depth over propagated (depth >= 1) kills; 0.0 when none."""
        propagated = [d for d in self._all_depths() if d >= 1]
        return (sum(propagated) / len(propagated)) if propagated else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Flat scalar record — one row of the experiment table."""
        return {
            "mode": self.mode,
            "seed_rows": self.seed_rows,
            "seed_cols": self.seed_cols,
            "closure_coupling_rows": self.closure_coupling_rows,
            "closure_coupling_cols": self.closure_coupling_cols,
            "emergent_propagation_rows": self.emergent_propagation_rows,
            "emergent_propagation_cols": self.emergent_propagation_cols,
            "liveness_dead_rows": self.liveness_dead_rows,
            "liveness_dead_cols": self.liveness_dead_cols,
            "constant_fold_rows": self.constant_fold_rows,
            "total_rows_eliminated": self._sum("seed_rows")
            + self._sum("closure_rows") + self._sum("emergent_rows")
            + self._sum("liveness_rows") + self._sum("constant_rows"),
            "total_cols_eliminated": self._sum("seed_cols")
            + self._sum("closure_cols") + self._sum("emergent_cols")
            + self._sum("liveness_cols"),
            "cores_deleted": self.cores_deleted,
            "bias_only_collapses": self.bias_only_collapses,
            "fixpoint_iterations": self.fixpoint_iterations,
            "max_propagation_depth": self.max_propagation_depth,
            "mean_propagation_depth": self.mean_propagation_depth,
            "nodes_recorded": len(self.per_node),
            "banks_recorded": len(self.per_bank),
        }

    def summary(self) -> str:
        return (
            f"[Ledger] mode={self.mode} "
            f"seed r/c={self.seed_rows}/{self.seed_cols} "
            f"closure r/c={self.closure_coupling_rows}/"
            f"{self.closure_coupling_cols} "
            f"emergent r/c={self.emergent_propagation_rows}/"
            f"{self.emergent_propagation_cols} "
            f"liveness r/c={self.liveness_dead_rows}/{self.liveness_dead_cols} "
            f"constant_fold rows={self.constant_fold_rows} "
            f"cores_deleted={self.cores_deleted} "
            f"bias_only={self.bias_only_collapses} "
            f"iters={self.fixpoint_iterations} "
            f"max_depth={self.max_propagation_depth}"
        )


def write_elimination_ledger_record(
    ledger: EliminationLedger, run_directory: str
) -> str:
    """Serialize the flat record as JSON into the run directory; returns the path."""
    path = os.path.join(run_directory, ELIMINATION_LEDGER_RECORD_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_dict(), f, indent=2)
    return path
