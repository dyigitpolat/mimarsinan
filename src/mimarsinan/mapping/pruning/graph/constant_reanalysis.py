"""Retrospective re-harvest: re-analyze an existing IR + seeds under the lattice.

The whole point of an exact, purely structural analysis is that it costs no
GPU: a run that was already deployed and measured can be re-scored under the
constant lattice from its persisted IR and seed masks alone, with no retrain,
no re-simulation, and no mutation of the artifact. :func:`reanalyze_constant_folding`
is that entry point — a PURE function that runs the requested arm twice
(folding off, then full) on the same graph and reports the delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    DEFAULT_ELIMINATION_PROPAGATION,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
    compute_global_pruned_sets,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
    GlobalPruningResult,
)
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
    _collect_initial_seeds,
)
from mimarsinan.mapping.pruning.liveness_transfer import (
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    ELIMINATION_CONSTANT_FOLDING_FULL,
    ELIMINATION_CONSTANT_FOLDING_OFF,
)

__all__ = ["ConstantFoldingReport", "reanalyze_constant_folding"]

SeedMasks = Dict[int, Tuple[Sequence[bool], Sequence[bool]]]


def _kills(result: GlobalPruningResult) -> int:
    return (
        sum(len(s) for s in result.pruned_rows_per_node.values())
        + sum(len(s) for s in result.pruned_cols_per_node.values())
        + sum(len(s) for s in result.pruned_rows_per_bank.values())
        + sum(len(s) for s in result.pruned_cols_per_bank.values())
    )


@dataclass(frozen=True)
class ConstantFoldingReport:
    """What the constant lattice adds to an already-analyzed artifact."""

    mode: str
    spiking_mode: str
    baseline_kills: int
    folded_kills: int
    folded_rows: int
    constant_lines: int
    cores_with_folds: int

    @property
    def additional_kills(self) -> int:
        return self.folded_kills - self.baseline_kills

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "spiking_mode": self.spiking_mode,
            "baseline_kills": self.baseline_kills,
            "folded_kills": self.folded_kills,
            "additional_kills": self.additional_kills,
            "folded_rows": self.folded_rows,
            "constant_lines": self.constant_lines,
            "cores_with_folds": self.cores_with_folds,
        }

    def summary(self) -> str:
        return (
            f"[ConstFold] mode={self.mode} domain={self.spiking_mode} "
            f"kills {self.baseline_kills}->{self.folded_kills} "
            f"(+{self.additional_kills}) folded_rows={self.folded_rows} "
            f"const_lines={self.constant_lines} "
            f"cores_with_folds={self.cores_with_folds}"
        )


def reanalyze_constant_folding(
    ir_graph: IRGraph,
    *,
    initial_pruned_per_node: SeedMasks | None = None,
    initial_pruned_per_bank: SeedMasks | None = None,
    zero_threshold: float = 1e-8,
    elimination_propagation: str = DEFAULT_ELIMINATION_PROPAGATION,
    computeop_liveness_transfers: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    spiking_mode: str = "lif",
) -> ConstantFoldingReport:
    """Re-score one stored (uncompacted) IR + seed masks under the lattice.

    PURE: reads the graph, mutates nothing (the folds live only in the returned
    analysis state), touches no accelerator. Pass the SAME arm and
    ``spiking_mode`` the original run used, or the comparison is not a
    like-for-like re-harvest.
    """
    exempt_rows, exempt_cols = _boundary_policy_exemptions(ir_graph)
    seed_per_node, seed_per_bank = _collect_initial_seeds(
        ir_graph, initial_pruned_per_node, initial_pruned_per_bank
    )

    def _run(folding: str) -> GlobalPruningResult:
        return compute_global_pruned_sets(
            ir_graph,
            zero_threshold=zero_threshold,
            initial_per_node=seed_per_node,
            initial_per_bank=seed_per_bank,
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
            mode=elimination_propagation,
            computeop_liveness_transfers=computeop_liveness_transfers,
            elimination_constant_folding=folding,
            spiking_mode=spiking_mode,
        )

    baseline = _run(ELIMINATION_CONSTANT_FOLDING_OFF)
    folded = _run(ELIMINATION_CONSTANT_FOLDING_FULL)
    state = folded.constant_folds
    return ConstantFoldingReport(
        mode=str(elimination_propagation),
        spiking_mode=str(spiking_mode),
        baseline_kills=_kills(baseline),
        folded_kills=_kills(folded),
        folded_rows=state.total_folded_rows(),
        constant_lines=len(state.lattice.values),
        cores_with_folds=len(state.folded_rows),
    )


def constant_line_values(result: GlobalPruningResult) -> Mapping[Tuple[int, int], float]:
    """The CONST lines of a finished analysis (setwise comparison across arms)."""
    return result.constant_folds.lattice.snapshot()
