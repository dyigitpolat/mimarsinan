"""[W6b] The masked / closure / final arm runs that one pruning pass shares.

Two consumers need the SAME three propagation arms over the SAME seeded,
exemption-filtered graph: the elimination ledger (per-kill attribution) and
the softcore-elimination report (the paper's headline per-crossbar weight-cell
metric). Running the arms once here and handing the results to both keeps a
deployment paying for ONE analysis instead of two.

``results_by_arm`` exposes only the arms that were actually RUN — under
``mode="masked"`` there is no closure result to report, and echoing the masked
sets under a "closure" label would fabricate a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Mapping, Sequence, Set, Tuple

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.pruning.boundary_policy import (
    assert_unified_ir_for_pruning,
)
from mimarsinan.mapping.pruning.elimination_ledger.ledger_types import (
    EliminationLedgerError,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    DEFAULT_ELIMINATION_PROPAGATION,
    ELIMINATION_PROPAGATION_CASCADE,
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_MASKED,
    require_elimination_propagation,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_graph_index,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import GraphIndex
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
    DEFAULT_ELIMINATION_CONSTANT_FOLDING,
)

SeedMasks = Dict[int, Tuple[Sequence[bool], Sequence[bool]]]
SeedSets = Dict[int, Tuple[Set[int], Set[int]]]


@dataclass(frozen=True)
class EliminationArms:
    """The propagation arms of one pruning pass, plus the inputs they shared.

    ``final`` is the arm the deployment actually applies (``mode``); ``masked``
    and ``closure`` are the weaker arms the attribution and the per-arm table
    difference against. The seed/exemption maps are carried so a consumer
    (the ledger's depth replay) can rebuild the identical context.
    """

    mode: str
    masked: GlobalPruningResult
    closure: GlobalPruningResult
    final: GlobalPruningResult
    seed_per_node: SeedSets
    seed_per_bank: SeedSets
    exempt_rows: Mapping[int, frozenset]
    exempt_cols: Mapping[int, frozenset]
    zero_threshold: float
    computeop_liveness_transfers: str
    elimination_constant_folding: str
    spiking_mode: str
    probe_memo: dict = field(default_factory=dict)
    graph_index: "GraphIndex | None" = None

    def results_by_arm(self) -> Dict[str, GlobalPruningResult]:
        """Arm name -> kill sets, for the arms this pass actually ran."""
        by_arm: Dict[str, GlobalPruningResult] = {
            ELIMINATION_PROPAGATION_MASKED: self.masked,
        }
        if self.mode != ELIMINATION_PROPAGATION_MASKED:
            by_arm[ELIMINATION_PROPAGATION_CLOSURE] = self.closure
        if self.mode == ELIMINATION_PROPAGATION_CASCADE:
            by_arm[ELIMINATION_PROPAGATION_CASCADE] = self.final
        return by_arm


def compute_elimination_arms(
    ir_graph: IRGraph,
    *,
    zero_threshold: float = 1e-8,
    initial_pruned_per_node: SeedMasks | None = None,
    initial_pruned_per_bank: SeedMasks | None = None,
    elimination_propagation: str = DEFAULT_ELIMINATION_PROPAGATION,
    computeop_liveness_transfers: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    elimination_constant_folding: str = DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    spiking_mode: str = "lif",
) -> EliminationArms:
    """Run masked / closure / requested arm over one seeded graph; never mutates."""
    mode = require_elimination_propagation(elimination_propagation)
    assert_unified_ir_for_pruning(ir_graph)

    exempt_rows, exempt_cols = _boundary_policy_exemptions(ir_graph)
    seed_per_node, seed_per_bank = _collect_initial_seeds(
        ir_graph, initial_pruned_per_node, initial_pruned_per_bank
    )
    probe_memo: dict = {}   # one run, one battery per (op, bitwise key)
    graph_index = build_graph_index(
        ir_graph, computeop_liveness_transfers=computeop_liveness_transfers,
    )

    def _run_arm(arm: str) -> GlobalPruningResult:
        t0 = time.perf_counter()
        result = compute_global_pruned_sets(
            ir_graph,
            zero_threshold=zero_threshold,
            initial_per_node=seed_per_node,
            initial_per_bank=seed_per_bank,
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
            mode=arm,
            computeop_liveness_transfers=computeop_liveness_transfers,
            elimination_constant_folding=elimination_constant_folding,
            spiking_mode=spiking_mode,
            probe_memo=probe_memo,
            graph_index=graph_index,
        )
        print(f"[EliminationLedger] arm={arm} wall={time.perf_counter() - t0:.1f}s",
              flush=True)
        return result

    masked = _run_arm(ELIMINATION_PROPAGATION_MASKED)
    if mode == ELIMINATION_PROPAGATION_MASKED:
        closure = masked
        final = masked
    elif mode == ELIMINATION_PROPAGATION_CLOSURE:
        closure = _run_arm(ELIMINATION_PROPAGATION_CLOSURE)
        final = closure
    else:
        closure = _run_arm(ELIMINATION_PROPAGATION_CLOSURE)
        final = _run_arm(ELIMINATION_PROPAGATION_CASCADE)
    assert_arm_ordering(masked, closure, final)

    return EliminationArms(
        mode=mode,
        masked=masked,
        closure=closure,
        final=final,
        seed_per_node=seed_per_node,
        seed_per_bank=seed_per_bank,
        exempt_rows=exempt_rows,
        exempt_cols=exempt_cols,
        zero_threshold=zero_threshold,
        computeop_liveness_transfers=computeop_liveness_transfers,
        elimination_constant_folding=elimination_constant_folding,
        spiking_mode=spiking_mode,
        probe_memo=probe_memo,
        graph_index=graph_index,
    )


def assert_arm_ordering(
    masked: GlobalPruningResult,
    closure: GlobalPruningResult,
    final: GlobalPruningResult,
) -> None:
    """kills(masked) <= kills(closure) <= kills(final) — fail loud otherwise."""
    for lo, hi, pair in (
        (masked, closure, "masked<=closure"),
        (closure, final, "closure<=final"),
    ):
        for attr in (
            "pruned_rows_per_node", "pruned_cols_per_node",
            "pruned_rows_per_bank", "pruned_cols_per_bank",
        ):
            lo_map, hi_map = getattr(lo, attr), getattr(hi, attr)
            for key, lo_set in lo_map.items():
                if not lo_set <= hi_map.get(key, set()):
                    raise EliminationLedgerError(
                        f"arm ordering violated ({pair}) on {attr}[{key}]: "
                        f"{sorted(lo_set - hi_map.get(key, set()))} killed by "
                        "the weaker arm only."
                    )
