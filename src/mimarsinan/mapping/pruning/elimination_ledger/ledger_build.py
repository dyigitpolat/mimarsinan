"""Build the per-run elimination ledger: attribution + depth + liveness accounting."""

from __future__ import annotations

from typing import Dict, Sequence, Set, Tuple

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.boundary_policy import (
    assert_unified_ir_for_pruning,
)
from mimarsinan.mapping.pruning.elimination_ledger.depth_replay import (
    DepthReplay,
    replay_kill_depths,
)
from mimarsinan.mapping.pruning.elimination_ledger.ledger_types import (
    BankEliminationRecord,
    EliminationCounts,
    EliminationLedger,
    EliminationLedgerError,
    NodeEliminationRecord,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    DEFAULT_ELIMINATION_PROPAGATION,
    ELIMINATION_PROPAGATION_CASCADE,
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_MASKED,
    require_elimination_propagation,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
    compute_global_pruned_sets,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_global_pruning_context,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
    GlobalPruningResult,
)
from mimarsinan.mapping.pruning.ir_liveness import NodeLiveness, compute_liveness
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
    _collect_initial_seeds,
)

SeedMasks = Dict[int, Tuple[Sequence[bool], Sequence[bool]]]


def compute_elimination_ledger(
    ir_graph: IRGraph,
    *,
    zero_threshold: float = 1e-8,
    initial_pruned_per_node: SeedMasks | None = None,
    initial_pruned_per_bank: SeedMasks | None = None,
    elimination_propagation: str = DEFAULT_ELIMINATION_PROPAGATION,
    spiking_mode: str = "lif",
    simulation_steps: int = 32,
) -> EliminationLedger:
    """Attribute every kill of one (uncompacted) graph + seed set + mode.

    Runs the masked and closure arms alongside the requested arm, so each
    kill is attributed by set difference (SEED = masked; CLOSURE-COUPLING =
    closure - masked; EMERGENT-PROPAGATION = final - closure), replays the
    causal waves for per-kill depth, and folds in the liveness pass (DEAD
    core deletions, BIAS_ONLY collapses). Never mutates ``ir_graph``.
    """
    mode = require_elimination_propagation(elimination_propagation)
    if not ir_graph.nodes:
        return EliminationLedger(
            mode=mode, per_node=(), per_bank=(), cores_deleted=0,
            bias_only_collapses=0, fixpoint_iterations=0,
        )
    assert_unified_ir_for_pruning(ir_graph)

    exempt_rows, exempt_cols = _boundary_policy_exemptions(ir_graph)
    seed_per_node, seed_per_bank = _collect_initial_seeds(
        ir_graph, initial_pruned_per_node, initial_pruned_per_bank
    )
    def _run_arm(arm: str) -> GlobalPruningResult:
        return compute_global_pruned_sets(
            ir_graph,
            zero_threshold=zero_threshold,
            initial_per_node=seed_per_node,
            initial_per_bank=seed_per_bank,
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
            mode=arm,
        )

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
    _assert_arm_ordering(masked, closure, final)

    replay_ctx = build_global_pruning_context(
        ir_graph,
        zero_threshold=zero_threshold,
        initial_per_node=seed_per_node,
        initial_per_bank=seed_per_bank,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
    )
    replay = replay_kill_depths(replay_ctx, mode=mode)
    _assert_replay_reconciles(replay_ctx, final)

    liveness = compute_liveness(
        ir_graph,
        simulation_steps=simulation_steps,
        spiking_mode=spiking_mode,
        pruning_result=final,
        zero_threshold=zero_threshold,
    )
    dead_ids = {
        nid for nid, status in liveness.per_node.items()
        if status == NodeLiveness.DEAD
    }
    bias_only = sum(
        1 for status in liveness.per_node.values()
        if status == NodeLiveness.BIAS_ONLY
    )

    per_node = tuple(
        _node_record(node, ir_graph, masked, closure, final, replay, dead_ids)
        for node in ir_graph.nodes
        if isinstance(node, NeuralCore)
    )
    per_bank = tuple(
        _bank_record(bank_id, bank, masked, closure, final)
        for bank_id, bank in (getattr(ir_graph, "weight_banks", {}) or {}).items()
    )
    return EliminationLedger(
        mode=mode,
        per_node=per_node,
        per_bank=per_bank,
        cores_deleted=len(dead_ids),
        bias_only_collapses=bias_only,
        fixpoint_iterations=final.fixpoint_iterations,
    )


def _split_counts(
    masked: Set[int], closure: Set[int], final: Set[int], liveness_extra: Set[int]
) -> Tuple[int, int, int, int]:
    seed = final & masked
    coupled = (final & closure) - masked
    emergent = final - closure
    return len(seed), len(coupled), len(emergent), len(liveness_extra)


def _node_record(
    node: NeuralCore,
    graph: IRGraph,
    masked: GlobalPruningResult,
    closure: GlobalPruningResult,
    final: GlobalPruningResult,
    replay: DepthReplay,
    dead_ids: Set[int],
) -> NodeEliminationRecord:
    mat = node.get_core_matrix(graph)
    n_axons, n_neurons = mat.shape
    nid = node.id
    final_rows = final.pruned_rows_per_node.get(nid, set())
    final_cols = final.pruned_cols_per_node.get(nid, set())
    # A DEAD node is force-pruned whole before removal; structure beyond the
    # fixpoint kill set is attributed to the liveness pass.
    liveness_rows = (
        set(range(n_axons)) - final_rows if nid in dead_ids else set()
    )
    liveness_cols = (
        set(range(n_neurons)) - final_cols if nid in dead_ids else set()
    )
    s_r, c_r, e_r, l_r = _split_counts(
        masked.pruned_rows_per_node.get(nid, set()),
        closure.pruned_rows_per_node.get(nid, set()),
        final_rows, liveness_rows,
    )
    s_c, c_c, e_c, l_c = _split_counts(
        masked.pruned_cols_per_node.get(nid, set()),
        closure.pruned_cols_per_node.get(nid, set()),
        final_cols, liveness_cols,
    )
    return NodeEliminationRecord(
        node_id=nid,
        name=str(getattr(node, "name", nid)),
        n_axons=n_axons,
        n_neurons=n_neurons,
        counts=EliminationCounts(
            seed_rows=s_r, closure_rows=c_r, emergent_rows=e_r,
            liveness_rows=l_r, seed_cols=s_c, closure_cols=c_c,
            emergent_cols=e_c, liveness_cols=l_c,
        ),
        row_depths=dict(replay.row_depths.get(nid, {})),
        col_depths=dict(replay.col_depths.get(nid, {})),
    )


def _bank_record(
    bank_id: int,
    bank,
    masked: GlobalPruningResult,
    closure: GlobalPruningResult,
    final: GlobalPruningResult,
) -> BankEliminationRecord:
    n_axons, n_neurons = bank.core_matrix.shape
    s_r, c_r, e_r, _ = _split_counts(
        masked.pruned_rows_per_bank.get(bank_id, set()),
        closure.pruned_rows_per_bank.get(bank_id, set()),
        final.pruned_rows_per_bank.get(bank_id, set()), set(),
    )
    s_c, c_c, e_c, _ = _split_counts(
        masked.pruned_cols_per_bank.get(bank_id, set()),
        closure.pruned_cols_per_bank.get(bank_id, set()),
        final.pruned_cols_per_bank.get(bank_id, set()), set(),
    )
    return BankEliminationRecord(
        bank_id=bank_id,
        n_axons=n_axons,
        n_neurons=n_neurons,
        counts=EliminationCounts(
            seed_rows=s_r, closure_rows=c_r, emergent_rows=e_r,
            seed_cols=s_c, closure_cols=c_c, emergent_cols=e_c,
        ),
    )


def _assert_arm_ordering(
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


def _assert_replay_reconciles(
    ctx, final: GlobalPruningResult
) -> None:
    """The depth replay must land on exactly the production kill sets."""
    pairs = (
        (ctx.pruned_rows, final.pruned_rows_per_node, "rows"),
        (ctx.pruned_cols, final.pruned_cols_per_node, "cols"),
        (ctx.bank_pruned_rows, final.pruned_rows_per_bank, "bank rows"),
        (ctx.bank_pruned_cols, final.pruned_cols_per_bank, "bank cols"),
    )
    for replayed, produced, label in pairs:
        if replayed != produced:
            raise EliminationLedgerError(
                f"depth replay diverged from the production fixpoint on "
                f"{label}: replay={replayed} production={produced}. The "
                "ledger cannot attribute kills it cannot reproduce."
            )
