"""Build the per-run elimination ledger: attribution + depth + liveness accounting."""

from __future__ import annotations

import time
from typing import AbstractSet, Dict, Sequence, Set, Tuple

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.elimination_ledger.arm_runs import (
    EliminationArms,
    compute_elimination_arms,
)
from mimarsinan.mapping.pruning.elimination_ledger.depth_replay import (
    DepthReplay,
    replay_kill_depths,
)
from mimarsinan.mapping.pruning.elimination_ledger.ledger_inputs import (
    UNSET,
    Unset,
    resolve_arm_inputs,
)
from mimarsinan.mapping.pruning.elimination_ledger.ledger_types import (
    BankEliminationRecord,
    EliminationCounts,
    EliminationLedger,
    EliminationLedgerError,
    NodeEliminationRecord,
    count_grid_certifiable_folds,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    DEFAULT_ELIMINATION_PROPAGATION,
    require_elimination_propagation,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    build_global_pruning_context,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
    GlobalPruningResult,
)
from mimarsinan.mapping.pruning.ir_liveness import NodeLiveness, compute_liveness

SeedMasks = Dict[int, Tuple[Sequence[bool], Sequence[bool]]]


def compute_elimination_ledger(
    ir_graph: IRGraph,
    *,
    zero_threshold: float | Unset = UNSET,
    initial_pruned_per_node: SeedMasks | None = None,
    initial_pruned_per_bank: SeedMasks | None = None,
    elimination_propagation: str = DEFAULT_ELIMINATION_PROPAGATION,
    computeop_liveness_transfers: str | Unset = UNSET,
    elimination_constant_folding: str | Unset = UNSET,
    spiking_mode: str | Unset = UNSET,
    simulation_steps: int = 32,
    arms: EliminationArms | None = None,
) -> EliminationLedger:
    """Attribute every kill of one (uncompacted) graph + seed set + mode.

    Runs the masked and closure arms alongside the requested arm, so each
    kill is attributed by set difference (CONSTANT-FOLD carved out first, then
    SEED = masked; CLOSURE-COUPLING = closure - masked; EMERGENT-PROPAGATION =
    final - closure), replays the causal waves for per-kill depth, and folds
    in the liveness pass (DEAD core deletions, BIAS_ONLY collapses). Never
    mutates ``ir_graph``.

    ``arms`` accepts an already-computed :class:`EliminationArms` so a caller
    that also builds the softcore-elimination report pays for the arm runs
    once. It CARRIES the analysis inputs it was run with, and this function
    then consumes those — including for the liveness pass, which used to read
    the ledger's own (defaulted, therefore possibly wrong) ``zero_threshold``
    and ``spiking_mode``. Supplying an arm input that CONTRADICTS the arms
    fails loud rather than being silently dropped; the seed maps cannot be
    compared meaningfully once seeded, so passing them with ``arms`` at all is
    an error.
    """
    mode = require_elimination_propagation(elimination_propagation)
    resolved = resolve_arm_inputs(
        {
            "zero_threshold": zero_threshold,
            "computeop_liveness_transfers": computeop_liveness_transfers,
            "elimination_constant_folding": elimination_constant_folding,
            "spiking_mode": spiking_mode,
        },
        {
            "initial_pruned_per_node": initial_pruned_per_node,
            "initial_pruned_per_bank": initial_pruned_per_bank,
        },
        arms,
        mode,
    )
    if not ir_graph.nodes:
        return EliminationLedger(
            mode=mode, per_node=(), per_bank=(), cores_deleted=0,
            bias_only_collapses=0, fixpoint_iterations=0,
        )
    if arms is None:
        arms = compute_elimination_arms(
            ir_graph,
            zero_threshold=resolved.zero_threshold,
            initial_pruned_per_node=initial_pruned_per_node,
            initial_pruned_per_bank=initial_pruned_per_bank,
            elimination_propagation=mode,
            computeop_liveness_transfers=resolved.computeop_liveness_transfers,
            elimination_constant_folding=resolved.elimination_constant_folding,
            spiking_mode=resolved.spiking_mode,
        )
    masked, closure, final = arms.masked, arms.closure, arms.final

    replay_ctx = build_global_pruning_context(
        ir_graph,
        zero_threshold=arms.zero_threshold,
        initial_per_node=arms.seed_per_node,
        initial_per_bank=arms.seed_per_bank,
        exempt_rows_per_node=arms.exempt_rows,
        exempt_cols_per_node=arms.exempt_cols,
        computeop_liveness_transfers=arms.computeop_liveness_transfers,
        elimination_constant_folding=arms.elimination_constant_folding,
        spiking_mode=arms.spiking_mode,
    )
    _t0 = time.perf_counter()
    replay = replay_kill_depths(replay_ctx, mode=mode)
    print(f"[EliminationLedger] replay wall={time.perf_counter() - _t0:.1f}s "
          f"waves={replay.waves}", flush=True)
    _assert_replay_reconciles(replay_ctx, final)

    # The liveness pass must read the SAME analysis inputs the arms ran with;
    # before W6c it read this function's own defaults, so a precomputed-arms
    # caller silently got zero_threshold=1e-8 / spiking_mode="lif" here.
    liveness = compute_liveness(
        ir_graph,
        simulation_steps=simulation_steps,
        spiking_mode=resolved.spiking_mode,
        pruning_result=final,
        zero_threshold=resolved.zero_threshold,
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
    masked: AbstractSet[int],
    closure: AbstractSet[int],
    final: AbstractSet[int],
    liveness_extra: AbstractSet[int],
    constant: AbstractSet[int] = frozenset(),
) -> Tuple[int, int, int, int, int]:
    """Partition one kill set into the attribution categories; CONSTANT-FOLD
    kills are carved out FIRST, else a folded row would land in the closure or
    emergent bucket depending on the wave that resolved it."""
    folded = final & constant
    rest = final - folded
    seed = rest & masked
    coupled = (rest & closure) - masked
    emergent = rest - closure
    return (
        len(seed), len(coupled), len(emergent), len(liveness_extra), len(folded)
    )


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
    fold_values = final.constant_folds.folded_rows.get(nid, {})
    folded = frozenset(fold_values)
    s_r, c_r, e_r, l_r, k_r = _split_counts(
        masked.pruned_rows_per_node.get(nid, set()),
        closure.pruned_rows_per_node.get(nid, set()),
        final_rows, liveness_rows, folded,
    )
    on_grid = count_grid_certifiable_folds(
        fold_values[row] for row in sorted(final_rows & folded)
    )
    s_c, c_c, e_c, l_c, _ = _split_counts(
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
            liveness_rows=l_r, constant_rows=k_r,
            constant_rows_grid_certifiable=on_grid, seed_cols=s_c,
            closure_cols=c_c, emergent_cols=e_c, liveness_cols=l_c,
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
    s_r, c_r, e_r, _, _ = _split_counts(
        masked.pruned_rows_per_bank.get(bank_id, set()),
        closure.pruned_rows_per_bank.get(bank_id, set()),
        final.pruned_rows_per_bank.get(bank_id, set()), set(),
    )
    s_c, c_c, e_c, _, _ = _split_counts(
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
