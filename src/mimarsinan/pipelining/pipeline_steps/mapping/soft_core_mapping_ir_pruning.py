"""IR pruning helpers for :class:`SoftCoreMappingStep`."""

from __future__ import annotations

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.diagnostics import phase_profiler
from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.mapping.pruning.elimination_ledger import (
    compute_elimination_arms,
    compute_elimination_ledger,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    resolve_elimination_propagation,
)
from mimarsinan.mapping.softcore_elimination import (
    report_from_arms,
    summarize_softcore_elimination,
    write_softcore_elimination_markdown,
    write_softcore_elimination_record,
)
from mimarsinan.mapping.pruning.liveness_transfer import (
    effective_constant_folding,
    resolve_computeop_liveness_transfers,
    resolve_elimination_constant_folding,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.pruning.ir_pruning_masks import get_initial_pruning_masks_from_model
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


def emit_softcore_elimination_record(step, ir_graph, arms):
    """[W6b] Print, reporter-event and serialize the softcore-elimination
    record (JSON + the drop-in markdown table) for this deployment."""
    report = report_from_arms(ir_graph, arms)
    print(f"[SoftCoreMappingStep] {summarize_softcore_elimination(report)}")
    emit_reporter_event(
        step.pipeline.reporter, "softcore_elimination", report.to_dict()
    )
    write_softcore_elimination_record(report, step.pipeline.working_directory)
    write_softcore_elimination_markdown(report, step.pipeline.working_directory)
    return report


def apply_ir_pruning_if_enabled(step, model, ir_graph, phase_tag: str):
    """Compact zeroed rows/columns when pruning was applied."""
    plan = DeploymentPlan.of(step.pipeline)
    if not plan.pruning:
        return ir_graph
    elimination_propagation = resolve_elimination_propagation(
        step.pipeline.config
    )
    computeop_liveness_transfers = resolve_computeop_liveness_transfers(
        step.pipeline.config
    )
    elimination_constant_folding = effective_constant_folding(
        policy=resolve_elimination_constant_folding(step.pipeline.config),
        computeop_liveness_transfers=computeop_liveness_transfers,
    )

    with best_effort("report first-perceptron prune-mask buffers"):
        perceptrons_pre = model.get_perceptrons()
        if perceptrons_pre:
            layer0 = getattr(perceptrons_pre[0], "layer", None)
            has_row = getattr(layer0, "prune_row_mask", None) is not None
            has_col = getattr(layer0, "prune_col_mask", None) is not None
            print(
                f"[SoftCoreMappingStep] Pruning: before mask extraction — first perceptron layer "
                f"prune_row_mask={has_row} prune_col_mask={has_col}"
            )

    initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_graph)
    with best_effort("report pruning-mask diagnostics"):
        perceptrons = model.get_perceptrons()
        neural_cores = ir_graph.get_neural_cores()
        n_banks = len(getattr(ir_graph, "weight_banks", {}))
        print(
            f"[SoftCoreMappingStep] Pruning: perceptrons={len(perceptrons)} neural_cores={len(neural_cores)} "
            f"weight_banks={n_banks} initial_pruned_per_node={len(initial_node or {})} "
            f"initial_pruned_per_bank={len(initial_bank or {})}"
        )
        if len(initial_node or {}) == 0 and len(initial_bank or {}) == 0 and len(neural_cores) != len(perceptrons):
            print(
                "[SoftCoreMappingStep] Pruning: no model masks applied (neural_cores != perceptrons; "
                "ensure mapper assigns perceptron_index for tiled IR)."
            )

    store_heatmap = True
    heatmap_budget_bytes = int(step.pipeline.config.get(
        "pre_pruning_heatmap_budget_bytes", 2 * 1024**3,
    ))
    est_bytes = 0
    for nc in ir_graph.get_neural_cores():
        if nc.core_matrix is not None:
            est_bytes += nc.core_matrix.shape[0] * nc.core_matrix.shape[1] * 4
    for bank in (getattr(ir_graph, "weight_banks", {}) or {}).values():
        est_bytes += int(bank.core_matrix.nbytes)   # one snapshot per bank
    if est_bytes > heatmap_budget_bytes:
        print(
            f"[SoftCoreMappingStep] Pre-pruning heatmap would require "
            f"{est_bytes/1e9:.1f} GB (budget {heatmap_budget_bytes/1e9:.1f} GB); "
            f"disabling heatmap storage for this run. "
            f"Set `pre_pruning_heatmap_budget_bytes` higher to override."
        )
        store_heatmap = False

    with phase_profiler(phase_tag, "elimination_ledger"):
        arms = compute_elimination_arms(
            ir_graph,
            initial_pruned_per_node=initial_node if initial_node else None,
            initial_pruned_per_bank=initial_bank if initial_bank else None,
            elimination_propagation=elimination_propagation,
            computeop_liveness_transfers=computeop_liveness_transfers,
            elimination_constant_folding=elimination_constant_folding,
            spiking_mode=str(plan.spiking_mode),
        )
        # Every analysis input above is already baked into `arms`; the ledger
        # reads them back off it (and would refuse a contradicting duplicate),
        # so only what the arm runs do NOT fix is passed here.
        ledger = compute_elimination_ledger(
            ir_graph,
            elimination_propagation=elimination_propagation,
            simulation_steps=int(step.pipeline.config["simulation_steps"]),
            arms=arms,
        )
    print(f"[SoftCoreMappingStep] {ledger.summary()}")

    # [W6b] The paper's HEADLINE metric, measured here and nowhere else: this
    # is the only seam holding BOTH the mapping (every softcore at its full
    # pre-elimination a x n) and the per-arm kill sets. One step later
    # `prune_ir_graph` compacts owned matrices away and the weaker arms are
    # gone, so the denominator and the C1 columns become unrecoverable.
    emit_softcore_elimination_record(step, ir_graph, arms)

    with phase_profiler(phase_tag, "prune_ir_graph"):
        ir_graph = prune_ir_graph(
            ir_graph,
            initial_pruned_per_node=initial_node if initial_node else None,
            initial_pruned_per_bank=initial_bank if initial_bank else None,
            store_heatmap=store_heatmap,
            simulation_steps=int(step.pipeline.config["simulation_steps"]),
            spiking_mode=str(plan.spiking_mode),
            elimination_propagation=elimination_propagation,
            computeop_liveness_transfers=computeop_liveness_transfers,
            elimination_constant_folding=elimination_constant_folding,
            # [O2] the ledger's cascade arm IS this analysis; reuse it rather
            # than running a second identical fixpoint over the same graph.
            precomputed_result=arms.final,
        )
    print(
        "[SoftCoreMappingStep] Applied IR pruning (zeroed row/col elimination, "
        f"propagation={elimination_propagation}, "
        f"computeop_liveness_transfers={computeop_liveness_transfers}, "
        f"elimination_constant_folding={elimination_constant_folding})"
    )
    return ir_graph
