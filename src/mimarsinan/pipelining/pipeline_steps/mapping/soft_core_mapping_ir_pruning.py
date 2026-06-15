"""IR pruning helpers for :class:`SoftCoreMappingStep`."""

from __future__ import annotations

from mimarsinan.common.diagnostics import phase_profiler


def apply_ir_pruning_if_enabled(step, model, ir_graph, phase_tag: str):
    """Compact zeroed rows/columns when pruning was applied."""
    if not step.pipeline.config.get("pruning", False):
        return ir_graph

    from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
    from mimarsinan.mapping.pruning.ir_pruning_masks import get_initial_pruning_masks_from_model

    try:
        perceptrons_pre = model.get_perceptrons()
        if perceptrons_pre:
            layer0 = getattr(perceptrons_pre[0], "layer", None)
            has_row = getattr(layer0, "prune_row_mask", None) is not None
            has_col = getattr(layer0, "prune_col_mask", None) is not None
            print(
                f"[SoftCoreMappingStep] Pruning: before mask extraction — first perceptron layer "
                f"prune_row_mask={has_row} prune_col_mask={has_col}"
            )
    except Exception as e:
        print(f"[SoftCoreMappingStep] Pruning: could not check first perceptron buffers: {e}")

    initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_graph)
    try:
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
    except Exception:
        pass

    # The pre-pruning heatmap is always stored, subject only to a memory-budget
    # guard (a heatmap larger than ``pre_pruning_heatmap_budget_bytes`` is skipped).
    store_heatmap = True
    heatmap_budget_bytes = int(step.pipeline.config.get(
        "pre_pruning_heatmap_budget_bytes", 2 * 1024**3,
    ))
    est_bytes = 0
    for nc in ir_graph.get_neural_cores():
        if nc.core_matrix is not None:
            est_bytes += nc.core_matrix.shape[0] * nc.core_matrix.shape[1] * 4
    if est_bytes > heatmap_budget_bytes:
        print(
            f"[SoftCoreMappingStep] Pre-pruning heatmap would require "
            f"{est_bytes/1e9:.1f} GB (budget {heatmap_budget_bytes/1e9:.1f} GB); "
            f"disabling heatmap storage for this run. "
            f"Set `pre_pruning_heatmap_budget_bytes` higher to override."
        )
        store_heatmap = False

    with phase_profiler(phase_tag, "prune_ir_graph"):
        ir_graph = prune_ir_graph(
            ir_graph,
            initial_pruned_per_node=initial_node if initial_node else None,
            initial_pruned_per_bank=initial_bank if initial_bank else None,
            store_heatmap=store_heatmap,
            simulation_steps=int(step.pipeline.config["simulation_steps"]),
            spiking_mode=str(step.pipeline.config.get("spiking_mode", "lif")),
        )
    print("[SoftCoreMappingStep] Applied IR pruning (zeroed row/col elimination)")
    return ir_graph
