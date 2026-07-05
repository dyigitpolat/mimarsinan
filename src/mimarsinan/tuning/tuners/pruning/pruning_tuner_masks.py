"""Monotonic pruning mask computation and recovery hooks."""

from __future__ import annotations

import math

import torch

from mimarsinan.mapping.pruning.boundary_policy import (
    build_boundary_ir_graph,
    compute_perceptron_io_exemption_indices,
)
from mimarsinan.transformations.pruning import collect_activation_stats
from mimarsinan.tuning.orchestration.tuner_base import _RECOVERY_PATIENCE


def _boundary_exemption_layers(tuner):
    """Per-tuner cache of (exempt_input_layers, exempt_output_layers); topology-invariant."""
    cached = getattr(tuner, "_boundary_exemption_cache", None)
    if cached is None:
        ir_graph = build_boundary_ir_graph(tuner.model, tuner.pipeline)
        cached = compute_perceptron_io_exemption_indices(
            ir_graph, tuner.model.get_perceptrons()
        )
        tuner._boundary_exemption_cache = cached
    return cached


def _invalidate_boundary_cache(tuner) -> None:
    if hasattr(tuner, "_boundary_exemption_cache"):
        tuner._boundary_exemption_cache = None


def get_masks(tuner, rate, *, commit=True):
    """Monotone masks for ``rate``; ``commit=False`` computes them without
    growing the tuner's persistent pruned sets (probe-replica reads)."""
    perceptrons = tuner.model.get_perceptrons()
    n_layers = len(perceptrons)
    exempt_input_layers, exempt_output_layers = _boundary_exemption_layers(tuner)

    persistent_rows = tuner._persistent_pruned_rows
    persistent_cols = tuner._persistent_pruned_cols
    if len(persistent_rows) != n_layers:
        persistent_rows = [set() for _ in range(n_layers)]
        persistent_cols = [set() for _ in range(n_layers)]
        if commit:
            tuner._persistent_pruned_rows = persistent_rows
            tuner._persistent_pruned_cols = persistent_cols

    row_masks = []
    col_masks = []
    for i, p in enumerate(perceptrons):
        out_f, in_f = p.layer.weight.data.shape
        device = p.layer.weight.device

        k_r = int(math.floor(rate * tuner.pruning_fraction * out_f))
        if i in exempt_output_layers:
            k_r = 0
        pruned_r = set(persistent_rows[i])
        if len(pruned_r) < k_r and i < len(tuner.base_row_imp):
            _, idx = tuner.base_row_imp[i].to(device).sort()
            for j in idx.tolist():
                if len(pruned_r) >= k_r:
                    break
                pruned_r.add(int(j))
        if commit:
            persistent_rows[i] = pruned_r
        rm = torch.ones(out_f, dtype=torch.bool, device=device)
        if pruned_r:
            rm[list(pruned_r)] = False
        row_masks.append(rm)

        k_c = int(math.floor(rate * tuner.pruning_fraction * in_f))
        if i in exempt_input_layers:
            k_c = 0
        pruned_c = set(persistent_cols[i])
        if len(pruned_c) < k_c and i < len(tuner.base_col_imp):
            _, idx = tuner.base_col_imp[i].to(device).sort()
            for j in idx.tolist():
                if len(pruned_c) >= k_c:
                    break
                pruned_c.add(int(j))
        if commit:
            persistent_cols[i] = pruned_c
        cm = torch.ones(in_f, dtype=torch.bool, device=device)
        if pruned_c:
            cm[list(pruned_c)] = False
        col_masks.append(cm)

    for i in range(n_layers - 1):
        if row_masks[i].shape[0] == col_masks[i + 1].shape[0]:
            col_masks[i + 1] = col_masks[i + 1] & row_masks[i]
    return row_masks, col_masks


def refresh_pruning_importance(tuner):
    perceptrons = tuner.model.get_perceptrons()
    activation_stats = collect_activation_stats(
        tuner.model,
        tuner.trainer.validation_loader,
        tuner._device,
        num_batches=5,
    )
    tuner.base_row_imp.clear()
    tuner.base_col_imp.clear()
    for i, p in enumerate(perceptrons):
        w = p.layer.weight.data
        if activation_stats[i]["output_importance"] is not None:
            tuner.base_row_imp.append(activation_stats[i]["output_importance"].clone())
        else:
            tuner.base_row_imp.append(w.abs().sum(dim=1))
        if activation_stats[i]["input_importance"] is not None:
            tuner.base_col_imp.append(activation_stats[i]["input_importance"].clone())
        else:
            tuner.base_col_imp.append(w.abs().sum(dim=0))


def register_recovery_hooks(tuner, target_row_masks, target_col_masks, rate):
    hooks = []
    for i, p in enumerate(tuner.model.get_perceptrons()):
        rm = target_row_masks[i]
        cm = target_col_masks[i]

        pruned_rows = ~rm
        pruned_cols = ~cm
        prune_mask = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)

        scale = 1.0 - rate
        target_w = tuner.original_weights[i][prune_mask] * scale

        b_mask = None
        target_b = None
        if p.layer.bias is not None:
            b_mask = pruned_rows
            target_b = tuner.original_biases[i][pruned_rows] * scale

        def make_hook(layer, p_mask, t_w, b_m, t_b):
            def hook(module, inputs):
                module.weight.data[p_mask] = t_w
                if b_m is not None and module.bias is not None:
                    module.bias.data[b_m] = t_b
            return hook

        hooks.append(
            p.layer.register_forward_pre_hook(
                make_hook(p.layer, prune_mask, target_w, b_mask, target_b)
            )
        )
    return hooks


def force_to_full_rate(tuner):
    current = tuner._committed_rate
    remaining = 1.0 - current
    n_increments = max(3, min(6, int(remaining / 0.15) + 1))

    for i in range(1, n_increments + 1):
        target = current + remaining * i / n_increments
        target = min(target, 1.0)

        tuner._apply_masks(target)

        hooks = tuner._recovery_training_hooks(target)
        try:
            lr = tuner._find_lr()
            tuner.trainer.train_steps_until_target(
                lr,
                tuner._budget.max_training_steps,
                tuner._get_target(),
                0,
                validation_n_batches=tuner._budget.progress_eval_batches,
                check_interval=tuner._budget.check_interval,
                patience=_RECOVERY_PATIENCE,
                min_steps=tuner._budget.check_interval * 3,
                min_improvement=tuner._budget.accuracy_se() / 2,
            )
        finally:
            for h in hooks:
                h.remove()

    tuner._apply_masks(1.0)
    tuner._committed_rate = 1.0
