"""Pruning utilities: mask computation and weight pruning application."""

from mimarsinan.transformations.pruning.activation import (
    collect_activation_stats,
    compute_pruning_masks_from_activations,
)
from mimarsinan.transformations.pruning.masks import (
    compute_all_pruning_masks,
    compute_masks_from_importance,
    compute_pruning_masks,
)
from mimarsinan.transformations.pruning.apply import apply_pruning_masks

_collect_activation_stats = collect_activation_stats

__all__ = [
    "apply_pruning_masks",
    "compute_pruning_masks",
    "compute_all_pruning_masks",
    "compute_masks_from_importance",
    "collect_activation_stats",
    "compute_pruning_masks_from_activations",
    "_collect_activation_stats",
]
