"""Module-level forward hooks for persistent pruning enforcement."""

from mimarsinan.transformations.pruning.committed_masks import (
    commit_layer_pruning,
    commit_norm_pruning,
)


def pruning_enforce_linear_pre_hook(module, inputs):
    commit_layer_pruning(module)


def pruning_enforce_norm_pre_hook(module, inputs):
    commit_norm_pruning(module)
