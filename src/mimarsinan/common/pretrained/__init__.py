"""Pretrained-weight-set registration contract (model-builder declared)."""

from mimarsinan.common.pretrained.weight_sets import (
    WEIGHT_SET_KEY,
    WEIGHT_SETS_KEY,
    PretrainedWeightSet,
    applicable_weight_sets,
    derived_weight_set_id,
    legal_preload_values,
    legal_weight_set_ids,
    preload_regime_error,
    preload_unavailable_reason,
    registered_weight_sets,
    select_weight_set,
    selected_source,
    weight_set_mismatch,
)

__all__ = [
    "WEIGHT_SET_KEY",
    "WEIGHT_SETS_KEY",
    "PretrainedWeightSet",
    "applicable_weight_sets",
    "derived_weight_set_id",
    "legal_preload_values",
    "legal_weight_set_ids",
    "preload_regime_error",
    "preload_unavailable_reason",
    "registered_weight_sets",
    "select_weight_set",
    "selected_source",
    "weight_set_mismatch",
]
