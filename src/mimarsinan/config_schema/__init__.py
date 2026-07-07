"""Config schema SSOT: default deployment parameters, presets, and validation for both JSON input and the merged runtime config."""

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
    DEFAULT_TRAINING_RECIPE,
    DEFAULT_TUNING_RECIPE,
    PIPELINE_MODE_PRESETS,
    CONFIG_KEYS_SET,
    get_default_deployment_parameters,
    get_default_platform_constraints,
    get_default_training_recipe,
    get_default_tuning_recipe,
    get_pipeline_mode_presets,
    get_config_keys_set,
    apply_preset,
)
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.config_schema.display_view import build_config_display_view, build_pipeline_config_view
from mimarsinan.config_schema.validation import (
    validate_deployment_config,
    validate_merged_config,
    s_allocation_config_errors,
)
from mimarsinan.config_schema.namespaced_schema import (
    CONCERN_GROUPS,
    KEY_SPECS,
    KeySpec,
    LEGACY_KEY_TABLE,
    keys_with_exposure,
    keys_with_derivation,
    provenance_table,
    to_flat,
    to_namespaced,
)

__all__ = [
    "DEFAULT_DEPLOYMENT_PARAMETERS",
    "DEFAULT_PLATFORM_CONSTRAINTS",
    "DEFAULT_TRAINING_RECIPE",
    "DEFAULT_TUNING_RECIPE",
    "PIPELINE_MODE_PRESETS",
    "CONFIG_KEYS_SET",
    "get_default_deployment_parameters",
    "get_default_platform_constraints",
    "get_default_training_recipe",
    "get_default_tuning_recipe",
    "get_pipeline_mode_presets",
    "get_config_keys_set",
    "apply_preset",
    "validate_deployment_config",
    "validate_merged_config",
    "s_allocation_config_errors",
    "build_flat_pipeline_config",
    "build_config_display_view",
    "build_pipeline_config_view",
    "CONCERN_GROUPS",
    "KEY_SPECS",
    "KeySpec",
    "LEGACY_KEY_TABLE",
    "keys_with_exposure",
    "keys_with_derivation",
    "provenance_table",
    "to_flat",
    "to_namespaced",
]
