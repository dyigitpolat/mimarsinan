"""
Config schema and validation for deployment pipeline configuration.

Single source of truth for default deployment parameters, platform constraints,
pipeline-mode presets, and validation of both the JSON shape (main.py input)
and the merged flat config (pipeline.config at runtime).

Pipeline code (DeploymentPipeline) should use these defaults so wizard and
pipeline stay in sync.
"""

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
from mimarsinan.config_schema.validation import (
    validate_deployment_config,
    validate_merged_config,
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
]
