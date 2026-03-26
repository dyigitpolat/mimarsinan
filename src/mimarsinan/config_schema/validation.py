"""
Validation for deployment config JSON (main.py input) and merged flat config.

Validates:
- Top-level keys and nested deployment_parameters / platform_constraints.
- User mode: model_type and model_config required when model_config_mode is "user".
- TTFS: firing_mode and spike_generation_mode must be "TTFS" when spiking_mode is ttfs/ttfs_quantized.
- Search: arch_search required when any search is active.
"""

from __future__ import annotations

from typing import Any, Dict, List


def validate_deployment_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate the deployment config JSON shape that main.py expects.

    Returns a list of error messages (empty if valid).
    """
    errors: List[str] = []

    if not isinstance(config, dict):
        errors.append("Config must be a dict")
        return errors

    # Top-level required
    for key in ("data_provider_name", "experiment_name", "generated_files_path", "platform_constraints", "deployment_parameters", "start_step"):
        if key not in config:
            errors.append(f"Missing top-level key: {key}")

    dp = config.get("deployment_parameters")
    if not isinstance(dp, dict):
        if "deployment_parameters" in config:
            errors.append("deployment_parameters must be a dict")
    else:
        model_mode = dp.get("model_config_mode", "user")
        hw_mode = dp.get("hw_config_mode", "fixed")
        any_search = model_mode == "search" or hw_mode == "search"

        if model_mode == "user":
            if "model_type" not in dp:
                errors.append("model_config_mode is 'user' but model_type is missing")
            if "model_config" not in dp:
                errors.append("model_config_mode is 'user' but model_config is missing")

        if any_search:
            if "arch_search" not in dp or not isinstance(dp.get("arch_search"), dict):
                errors.append("Search is active but arch_search is missing or not a dict")

        # TTFS consistency
        spiking = dp.get("spiking_mode", "rate")
        if spiking in ("ttfs", "ttfs_quantized"):
            if dp.get("firing_mode") != "TTFS":
                errors.append(
                    f"spiking_mode is '{spiking}' but firing_mode must be 'TTFS', got {dp.get('firing_mode')!r}"
                )
            if dp.get("spike_generation_mode") != "TTFS":
                errors.append(
                    f"spiking_mode is '{spiking}' but spike_generation_mode must be 'TTFS', got {dp.get('spike_generation_mode')!r}"
                )

    return errors


def validate_merged_config(flat: Dict[str, Any]) -> List[str]:
    """
    Validate the merged flat config (runtime pipeline.config).

    Returns a list of error messages (empty if valid).
    """
    errors: List[str] = []

    if not isinstance(flat, dict):
        errors.append("Merged config must be a dict")
        return errors

    spiking = flat.get("spiking_mode", "rate")
    if spiking in ("ttfs", "ttfs_quantized"):
        if flat.get("firing_mode") != "TTFS":
            errors.append(
                f"spiking_mode is '{spiking}' but firing_mode must be 'TTFS', got {flat.get('firing_mode')!r}"
            )
        if flat.get("spike_generation_mode") != "TTFS":
            errors.append(
                f"spiking_mode is '{spiking}' but spike_generation_mode must be 'TTFS', got {flat.get('spike_generation_mode')!r}"
            )

    return errors
