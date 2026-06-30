"""
Validation for deployment config JSON (main.py input) and merged flat config.

Validates:
- Top-level keys and nested deployment_parameters / platform_constraints.
- User mode: model_type and model_config required when model_config_mode is "user".
- TTFS: firing_mode and spike_generation_mode must be "TTFS" when spiking_mode is ttfs/ttfs_quantized.
- Search: arch_search required when any search is active.
- Per-layer-S (EW2): the ``s_allocation`` intent is a recognized mode; only ``uniform``
  is wired, so the reserved ``explicit``/``budget`` modes are loud-rejected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.chip_simulation.spiking_semantics import (
    require_known_spiking_mode,
    requires_ttfs_firing,
)
from mimarsinan.mapping.platform.coalescing import coalescing_config_errors
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_MODES,
    S_ALLOCATION_SUPPORTED_MODES,
    S_ALLOCATION_UNIFORM,
    unsupported_s_allocation_error,
)


def s_allocation_config_errors(
    deployment_parameters: Mapping[str, Any],
    platform_constraints: Mapping[str, Any],
) -> List[str]:
    """Validate the per-layer-S temporal-allocation declaration (EW2).

    Only ``uniform`` is actually wired into the forwards/sim. ``explicit`` and ``budget``
    are RESERVED resolver seams that would silently no-op to uniform (the Q2 foot-gun: a
    user thinks budget/explicit S-allocation works when it does nothing). This loud-rejects
    them at config-validation time, BEFORE the silent-uniform resolver path is reachable:

    * ``s_allocation`` must be one of {uniform | explicit | budget}.
    * only ``uniform`` is supported; ``explicit`` / ``budget`` are rejected as
      reserved/not-implemented (``unsupported_s_allocation_error``).

    ``uniform`` (the default) is always valid and ungated => byte-identical. The
    ``platform_constraints`` argument is retained for call-site stability; the reserved
    modes are rejected outright, so no capability gate is consulted.
    """
    errors: List[str] = []
    dp = deployment_parameters if isinstance(deployment_parameters, Mapping) else {}

    raw_mode = dp.get("s_allocation")
    if raw_mode is None:
        mode = S_ALLOCATION_UNIFORM
    else:
        mode = str(raw_mode).lower()
        if mode not in S_ALLOCATION_MODES:
            errors.append(
                f"s_allocation must be one of {list(S_ALLOCATION_MODES)}, got {raw_mode!r}"
            )
            return errors

    if mode not in S_ALLOCATION_SUPPORTED_MODES:
        errors.append(unsupported_s_allocation_error(mode))

    return errors


def validate_deployment_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate the deployment config JSON shape that main.py expects.

    Returns a list of error messages (empty if valid).
    """
    errors: List[str] = []

    if not isinstance(config, dict):
        errors.append("Config must be a dict")
        return errors

    pc = config.get("platform_constraints")
    if isinstance(pc, dict):
        errors.extend(coalescing_config_errors(pc))

    dp_for_axis = config.get("deployment_parameters")
    errors.extend(
        s_allocation_config_errors(
            dp_for_axis if isinstance(dp_for_axis, Mapping) else {},
            pc if isinstance(pc, Mapping) else {},
        )
    )

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

        # spiking_mode membership (rejects removed/unknown modes, e.g. 'rate')
        spiking = dp.get("spiking_mode", "lif")
        try:
            require_known_spiking_mode(spiking)
        except ValueError as exc:
            errors.append(str(exc))
        # TTFS consistency
        if requires_ttfs_firing(spiking):
            if "firing_mode" in dp and dp.get("firing_mode") != "TTFS":
                errors.append(
                    f"spiking_mode is '{spiking}' but firing_mode must be 'TTFS', got {dp.get('firing_mode')!r}"
                )
            if "spike_generation_mode" in dp and dp.get("spike_generation_mode") != "TTFS":
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

    spiking = flat.get("spiking_mode", "lif")
    try:
        require_known_spiking_mode(spiking)
    except ValueError as exc:
        errors.append(str(exc))
    if requires_ttfs_firing(spiking):
        if flat.get("firing_mode") != "TTFS":
            errors.append(
                f"spiking_mode is '{spiking}' but firing_mode must be 'TTFS', got {flat.get('firing_mode')!r}"
            )
        if flat.get("spike_generation_mode") != "TTFS":
            errors.append(
                f"spiking_mode is '{spiking}' but spike_generation_mode must be 'TTFS', got {flat.get('spike_generation_mode')!r}"
            )

    return errors
