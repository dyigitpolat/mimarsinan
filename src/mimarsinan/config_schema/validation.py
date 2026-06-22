"""
Validation for deployment config JSON (main.py input) and merged flat config.

Validates:
- Top-level keys and nested deployment_parameters / platform_constraints.
- User mode: model_type and model_config required when model_config_mode is "user".
- TTFS: firing_mode and spike_generation_mode must be "TTFS" when spiking_mode is ttfs/ttfs_quantized.
- Search: arch_search required when any search is active.
- Per-layer-S (EW2): the ``s_allocation`` intent is well-formed and capability-gated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing
from mimarsinan.mapping.platform.coalescing import coalescing_config_errors
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_BUDGET,
    S_ALLOCATION_EXPLICIT,
    S_ALLOCATION_MODES,
    S_ALLOCATION_UNIFORM,
)

# The budget body must declare at least one of these objectives (the rest are
# optional). Mirrors temporal_allocation._VALID_BUDGET_KEYS; kept local so the
# wizard-level "needs at least one objective" rule does not couple to the resolver's
# private set.
_BUDGET_OBJECTIVE_KEYS = ("max_energy_proxy", "max_latency_steps", "target")


def s_allocation_config_errors(
    deployment_parameters: Mapping[str, Any],
    platform_constraints: Mapping[str, Any],
) -> List[str]:
    """Validate the per-layer-S temporal-allocation declaration (EW2).

    The Wizard DECLARES the intent (``s_allocation`` + the reserved per-mode inputs)
    and the chip DECLARES the capability (``allow_per_layer_s``). This checks the two
    agree BEFORE the resolver runs, so a malformed/ungated declaration fails loud at
    wizard-submit time rather than mid-pipeline:

    * ``s_allocation`` must be one of {uniform | explicit | budget}.
    * ``explicit`` requires ``s_allocation_explicit`` = a non-empty list of positive ints.
    * ``budget`` requires ``s_allocation_budget`` with at least one of
      {max_energy_proxy, max_latency_steps, target}.
    * ``explicit`` / ``budget`` (the non-uniform modes) require the
      ``allow_per_layer_s`` chip capability (a capability gate, like allow_coalescing).

    ``uniform`` (the default) is always valid and ungated => byte-identical.
    """
    errors: List[str] = []
    dp = deployment_parameters if isinstance(deployment_parameters, Mapping) else {}
    pc = platform_constraints if isinstance(platform_constraints, Mapping) else {}

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

    if mode == S_ALLOCATION_UNIFORM:
        return errors

    capability = bool(pc.get("allow_per_layer_s", False))
    if not capability:
        errors.append(
            f"s_allocation={mode!r} requires the allow_per_layer_s chip capability "
            "(set platform_constraints.allow_per_layer_s = true)"
        )

    if mode == S_ALLOCATION_EXPLICIT:
        errors.extend(_explicit_errors(dp.get("s_allocation_explicit")))
    elif mode == S_ALLOCATION_BUDGET:
        errors.extend(_budget_errors(dp.get("s_allocation_budget")))

    return errors


def _flat_platform_constraints(pc: Any) -> Mapping[str, Any]:
    """Project a platform_constraints body to the flat capability dict.

    Handles the wizard's wrapped shapes (``{mode: 'user', user: {...}}`` /
    ``{mode: 'auto', auto: {fixed: {...}}}``) and a bare flat dict, so the
    capability gate (``allow_per_layer_s``) is read from wherever the user put it."""
    if not isinstance(pc, Mapping):
        return {}
    mode = pc.get("mode")
    if mode == "user":
        user = pc.get("user")
        return user if isinstance(user, Mapping) else {}
    if mode == "auto":
        auto = pc.get("auto")
        if isinstance(auto, Mapping):
            fixed = auto.get("fixed")
            return fixed if isinstance(fixed, Mapping) else {}
        return {}
    return pc


def _explicit_errors(raw: Any) -> List[str]:
    msg = (
        "s_allocation='explicit' requires s_allocation_explicit "
        "(a non-empty list of positive ints)"
    )
    if raw is None:
        return [msg]
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, (list, tuple)):
        return [msg]
    if len(raw) == 0:
        return [msg]
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            return [msg]
    return []


def _budget_errors(raw: Any) -> List[str]:
    if not isinstance(raw, Mapping):
        return [
            "s_allocation='budget' requires s_allocation_budget (a dict with at least "
            f"one of {list(_BUDGET_OBJECTIVE_KEYS)})"
        ]
    if not any(raw.get(k) is not None for k in _BUDGET_OBJECTIVE_KEYS):
        return [
            "s_allocation_budget must declare at least one of "
            f"{list(_BUDGET_OBJECTIVE_KEYS)}"
        ]
    return []


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
            _flat_platform_constraints(pc),
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

        # TTFS consistency
        spiking = dp.get("spiking_mode", "lif")
        if requires_ttfs_firing(spiking):
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

    spiking = flat.get("spiking_mode", "lif")
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
