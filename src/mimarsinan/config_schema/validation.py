"""Validation for deployment config JSON (main.py input) and merged flat config."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.chip_simulation.spiking_semantics import require_known_spiking_mode
from mimarsinan.config_schema.deployment_derivation import (
    legal_value_error,
    legal_values_for,
    legality_bearing_keys,
)
from mimarsinan.config_schema.registry import Category, REGISTRY
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

    ``s_allocation`` must be uniform/explicit/budget; only ``uniform`` is
    wired, so the reserved ``explicit``/``budget`` modes are loud-rejected.
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


def _legality_remedies(flat_key: str, legal: List[Any]) -> List[Dict[str, Any]]:
    """One-click remedies prescribed by the RULE, not by the frontend: a
    singleton legal set can be applied directly; clearing always works."""
    label = REGISTRY[flat_key].label
    remedies: List[Dict[str, Any]] = []
    if len(legal) == 1:
        remedies.append({
            "label": f"Set {label} = {legal[0]}", "action": "set",
            "key": flat_key, "value": legal[0],
        })
    remedies.append({
        "label": f"Clear {label} (accept the derived value)",
        "action": "clear", "key": flat_key,
    })
    return remedies


def legality_errors(
    cfg: Mapping[str, Any], declared: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """THE legal-value-set check: keyed, remediable rows for DECLARED values
    outside their legal set under ``cfg``.

    An absent key is legal by construction (the derivation supplies a legal
    value), so only declarations are judged — and each row names the offending
    key, never the mode that constrains it. Generic over the registry: no
    per-mode ladder lives here or anywhere downstream.
    """
    errors: List[Dict[str, Any]] = []
    for flat_key in legality_bearing_keys():
        value = declared.get(flat_key)
        if flat_key not in declared or value is None:
            continue
        legal_tuple = legal_values_for(flat_key, cfg)
        if legal_tuple is None:  # legality does not apply here — do not judge
            continue
        legal = list(legal_tuple)
        if value in legal:
            continue
        errors.append({
            "key": flat_key,
            "message": str(legal_value_error(flat_key, value, legal)),
            "rule_id": "legal_value_set",
            "remedies": _legality_remedies(flat_key, legal),
        })
    return errors


def non_declarable_key_errors(config: Mapping[str, Any]) -> List[str]:
    """Reject document declarations of keys the derivation/runtime owns.

    A RUNTIME key or a non-declarable DERIVED key in a config file would
    shadow its owner (or silently be overwritten) — unexposed knobs must not
    be settable programmatically either.
    """
    containers = {
        "top": config,
        "deployment_parameters": config.get("deployment_parameters"),
        "platform_constraints": config.get("platform_constraints"),
    }
    errors: List[str] = []
    for flat_key, entry in REGISTRY.items():
        if entry.category is Category.RUNTIME:
            owner_kind = "the runtime"
        elif entry.category is Category.DERIVED and not entry.declarable:
            owner_kind = "the derivation"
        else:
            continue
        container = containers.get(entry.section)
        if isinstance(container, Mapping) and flat_key in container:
            errors.append(
                f"{flat_key} is not declarable in a config document — "
                f"{owner_kind} ({entry.owner}) owns it. Remove the key."
            )
    return errors


def validate_deployment_config(config: Dict[str, Any]) -> List[str]:
    """Validate the deployment config JSON shape main.py expects; return error messages (empty if valid)."""
    errors: List[str] = []

    if not isinstance(config, dict):
        errors.append("Config must be a dict")
        return errors

    errors.extend(non_declarable_key_errors(config))

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
            if "model_type" not in dp:
                errors.append(
                    "Search is active but model_type is missing (the builder "
                    "family whose config space the search explores)"
                )

        spiking = dp.get("spiking_mode", "lif")
        try:
            require_known_spiking_mode(spiking)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            errors.extend(row["message"] for row in legality_errors(dp, dp))

    return errors


def validate_merged_config(flat: Dict[str, Any]) -> List[str]:
    """Validate the merged flat config (runtime pipeline.config); return error messages (empty if valid)."""
    errors: List[str] = []

    if not isinstance(flat, dict):
        errors.append("Merged config must be a dict")
        return errors

    spiking = flat.get("spiking_mode", "lif")
    try:
        require_known_spiking_mode(spiking)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        errors.extend(row["message"] for row in legality_errors(flat, flat))

    return errors
