"""Validation for deployment config JSON (main.py input) and merged flat config."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.chip_simulation.activation_semantics import RETIRED_SPIKING_KEYS
from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import floorplan_config_errors
from mimarsinan.chip_simulation.spiking_semantics import require_known_spiking_mode
from mimarsinan.config_schema.deployment_derivation import (
    legal_value_error, legal_values_for, legality_bearing_keys,
)
from mimarsinan.config_schema.registry import (
    Category, REGISTRY,
    mvm_document_errors, parse_deployment_document, retired_key_errors,
)
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers
from mimarsinan.config_schema.registry.entries_write import platform_bias_errors
from mimarsinan.config_schema.registry.parse import read_document_like_consumers
from mimarsinan.config_schema.registry.value_schema import entry_value_errors
from mimarsinan.mapping.platform.coalescing import coalescing_config_errors
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_MODES,
    resolve_s_allocation_mode,
)


def s_allocation_config_errors(
    deployment_parameters: Mapping[str, Any],
    platform_constraints: Mapping[str, Any],
) -> List[str]:
    """Validate the per-layer-S temporal-allocation declaration (EW2).

    ``s_allocation`` must be uniform/explicit/budget, read as its consumer
    reads it (``resolve_s_allocation_mode``); only ``uniform`` is wired, so the
    reserved ``explicit``/``budget`` modes are loud-rejected.
    """
    dp = deployment_parameters if isinstance(deployment_parameters, Mapping) else {}
    try:
        mode = resolve_s_allocation_mode(dp)
    except ValueError:
        return [f"s_allocation must be one of {list(S_ALLOCATION_MODES)}, got {dp.get('s_allocation')!r}"]
    # The registry's value schema names explicit and budget reserved (no run threads them); the run
    # door judges the same schema (platform_constraints_resolver.judged_passthrough).
    return entry_value_errors(REGISTRY["s_allocation"], mode)


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


def _field_domain_error(flat_key: str, message: str) -> Dict[str, Any]:
    """A keyed, remediable field-domain row (same shape as the legality errors):
    the universal remedy is to clear the value and accept the derived/default."""
    label = REGISTRY[flat_key].label
    return {
        "key": flat_key,
        "message": message,
        "rule_id": "field_domain",
        "remedies": [{
            "label": f"Clear {label} (remove the invalid value)",
            "action": "clear", "key": flat_key,
        }],
    }


def validate_against_registry(flat: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Judge every present, registry-known key against its DECLARED value schema
    (``registry/value_schema.py``: scalar type, inclusive or exclusive bounds, enum
    options, and the nested fields of JSON and core-type values). Keyed, remediable
    rows in the same error channel as the legal-value-set law — one generic loop,
    no per-key ladder.

    Legality-bearing keys are owned wholesale by ``legality_errors`` (their
    admissible set is STATE-dependent), so they are skipped here; unknown keys are
    reported by the document parser, not duplicated; an absent/``None`` value is
    'unset' and never judged.
    """
    legality_owned = set(legality_bearing_keys())
    errors: List[Dict[str, Any]] = []
    for flat_key, value in flat.items():
        if flat_key not in REGISTRY or value is None or flat_key in legality_owned:
            continue
        messages = entry_value_errors(REGISTRY[flat_key], value)
        if messages:
            errors.append(_field_domain_error(flat_key, "; ".join(messages)))
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
        if flat_key in RETIRED_SPIKING_KEYS:
            continue  # the retired-key rule owns these with migration remedies
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


def document_run_layout(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The flat layout a run gives a document's two sections (``merge_pipeline_config``: the deployment
    parameters, then the platform constraints), where a ``platform_constraints`` object nested in either
    section, an unknown key to the parser, is the nested bias declaration the run door reads. The run's
    default layers beneath hold neither bias form; a parity test against ``merge_pipeline_config`` pins it."""
    sections = (config.get("deployment_parameters"), config.get("platform_constraints"))
    return {k: v for section in sections if isinstance(section, Mapping) for k, v in section.items()}


def validate_deployment_config(config: Dict[str, Any]) -> List[str]:
    """Validate the deployment config JSON shape main.py expects; return error messages (empty if valid).

    The document is read as its consumers read it before anything judges it
    (``read_document_like_consumers``), as the run door reads its flat config.
    """
    errors: List[str] = []

    if not isinstance(config, dict):
        errors.append("Config must be a dict")
        return errors

    config = read_document_like_consumers(config)
    errors.extend(non_declarable_key_errors(config))
    errors.extend(row["message"] for row in retired_key_errors(config))
    errors.extend(mvm_document_errors(config))

    pc = config.get("platform_constraints")
    if isinstance(pc, dict):
        errors.extend(coalescing_config_errors(pc))
        errors.extend(floorplan_config_errors(pc))

    dp_for_axis = config.get("deployment_parameters")
    errors.extend(
        s_allocation_config_errors(
            dp_for_axis if isinstance(dp_for_axis, Mapping) else {},
            pc if isinstance(pc, Mapping) else {},
        )
    )

    flat_view = parse_deployment_document(config).known_flat_keys()
    errors.extend(row["message"] for row in validate_against_registry(flat_view))
    # The run door's bias rule over the run layout of the two sections: each declaration by the BOOL schema,
    # then their agreement (a refusal the registry loop above already states is not repeated).
    stated = set(errors)
    bias_errors = platform_bias_errors(document_run_layout(config), REGISTRY["has_bias"])
    errors.extend(message for message in bias_errors if message not in stated)

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

        # Legality lambdas are total over raw documents; retired keys report above.
        errors.extend(row["message"] for row in legality_errors(dp, dp))

    return errors


def validate_merged_config(flat: Dict[str, Any]) -> List[str]:
    """Validate the merged flat config (runtime pipeline.config); return error messages (empty if valid)."""
    errors: List[str] = []

    if not isinstance(flat, dict):
        errors.append("Merged config must be a dict")
        return errors

    flat = read_like_consumers(flat)
    spiking = flat.get("spiking_mode", "lif")
    try:
        require_known_spiking_mode(spiking)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        errors.extend(row["message"] for row in legality_errors(flat, flat))

    errors.extend(row["message"] for row in validate_against_registry(flat))
    errors.extend(floorplan_config_errors(flat))

    return errors
