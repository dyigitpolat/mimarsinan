"""Validation for deployment config JSON (main.py input) and merged flat config."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from mimarsinan.chip_simulation.activation_semantics import RETIRED_SPIKING_KEYS
from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import floorplan_config_errors
from mimarsinan.chip_simulation.spiking_semantics import require_known_spiking_mode
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.deployment_derivation import legal_value_error
from mimarsinan.config_schema.registry import REGISTRY, parse_deployment_document, retired_key_errors
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers
from mimarsinan.config_schema.registry.declaration_judge import (
    applied_view_errors, declared_legality_rows, document_declaration_errors, legality_rows,
)
from mimarsinan.config_schema.registry.declaration_places import not_declarable_message, run_layers
from mimarsinan.config_schema.registry.parse import read_document_like_consumers
from mimarsinan.config_schema.runtime import run_derivation_errors, unset
from mimarsinan.config_schema.registry.value_schema import entry_value_errors
from mimarsinan.mapping.platform.coalescing import coalescing_config_errors
# Bound as a module (resolved from sys.modules while it initializes): target_expansion imports the registry,
# whose package imports this module first when target_expansion is the first import.
import mimarsinan.mapping.platform.target_expansion as target_expansion
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
    per-mode ladder lives here or anywhere downstream. The rows are the one
    judge's (``declaration_judge.legality_rows``): a value outside its own
    schema is the schema's to name, so the set judges only the rest.
    """
    return _legality_rows(legality_rows(cfg, declared))


def declared_legality_errors(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """``legality_errors`` of a document as a run judges it: its two sections' run layout and own declarations
    under the run's state (``declaration_judge.declared_legality_rows``), the rows the document door states."""
    top = {k: v for k, v in document.items() if k in REGISTRY and REGISTRY[k].section == "top"}
    return _legality_rows(declared_legality_rows(
        document.get("deployment_parameters"), document.get("platform_constraints"), top=top))


def _legality_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    return [{
        "key": flat_key,
        "message": str(legal_value_error(flat_key, value, legal)),
        "rule_id": "legal_value_set",
        "remedies": _legality_remedies(flat_key, list(legal)),
    } for flat_key, value, legal in rows]


def field_domain_error(flat_key: str, message: str) -> Dict[str, Any]:
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

    A legality-bearing key is judged here too, as the one judge judges it: its
    schema names a value outside it, and ``legality_errors`` judges only a value
    its schema admits (one message per violation); unknown keys are reported by
    the document parser, not duplicated; an absent/``None`` value is 'unset' and
    never judged.
    """
    errors: List[Dict[str, Any]] = []
    for flat_key, value in flat.items():
        if flat_key not in REGISTRY or value is None:
            continue
        messages = entry_value_errors(REGISTRY[flat_key], value)
        if messages:
            errors.append(field_domain_error(flat_key, "; ".join(messages)))
    return errors


def non_declarable_key_errors(config: Mapping[str, Any]) -> List[str]:
    """Reject document declarations of keys the derivation/runtime owns.

    A RUNTIME key or a non-declarable DERIVED key in a config document would
    shadow its owner (or silently be overwritten) — unexposed knobs must not
    be settable programmatically either. A key is judged in its own place and,
    for a section key, in either section, where the run's layout reads it; a
    null declaration is unset, as both doors read it.
    """
    places = [config.get(section) for section in ("deployment_parameters", "platform_constraints")]
    errors: List[str] = []
    for flat_key, entry in REGISTRY.items():
        if flat_key in RETIRED_SPIKING_KEYS or (message := not_declarable_message(flat_key)) is None:
            continue  # the retired-key rule owns these with migration remedies
        containers = [config] if entry.section == "top" else places
        if any(isinstance(container, Mapping) and container.get(flat_key) is not None for container in containers):
            errors.append(message)
    return errors


#: The top-level keys a run reads with no value of its own: a document leaves each one missing when it is absent,
#: and, but for ``start_step`` (null runs from the first step) and the sections (the shape rule names a null one),
#: when it is unset (null, or an empty path: ``runtime.unset``).
REQUIRED_TOP_LEVEL = ("data_provider_name", "experiment_name", "generated_files_path", "platform_constraints",
                      "deployment_parameters", "start_step")
_NULL_ALLOWED = ("platform_constraints", "deployment_parameters", "start_step")


def missing_top_level_errors(config: Mapping[str, Any], keys: Tuple[str, ...] = REQUIRED_TOP_LEVEL) -> List[str]:
    """Each of ``keys`` a document leaves missing, by name: absent, or null where a null is no value; the document
    door states them, and the CLI path refuses those the session reads."""
    return [f"Missing top-level key: {key}" for key in keys
            if key not in config or (unset(key, config[key]) and key not in _NULL_ALLOWED)]


def section_shape_errors(config: Mapping[str, Any]) -> List[str]:
    """A declared section that is no object: the document door and the CLI path refuse it by name."""
    return [f"{section} must be a dict" for section in ("platform_constraints", "deployment_parameters")
            if section in config and not isinstance(config[section], dict)]


def document_run_layout(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The flat layout a run gives a document's two sections (``merge_pipeline_config``: the deployment
    parameters, then the platform constraints), where a ``platform_constraints`` object nested in either
    section, an unknown key to the parser, is the nested bias declaration the run door reads. The run's
    default layers beneath hold neither bias form; a parity test against ``merge_pipeline_config`` pins it."""
    sections = (config.get("deployment_parameters"), config.get("platform_constraints"))
    return {k: v for section in sections if isinstance(section, Mapping) for k, v in section.items()}


def validate_deployment_config(config: Dict[str, Any], *, builder_registration: bool = True) -> List[str]:
    """Validate the deployment config JSON shape main.py expects; return error messages (empty if valid).

    The document is read as its consumers read it before anything judges it
    (``read_document_like_consumers``), as the run door reads its flat config.
    Every declaration is judged where, and as, a run judges it by the one judge
    (``document_declaration_errors``, G4 interface note 3), beside the parser's
    registry loop, which also judges the wizard's structural platform bodies,
    by their value schemas and the schemas their consumers apply
    (``applied_view_errors``); each message is stated once. A document the judge admits is derived as a
    run derives it, under the model builder's registration unless
    ``builder_registration`` is False (``run_derivation_errors``).
    """
    errors: List[str] = []

    if not isinstance(config, dict):
        errors.append("Config must be a dict")
        return errors

    config = read_document_like_consumers(config)
    errors.extend(non_declarable_key_errors(config))
    # A declared target expands before anything else judges the document, so the registry and domain rules
    # judge the bundle's keys; a refused expansion is stated below, once.
    try:
        config, target_refusals = dict(target_expansion.expand_target_document(config, write_derived=False)), []
    except ToolchainRefusal as refusal:
        target_refusals = [str(refusal)]
    errors.extend(row["message"] for row in retired_key_errors(config))

    pc = config.get("platform_constraints")
    if isinstance(pc, dict):
        errors.extend(coalescing_config_errors(pc))

    parsed = parse_deployment_document(config)
    errors.extend(row["message"] for row in validate_against_registry(parsed.known_flat_keys()))
    if isinstance(pc, dict) and pc.get("mode") is not None:  # the wizard's structural platform body, as parsed
        errors.extend(applied_view_errors(parsed.dp, parsed.pc, top=parsed.top))
    # The one judge both doors call: the run's layout of the two sections (a key declared in the other section
    # included) and each section's own declarations as written, by their value schemas, the domain rule
    # (``mvm_document_errors``), the legal-value-set law, the schemas consumers apply and the bias rule; then a
    # refused target expansion.
    judged = [*document_declaration_errors(config), *target_refusals]
    # A document the judge admits is derived as a run derives it, and the run's own refusal is stated
    # (``run_derivation_errors``: the quantization assembly contract and the derivation's pairings).
    errors.extend(judged or run_derivation_errors(config, builder_registration=builder_registration))

    errors.extend(missing_top_level_errors(config))
    errors.extend(section_shape_errors(config))
    if isinstance(config.get("deployment_parameters"), dict):
        errors.extend(presence_errors(run_layers(config["deployment_parameters"], config.get("platform_constraints"),
                                                 defaults=True)))

    return list(dict.fromkeys(errors))


def presence_errors(layout: Mapping[str, Any]) -> List[str]:
    """The model and search keys the chosen modes need, read where the run reads them (the run's layout of the two
    sections over their default layers)."""
    errors: List[str] = []
    model_mode, hw_mode = layout.get("model_config_mode", "user"), layout.get("hw_config_mode", "fixed")
    if model_mode == "user":
        errors += [f"model_config_mode is 'user' but {key} is missing" for key in ("model_type", "model_config")
                   if key not in layout]
    if model_mode == "search" or hw_mode == "search":
        if not isinstance(layout.get("arch_search"), dict):
            errors.append("Search is active but arch_search is missing or not a dict")
        if "model_type" not in layout:
            errors.append("Search is active but model_type is missing (the builder "
                          "family whose config space the search explores)")
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
