"""Structured draft resolution: derived values with WHY, keyed errors, diff-vs-defaults."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from mimarsinan.config_schema.deployment_derivation import (
    enforce_quantization_assembly_contract,
    legal_values_for,
    legality_bearing_keys,
)
from mimarsinan.config_schema.registry import (
    Category,
    REGISTRY,
    effective_value,
    parse_deployment_document,
    retired_key_errors,
    retired_scope_of,
)
from mimarsinan.config_schema.registry.build import split_domain_dormant
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.config_schema.validation import (
    legality_errors,
    validate_against_registry,
    validate_deployment_config,
)

_SESSION_DEFAULT_PIPELINE_MODE = "phased"


@dataclass(frozen=True)
class Resolution:
    """One draft resolved the way a run would resolve it, with structure kept."""

    resolved: Dict[str, Any]
    derived: Dict[str, Dict[str, Any]]
    errors: List[Dict[str, Any]]
    explicit_keys: List[str]
    unknown_keys: List[str]
    meta_keys: List[str]
    diff_vs_defaults: List[Dict[str, Any]] = field(default_factory=list)
    # Draft keys of the OTHER core-semantics domain: kept in the draft,
    # excluded from resolution and emission, restored on switch-back.
    dormant: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


_KEY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def attach_error_key(message: str) -> Optional[str]:
    """Attach a validation message to the first registry or retired key it
    names (a retired key removed from the registry must still attach, or its
    structural echo could never be superseded by the keyed retired row)."""
    for token in _KEY_TOKEN_RE.findall(message):
        if token in REGISTRY or retired_scope_of(token) is not None:
            return token
    return None


def _error_key_scope(key: Optional[str]) -> Optional[str]:
    """The sub-document scope an error row's key belongs to: the registry's
    section for live keys, the retirement table's scope for removed ones."""
    if key is None:
        return None
    entry = REGISTRY.get(key)
    if entry is not None:
        return entry.section
    return retired_scope_of(key)


def _structural_errors(draft: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"key": attach_error_key(msg), "message": msg, "rule_id": "config_shape"}
        for msg in validate_deployment_config(dict(draft))
    ]


def _contract_errors(
    dp: Mapping[str, Any], pc: Mapping[str, Any], pipeline_mode: Optional[str]
) -> List[Dict[str, Any]]:
    try:
        enforce_quantization_assembly_contract(dp, pc, pipeline_mode=pipeline_mode)
    except ValueError as exc:
        return [{
            "key": "weight_quantization",
            "message": str(exc),
            "rule_id": "quantization_assembly",
            # The rule prescribes its remedies; the frontend only applies them.
            "remedies": [
                {"label": "Declare vanilla (float weights)", "action": "set",
                 "key": "pipeline_mode", "value": "vanilla"},
                {"label": "Drop weight_bits", "action": "clear", "key": "weight_bits"},
            ],
        }]
    return []


def effective_view(draft_dp: Mapping[str, Any]) -> Dict[str, Any]:
    """Schema defaults overlaid with the draft's declarations — ALWAYS computable,
    so legality (like the vehicle rows) survives an erroring draft."""
    cfg = {
        key: entry.default for key, entry in REGISTRY.items() if entry.has_default()
    }
    cfg.update({k: v for k, v in draft_dp.items() if v is not None})
    return cfg


def legal_values_view(cfg: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """The LEGAL VALUE SET of every legality-bearing key that APPLIES to this
    config state.

    |legal| == 1 ⇒ the field is LOCKED (rendered read-only as the derived value);
    |legal| > 1 ⇒ derived default + override, offering ONLY these options; a key
    whose legality does not apply here (``None``) is omitted, so the UI leaves it
    an ordinary widget.
    """
    view: Dict[str, List[Any]] = {}
    for key in legality_bearing_keys():
        legal = legal_values_for(key, cfg)
        if legal is not None:
            view[key] = list(legal)
    return view


def derived_values_view(resolved: Mapping[str, Any]) -> Dict[str, Any]:
    """The CONCRETE prospective value of every SSOT-sourced key for this state.

    This is the wizard's green in-field text: what the deriver produces, never
    prose about where it comes from. ``None`` renders as '—' (uncomputable, or
    a registration the registry cannot see); ``{}`` while the draft errors, so no
    hypothetical value ever renders.
    """
    if not resolved:
        return {}
    return {
        key: effective_value(resolved, key)
        for key, entry in REGISTRY.items()
        if entry.provenance is not None
    }


def derived_view(resolved: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every DERIVED key's value + WHY (+ meta) against a resolved flat config.

    Public so a caller holding an ENRICHED resolved view (e.g. the wizard,
    which folds the model builder's registrations in) re-derives the rows
    against the values the run would actually see.
    """
    derived: Dict[str, Dict[str, Any]] = {}
    for flat_key, entry in REGISTRY.items():
        if entry.category is not Category.DERIVED:
            continue
        value = resolved.get(flat_key)
        row: Dict[str, Any] = {
            "value": value,
            "why": entry.why(dict(resolved)) if entry.why is not None else None,
            "derived_from": list(entry.derived_from),
        }
        if entry.meta is not None:
            row["meta"] = entry.meta(dict(resolved))
        derived[flat_key] = row
    return derived


def _diff_vs_defaults(explicit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    diff: List[Dict[str, Any]] = []
    for flat_key, value in explicit.items():
        entry = REGISTRY[flat_key]
        default = entry.default if entry.has_default() else None
        # An explicit null against no default (or a null default) is "unset",
        # never a differing knob; a real value against no default differs.
        if value is None:
            differs = entry.has_default() and entry.default is not None
        else:
            differs = (not entry.has_default()) or value != entry.default
        diff.append({
            "key": flat_key,
            "group": entry.group,
            "label": entry.label,
            "value": value,
            "default": default,
            "differs": differs,
        })
    return diff


def _strip_dormant_domain(draft: Mapping[str, Any]):
    """Return ``(stripped_draft, dormant_keys)`` — the wizard-authoring view
    with other-domain keys excluded from both sections."""
    out = dict(draft or {})
    dp = dict(out.get("deployment_parameters") or {})
    pc = dict(out.get("platform_constraints") or {})
    core = str(dp.get("core_semantics") or "spiking")
    dp_active, dp_dormant = split_domain_dormant(dp, core)
    pc_active, pc_dormant = split_domain_dormant(pc, core)
    out["deployment_parameters"] = dp_active
    out["platform_constraints"] = pc_active
    return out, sorted(dp_dormant) + sorted(pc_dormant)


def resolve_draft(draft: Mapping[str, Any]) -> Resolution:
    """Resolve a deployment-config draft exactly as a run would, structured.

    Returns the merged flat config, every DERIVED key with its WHY, errors
    attached to their keys, and the explicit-key diff against schema defaults.
    Never raises for config mistakes — they come back as ``errors`` rows.
    """
    # Dormant-domain keys never resolve and never error: the semantics switch
    # is zero-friction from any green state (the draft retains them).
    draft, dormant_keys = _strip_dormant_domain(draft)
    parsed = parse_deployment_document(draft)
    errors = _structural_errors(draft)

    declared_mode = draft.get(
        "pipeline_mode", (draft.get("deployment_parameters") or {}).get("pipeline_mode")
    )
    errors.extend(_contract_errors(parsed.dp, parsed.pc, declared_mode))

    # An illegal declaration would raise inside the derivation; report it as a
    # keyed, REMEDIABLE row and never derive a hypothetical config from it. The
    # remediable row supersedes the plain structural message for the same key.
    illegal = legality_errors(effective_view(parsed.dp), parsed.dp)
    illegal_keys = {row["key"] for row in illegal}
    errors = [error for error in errors if error["key"] not in illegal_keys]
    errors.extend(illegal)

    # Field-domain errors (type/bounds/options) are keyed, remediable rows too;
    # they supersede the plain structural message for the same key, exactly like
    # the legality rows. Legality-bearing keys are skipped inside the validator,
    # so these key sets never overlap.
    field_errors = validate_against_registry(parsed.known_flat_keys())
    field_keys = {row["key"] for row in field_errors}
    errors = [error for error in errors if error["key"] not in field_keys]
    errors.extend(field_errors)

    # Retired keys come back as keyed rows with one-click MIGRATION remedies,
    # each carrying its SCOPE, superseding every other row for the same
    # (scope, key). Registry-bridged legacy keys still preview through the
    # meaning-preserving bridge; fully-removed keys never reach the derivation.
    retired = retired_key_errors(
        parsed.retirement_scan_view(), paths=parsed.retired_paths
    )
    retired_pairs = {(row["scope"], row["key"]) for row in retired}
    errors = [
        error for error in errors
        if (_error_key_scope(error["key"]), error["key"]) not in retired_pairs
    ]
    errors.extend(retired)

    pipeline_mode = str(draft.get("pipeline_mode", _SESSION_DEFAULT_PIPELINE_MODE))
    resolved: Dict[str, Any] = {}
    derived: Dict[str, Dict[str, Any]] = {}
    if not illegal:
        try:
            resolved = build_flat_pipeline_config(
                dict(parsed.dp), dict(parsed.pc), pipeline_mode=pipeline_mode
            )
        except ValueError as exc:
            resolved = {}
            errors.append({
                "key": attach_error_key(str(exc)),
                "message": str(exc),
                "rule_id": "derivation",
            })
        else:
            for key, value in parsed.top.items():
                resolved.setdefault(key, value)
            derived = derived_view(resolved)

    explicit = parsed.known_flat_keys()
    return Resolution(
        resolved=resolved,
        derived=derived,
        errors=errors,
        explicit_keys=sorted(explicit),
        unknown_keys=list(parsed.unknown),
        meta_keys=sorted(parsed.meta),
        diff_vs_defaults=_diff_vs_defaults(explicit),
        dormant=dormant_keys,
    )
