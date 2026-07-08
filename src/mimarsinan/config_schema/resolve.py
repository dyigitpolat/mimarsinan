"""Structured draft resolution: derived values with WHY, keyed errors, diff-vs-defaults."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from mimarsinan.config_schema.deployment_derivation import (
    enforce_quantization_assembly_contract,
)
from mimarsinan.config_schema.registry import (
    Category,
    REGISTRY,
    parse_deployment_document,
)
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.config_schema.validation import validate_deployment_config

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

    @property
    def ok(self) -> bool:
        return not self.errors


_KEY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _attach_error_key(message: str) -> Optional[str]:
    """Attach a validation message to the first registry key it names."""
    for token in _KEY_TOKEN_RE.findall(message):
        if token in REGISTRY:
            return token
    return None


def _structural_errors(draft: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"key": _attach_error_key(msg), "message": msg, "rule_id": "config_shape"}
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
        }]
    return []


def _derived_view(resolved: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
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


def resolve_draft(draft: Mapping[str, Any]) -> Resolution:
    """Resolve a deployment-config draft exactly as a run would, structured.

    Returns the merged flat config, every DERIVED key with its WHY, errors
    attached to their keys, and the explicit-key diff against schema defaults.
    Never raises for config mistakes — they come back as ``errors`` rows.
    """
    parsed = parse_deployment_document(draft)
    errors = _structural_errors(draft)

    declared_mode = draft.get(
        "pipeline_mode", (draft.get("deployment_parameters") or {}).get("pipeline_mode")
    )
    errors.extend(_contract_errors(parsed.dp, parsed.pc, declared_mode))

    pipeline_mode = str(draft.get("pipeline_mode", _SESSION_DEFAULT_PIPELINE_MODE))
    resolved: Dict[str, Any] = {}
    derived: Dict[str, Dict[str, Any]] = {}
    try:
        resolved = build_flat_pipeline_config(
            dict(parsed.dp), dict(parsed.pc), pipeline_mode=pipeline_mode
        )
    except ValueError as exc:
        errors.append({
            "key": _attach_error_key(str(exc)),
            "message": str(exc),
            "rule_id": "derivation",
        })
    else:
        for key, value in parsed.top.items():
            resolved.setdefault(key, value)
        derived = _derived_view(resolved)

    explicit = parsed.known_flat_keys()
    return Resolution(
        resolved=resolved,
        derived=derived,
        errors=errors,
        explicit_keys=sorted(explicit),
        unknown_keys=list(parsed.unknown),
        meta_keys=sorted(parsed.meta),
        diff_vs_defaults=_diff_vs_defaults(explicit),
    )
