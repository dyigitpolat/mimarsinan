"""Explicit-keys-only deployment-config emission (the one builder for Deploy/templates)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from mimarsinan.config_schema.registry import REGISTRY, Category, parse_deployment_document
from mimarsinan.config_schema.registry.build import split_domain_dormant

# Canonical top-level order (matches templates/generate.py output style).
_CANONICAL_TOP_ORDER = (
    "seed",
    "pipeline_mode",
    "experiment_name",
    "generated_files_path",
    "datasets_path",
    "data_provider_name",
    "platform_constraints",
    "deployment_parameters",
    "target_metric_override",
    "start_step",
    "stop_step",
)

# Top-level keys every emitted config carries (filled from schema defaults
# when the draft omits them); start_step is emitted even when null so resume
# tooling always finds the key.
_REQUIRED_TOP_KEYS = (
    "data_provider_name", "experiment_name", "generated_files_path", "seed",
    "start_step",
)


def _droppable(flat_key: str) -> bool:
    """RUNTIME keys and non-declarable DERIVED keys never belong in a document."""
    entry = REGISTRY.get(flat_key)
    if entry is None:
        return False
    if entry.category is Category.RUNTIME:
        return True
    return entry.category is Category.DERIVED and not entry.declarable


def _emit_section(raw: Mapping[str, Any], core_semantics: str = "spiking") -> Dict[str, Any]:
    """Verbatim pass-through in draft order; runtime/non-declarable-derived
    keys are dropped, and DORMANT-domain keys (the other core-semantics
    domain) are stripped — an emitted document never contains a key its own
    core_semantics forbids. Unknown keys are PRESERVED (never silently lost).
    """
    active, _dormant = split_domain_dormant(dict(raw), core_semantics)
    return {k: v for k, v in active.items() if not _droppable(k)}


def _top_default(flat_key: str) -> Any:
    entry = REGISTRY.get(flat_key)
    if entry is not None and entry.has_default():
        return entry.default
    return None


def emit_deployment_config(draft: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the deployment JSON from a draft document, explicit keys only.

    The inverse of loading: every key the draft declares survives verbatim
    (unknown keys included), in canonical top-level order. Only runtime keys
    and non-declarable derived keys (``activation_quantization``) are removed
    — the derivation owns them and a pinned copy would shadow it.
    """
    draft = dict(draft or {})

    top: Dict[str, Any] = {
        k: v for k, v in draft.items()
        if k not in ("deployment_parameters", "platform_constraints")
        and not k.startswith("_") and not _droppable(k)
    }
    for key in _REQUIRED_TOP_KEYS:
        top.setdefault(key, _top_default(key))

    dp_raw = dict(draft.get("deployment_parameters") or {})
    core_semantics = str(dp_raw.get("core_semantics") or "spiking")
    out: Dict[str, Any] = {}
    for key in _CANONICAL_TOP_ORDER:
        if key == "platform_constraints":
            out[key] = _emit_section(
                dict(draft.get("platform_constraints") or {}), core_semantics,
            )
        elif key == "deployment_parameters":
            out[key] = _emit_section(dp_raw, core_semantics)
        elif key in top:
            out[key] = top.pop(key)
    out.update(top)

    for key, value in draft.items():
        if key.startswith("_") and value:
            out[key] = value
    return out


def emit_with_unknown_report(draft: Mapping[str, Any]) -> Tuple[Dict[str, Any], list]:
    """Emit plus the dotted paths of schema-unknown keys (the loud tray)."""
    return emit_deployment_config(draft), parse_deployment_document(draft).unknown
