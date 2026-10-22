"""Structured draft resolution: derived values with WHY, keyed errors, diff-vs-defaults."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.deployment_derivation import legal_values_for, legality_bearing_keys
from mimarsinan.config_schema.registry import (
    Category,
    REGISTRY,
    effective_value,
    parse_deployment_document,
    retired_key_errors,
    retired_scope_of,
)
from mimarsinan.config_schema.registry.build import split_domain_dormant
from mimarsinan.config_schema.registry.value_schema import entry_value_errors
from mimarsinan.config_schema.runtime import (
    enforce_run_contract, model_registration, registration_updates, run_config, run_sections,
)
from mimarsinan.config_schema.registry.declaration_judge import document_declaration_errors, sections_run_layout
from mimarsinan.config_schema.validation import (
    declared_legality_errors, field_domain_error, legality_errors, section_shape_errors, validate_against_registry,
    validate_deployment_config,
)
# Bound as a module (resolved from sys.modules while it initializes): target_expansion imports the registry,
# whose package imports this module first when target_expansion is the first import.
import mimarsinan.mapping.platform.target_expansion as target_expansion


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


def _panel_messages(dp: Mapping[str, Any]) -> Set[str]:
    """The rows the wizard's pretrained panel states itself, keyed with their remedies: the legal-value-set rows the
    builder registration adds to the deployment parameters' declarations ``dp``, judged as the panel judges them
    (the effective view with the registration beneath it). Every other placement's row is the wizard's own."""
    declared = {key: value for key, value in dp.items()  # the declarations their schemas admit (the run's state)
                if value is not None and not (key in REGISTRY and entry_value_errors(REGISTRY[key], value))}
    view = effective_view(declared)
    enriched = {**registration_updates(model_registration(view), None), **view}
    return ({row["message"] for row in legality_errors(enriched, declared)}
            - {row["message"] for row in legality_errors(view, declared)})


def _structural_errors(draft: Mapping[str, Any], dp: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The document door's rows, the builder registration's included (wherever the run reads the declaration),
    but those the pretrained panel states (``_panel_messages``)."""
    panel = _panel_messages(dp)
    return [{"key": attach_error_key(msg), "message": msg, "rule_id": "config_shape"}
            for msg in validate_deployment_config(dict(draft)) if msg not in panel]


def _run_sections(draft: Mapping[str, Any], parsed: Any) -> Dict[str, Any]:
    """The draft as a run takes it for its flat config (``run_sections``): each section whole, as declared, so every
    container the run reads is previewed (the nested bias object of ``PLATFORM_BIAS_PATHS`` among them; review
    round 3), the parser's flat body standing in only for a structural platform shape (``mode`` user or auto) the
    wizard nests. The dormant-domain keys the wizard keeps out of both sections are stripped before
    (``_strip_dormant_domain``)."""
    sections = {name: dict(draft[name])
                for name in ("deployment_parameters", "platform_constraints") if isinstance(draft.get(name), Mapping)}
    if sections.get("platform_constraints") is not None and draft["platform_constraints"].get("mode") is not None:
        sections["platform_constraints"] = dict(parsed.pc)
    return {**draft, "deployment_parameters": {}, **sections}


def _contract_errors(draft: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The quantization assembly contract over the run's layout of the draft (``run_sections``), as the CLI path
    applies it; a draft whose sections the run cannot lay states it elsewhere."""
    try:
        if isinstance(draft.get("deployment_parameters"), Mapping):
            enforce_run_contract(*run_sections(draft)[:2])
    except ValueError as exc:
        # The rule prescribes its remedies; the frontend only applies them.
        return [{"key": "weight_quantization", "message": str(exc), "rule_id": "quantization_assembly",
                 "remedies": [{"label": "Declare vanilla (float weights)", "action": "set",
                               "key": "pipeline_mode", "value": "vanilla"},
                              {"label": "Drop weight_bits", "action": "clear", "key": "weight_bits"}]}]
    return []


def effective_view(draft_dp: Mapping[str, Any]) -> Dict[str, Any]:
    """Schema defaults overlaid with the draft's declarations — ALWAYS computable,
    so legality (like the vehicle rows) survives an erroring draft."""
    return {**{key: entry.default for key, entry in REGISTRY.items() if entry.has_default()},
            **{k: v for k, v in draft_dp.items() if v is not None}}


def legal_values_view(cfg: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """The LEGAL VALUE SET of every legality-bearing key that APPLIES to this
    config state.

    |legal| == 1 ⇒ the field is LOCKED (rendered read-only as the derived value);
    |legal| > 1 ⇒ derived default + override, offering ONLY these options; a key
    whose legality does not apply here (``None``) is omitted, so the UI leaves it
    an ordinary widget.
    """
    view = {key: legal_values_for(key, cfg) for key in legality_bearing_keys()}
    return {key: list(legal) for key, legal in view.items() if legal is not None}


def derived_values_view(resolved: Mapping[str, Any]) -> Dict[str, Any]:
    """The CONCRETE prospective value of every SSOT-sourced key for this state.

    This is the wizard's green in-field text: what the deriver produces, never
    prose about where it comes from. ``None`` renders as '—' (uncomputable, or
    a registration the registry cannot see); ``{}`` while the draft errors, so no
    hypothetical value ever renders.
    """
    return {key: effective_value(resolved, key) for key, entry in REGISTRY.items()
            if entry.provenance is not None} if resolved else {}


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
    out, dormant = dict(draft or {}), []
    layout = sections_run_layout(out.get("deployment_parameters"), out.get("platform_constraints"))
    core = str(layout.get("core_semantics") or "spiking")  # the core semantics the run's layout reads
    for name in ("deployment_parameters", "platform_constraints"):
        section = out.get(name)  # an absent section, or one that is no object, is the doors' to refuse, as declared
        if isinstance(section, Mapping):
            out[name], stripped = split_domain_dormant(dict(section), core)
            dormant += sorted(stripped)
    return out, dormant


def resolve_draft(draft: Mapping[str, Any]) -> Resolution:
    """Resolve a deployment-config draft exactly as a run would, structured.

    Returns the merged flat config, every DERIVED key with its WHY, errors
    attached to their keys, and the explicit-key diff against schema defaults.
    Never raises for config mistakes — they come back as ``errors`` rows.
    """
    # A declared target expands first, so dormant-domain stripping reads the
    # bundle's core semantics; a refused expansion is the structural row below,
    # and no config is derived from it (as for an illegal declaration below).
    try:
        draft = target_expansion.expand_target_document(draft, write_derived=False)
        target_refused = False
    except ToolchainRefusal:
        target_refused = True
    # Dormant-domain keys never resolve and never error: the semantics switch
    # is zero-friction from any green state (the draft retains them).
    draft, dormant_keys = _strip_dormant_domain(draft)
    parsed = parse_deployment_document(draft)
    errors = _structural_errors(draft, parsed.dp)

    # An illegal declaration is a keyed, REMEDIABLE row (superseding the plain structural row of its key), and no
    # hypothetical config is derived from it, nor from any declaration the one judge refuses. The legal-value-set
    # law is the run's: the draft's layout and each section's own declarations under the run's state.
    illegal = declared_legality_errors(draft)
    illegal_keys = {row["key"] for row in illegal}
    errors = [error for error in errors if error["key"] not in illegal_keys]
    errors.extend(illegal)

    # Field-domain errors (type/bounds/options), the parser's and the one judge's
    # (the run's layout and the schemas consumers apply; G4 interface note 3),
    # are keyed, remediable rows too, once each (a keyed row supersedes the plain
    # structural row stating its message, below).
    judge = document_declaration_errors(draft)
    if not judge:  # the run applies the contract to the declarations its judge admits, as the document door states it
        errors.extend(_contract_errors(draft))
    judged = [(attach_error_key(message), message) for message in judge
              if message not in {row["message"] for row in illegal}]
    errors.extend({(row["key"], row["message"]): row for row in [
        *validate_against_registry(parsed.known_flat_keys()),
        *(field_domain_error(key, message) for key, message in judged if key and message.startswith(key))]}.values())

    # Retired keys come back as keyed rows with one-click MIGRATION remedies,
    # each carrying its SCOPE, superseding every other row for the same
    # (scope, key). Registry-bridged legacy keys still preview through the
    # meaning-preserving bridge; fully-removed keys never reach the derivation.
    retired = retired_key_errors(parsed.retirement_scan_view(), paths=parsed.retired_paths)
    retired_pairs = {(row["scope"], row["key"]) for row in retired}
    errors = [error for error in errors if (_error_key_scope(error["key"]), error["key"]) not in retired_pairs]
    errors.extend(retired)

    resolved: Dict[str, Any] = {}
    derived: Dict[str, Dict[str, Any]] = {}
    if not (judge or target_refused or section_shape_errors(draft)):  # the run refuses these before deriving
        try:  # the sections the run takes (``run_sections``), through its own derivation (``run_config``)
            resolved = run_config(*run_sections(_run_sections(draft, parsed))[:2])
        except ValueError as exc:
            resolved = {}
            errors.append({"key": attach_error_key(str(exc)), "message": str(exc), "rule_id": "derivation"})
        else:
            for key, value in parsed.top.items():
                resolved.setdefault(key, value)
            derived = derived_view(resolved)

    keyed = {row["message"] for row in errors if row.get("rule_id") != "config_shape"}
    errors = [row for row in errors if row.get("rule_id") != "config_shape" or row["message"] not in keyed]
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
