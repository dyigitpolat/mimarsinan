"""Classify every key of a deployment-config document against the registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from mimarsinan.config_schema.defaults import DEFAULT_TRAINING_RECIPE
from mimarsinan.config_schema.registry.build import REGISTRY, section_keys
from mimarsinan.config_schema.registry.types import FieldType

CORE_FIELDS = frozenset({"max_axons", "max_neurons", "count", "has_bias"})
RECIPE_FIELDS = frozenset(DEFAULT_TRAINING_RECIPE)
PREPROCESSING_FIELDS = frozenset({"resize_to", "normalize", "interpolation"})

# Platform-constraints structural containers (wizard hw-search shape).
_PC_STRUCTURAL = frozenset({"mode", "user", "auto", "fixed", "search_space"})


@dataclass
class ParsedDocument:
    """A deployment JSON split into schema-known values and unknown key paths."""

    top: Dict[str, Any] = field(default_factory=dict)
    dp: Dict[str, Any] = field(default_factory=dict)
    pc: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    unknown: List[str] = field(default_factory=list)

    def known_flat_keys(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        merged.update(self.top)
        merged.update(self.dp)
        merged.update(self.pc)
        return merged


def _check_nested_fields(value: Any, allowed: frozenset, path: str, out: ParsedDocument) -> None:
    if not isinstance(value, Mapping):
        return
    for sub_key in value:
        if sub_key not in allowed:
            out.unknown.append(f"{path}.{sub_key}")


def _check_value_shape(flat_key: str, value: Any, path: str, out: ParsedDocument) -> None:
    entry = REGISTRY[flat_key]
    if entry.type is FieldType.CORES and isinstance(value, list):
        for i, item in enumerate(value):
            _check_nested_fields(item, CORE_FIELDS, f"{path}[{i}]", out)
    elif entry.type is FieldType.RECIPE:
        _check_nested_fields(value, RECIPE_FIELDS, path, out)
    elif flat_key == "preprocessing":
        _check_nested_fields(value, PREPROCESSING_FIELDS, path, out)


def _parse_section(
    data: Mapping[str, Any], section: str, prefix: str, out: ParsedDocument,
    target: Dict[str, Any],
) -> None:
    known = section_keys(section)
    for key, value in data.items():
        if key.startswith("_"):
            out.meta[key] = value
        elif key in known:
            target[key] = value
            _check_value_shape(key, value, f"{prefix}{key}", out)
        else:
            out.unknown.append(f"{prefix}{key}")


def _parse_platform(pc: Mapping[str, Any], out: ParsedDocument) -> None:
    """Flat pc parses directly; the wizard's mode=user/auto shape parses its bodies."""
    mode = pc.get("mode")
    if mode is None:
        _parse_section(pc, "platform_constraints", "platform_constraints.", out, out.pc)
        return
    for key, value in pc.items():
        if key not in _PC_STRUCTURAL:
            out.unknown.append(f"platform_constraints.{key}")
    body = pc.get("user") if mode == "user" else (pc.get("auto") or {}).get("fixed")
    if isinstance(body, Mapping):
        _parse_section(body, "platform_constraints", "platform_constraints.", out, out.pc)
    search_space = pc.get("search_space") or (pc.get("auto") or {}).get("search_space")
    if isinstance(search_space, Mapping):
        out.pc["search_space"] = dict(search_space)


def parse_deployment_document(config: Mapping[str, Any]) -> ParsedDocument:
    """Split a deployment JSON into known flat keys / meta keys / unknown paths.

    Nothing is dropped: unknown keys are reported as dotted paths so callers
    surface them loudly instead of silently absorbing or losing them.
    """
    out = ParsedDocument()
    top_known = section_keys("top")
    for key, value in config.items():
        if key.startswith("_"):
            out.meta[key] = value
        elif key == "deployment_parameters":
            if isinstance(value, Mapping):
                _parse_section(value, "deployment_parameters", "deployment_parameters.", out, out.dp)
        elif key == "platform_constraints":
            if isinstance(value, Mapping):
                _parse_platform(value, out)
        elif key in top_known:
            out.top[key] = value
        else:
            out.unknown.append(key)
    return out
