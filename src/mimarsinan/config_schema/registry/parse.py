"""Classify every key of a deployment-config document against the registry."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from mimarsinan.config_schema.registry.build import REGISTRY, section_keys
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers
from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA, RECIPE_SCHEMA
from mimarsinan.config_schema.registry.retired_keys import retired_keys_in_scope
from mimarsinan.config_schema.registry.types import FieldType

CORE_FIELDS = frozenset(CORE_TYPE_SCHEMA.fields)
RECIPE_FIELDS = frozenset(RECIPE_SCHEMA.fields)  # the fields the recipe's consumer reads
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
    # Registry-REMOVED retired keys, by scope: neither unknown (no tray noise)
    # nor schema-known (they never reach the derivation or the explicit diff);
    # the retired-key rule owns them with a keyed migration remedy.
    retired: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "deployment_parameters": {}, "platform_constraints": {},
    })
    # WHERE each retired declaration was found: scope -> {key: container path
    # segments under the document root}. The wizard's structural pc shapes nest
    # the parsed body (platform_constraints.user / .auto.fixed), so a remedy
    # clearing a retired key must target the container it actually lives in —
    # a scope-root clear would miss it and the error would survive the click.
    retired_paths: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "deployment_parameters": {}, "platform_constraints": {},
    })
    # The container path (segments under the document root) of every schema-known declaration.
    paths: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # Declarations their consumer reads as another value (``read_like_consumers``): key -> the value
    # read, which top/dp/pc already carry in place of the value as written.
    read_as: Dict[str, Any] = field(default_factory=dict)

    def known_flat_keys(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        merged.update(self.top)
        merged.update(self.dp)
        merged.update(self.pc)
        return merged

    def retirement_scan_view(self) -> Dict[str, Dict[str, Any]]:
        """Both scopes with their retired declarations folded back in — the
        mapping the retired-key scan judges (registry-known legacy bridges
        parse into the section dicts; fully-removed keys ride ``retired``)."""
        return {
            "deployment_parameters": {**self.retired["deployment_parameters"],
                                      **self.dp},
            "platform_constraints": {**self.retired["platform_constraints"],
                                     **self.pc},
        }


def _check_nested_fields(value: Any, allowed: frozenset, path: str, out: ParsedDocument) -> None:
    if not isinstance(value, Mapping):
        return
    for sub_key in value:
        if sub_key not in allowed:
            out.unknown.append(f"{path}.{sub_key}")


def _check_value_shape(flat_key: str, value: Any, path: str, out: ParsedDocument) -> None:
    entry = REGISTRY[flat_key]
    schema = entry.value_schema
    if entry.type is FieldType.CORES and isinstance(value, list):
        for i, item in enumerate(value):
            _check_nested_fields(item, CORE_FIELDS, f"{path}[{i}]", out)
    elif entry.type is FieldType.RECIPE:
        _check_nested_fields(value, RECIPE_FIELDS, path, out)
    elif flat_key == "preprocessing":
        _check_nested_fields(value, PREPROCESSING_FIELDS, path, out)
    elif schema is not None and schema.fields and schema.entries is None:
        _check_nested_fields(value, frozenset(schema.fields), path, out)


def _parse_section(
    data: Mapping[str, Any], section: str, prefix: str, out: ParsedDocument,
    target: Dict[str, Any], container_path: tuple = (),
) -> None:
    known = section_keys(section)
    retired = retired_keys_in_scope(section)
    for key, value in data.items():
        if key.startswith("_"):
            out.meta[key] = value
        elif key in known:
            target[key] = value
            out.paths[key] = tuple(container_path or (section,))
            _check_value_shape(key, value, f"{prefix}{key}", out)
        elif key in retired:
            out.retired[section][key] = value
            out.retired_paths[section][key] = list(container_path or (section,))
        else:
            out.unknown.append(f"{prefix}{key}")


def _parse_platform(pc: Mapping[str, Any], out: ParsedDocument) -> None:
    """Flat pc parses directly; the wizard's mode=user/auto shape parses its bodies."""
    mode = pc.get("mode")
    if mode is None:
        _parse_section(pc, "platform_constraints", "platform_constraints.", out, out.pc)
        return
    for key, value in pc.items():
        # A structural container that is no object nests nothing to parse: the parser reports it unknown.
        if key not in _PC_STRUCTURAL or (key != "mode" and value is not None and not isinstance(value, Mapping)):
            out.unknown.append(f"platform_constraints.{key}")
    raw_auto = pc.get("auto")
    auto: Mapping[str, Any] = raw_auto if isinstance(raw_auto, Mapping) else {}
    if mode == "user":
        body, body_path = pc.get("user"), ("platform_constraints", "user")
    else:
        body = auto.get("fixed")
        body_path = ("platform_constraints", "auto", "fixed")
    if isinstance(body, Mapping):
        _parse_section(body, "platform_constraints", "platform_constraints.", out,
                       out.pc, container_path=body_path)
    search_space = pc.get("search_space") or auto.get("search_space")
    if isinstance(search_space, Mapping):
        out.pc["search_space"] = dict(search_space)


def parse_deployment_document(config: Mapping[str, Any]) -> ParsedDocument:
    """Split a deployment JSON into known flat keys / meta keys / unknown paths.

    Nothing is dropped: unknown keys are reported as dotted paths so callers
    surface them loudly instead of silently absorbing or losing them. Known
    values are read as their consumers read them (``read_as`` names each one
    read as another value), so every judge downstream judges what a run runs.
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
            out.paths[key] = ()
        else:
            out.unknown.append(key)
    _read_like_consumers(out)
    return out


def _read_like_consumers(out: ParsedDocument) -> None:
    """Carry, in each section, the value every declaration's consumer reads (``read_like_consumers``)."""
    declared = out.known_flat_keys()
    for key, value in read_like_consumers(declared).items():
        if type(value) is type(declared[key]) and value == declared[key]:
            continue
        out.read_as[key] = value
        for section in (out.top, out.dp, out.pc):
            if key in section:
                section[key] = value


def read_document_like_consumers(config: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of the document in which every declaration its consumer reads as another value carries
    that value, in the container that declares it (the document door's reading, before it judges)."""
    parsed = parse_deployment_document(config)
    document = copy.deepcopy(dict(config))
    for key, value in parsed.read_as.items():
        container = document
        for segment in parsed.paths[key]:
            container = container[segment]
        container[key] = value
    return document
