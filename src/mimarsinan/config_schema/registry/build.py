"""Registry assembly: defaults injection, coverage validation, queries, serialization."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Tuple

from mimarsinan.config_schema.defaults import (
    CONFIG_KEYS_SET,
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
)
from mimarsinan.config_schema.registry.entries_conversion import ENTRIES as _CONVERSION
from mimarsinan.config_schema.registry.entries_model import ENTRIES as _MODEL
from mimarsinan.config_schema.registry.entries_platform import ENTRIES as _PLATFORM
from mimarsinan.config_schema.registry.entries_run import ENTRIES as _RUN
from mimarsinan.config_schema.registry.groups import CONCERN_GROUPS, VALID_GROUP_IDS
from mimarsinan.config_schema.registry.types import Category, ConfigKeySchema

# Document keys that are not flat pipeline-config keys: the top-level run
# document surface plus platform structural extras. Everything else the
# registry covers MUST come from the live CONFIG_KEYS_SET.
NON_PIPELINE_DOC_KEYS: FrozenSet[str] = frozenset({
    "data_provider_name", "experiment_name", "generated_files_path",
    "datasets_path", "start_step", "stop_step", "target_metric_override",
    "pipeline_mode",
    "max_axons", "max_neurons", "has_bias", "search_space",
})

_TOP_DEFAULTS: Dict[str, Any] = {
    "data_provider_name": "MNIST_DataProvider",
    "experiment_name": "experiment",
    "generated_files_path": "./generated",
    "seed": 0,
}


def _inject_default(entry: ConfigKeySchema) -> ConfigKeySchema:
    """Pull the entry's default from the defaults SSOT (entries never declare one)."""
    source: Mapping[str, Any]
    if entry.section == "deployment_parameters":
        source = DEFAULT_DEPLOYMENT_PARAMETERS
    elif entry.section == "platform_constraints":
        source = DEFAULT_PLATFORM_CONSTRAINTS
    else:
        source = _TOP_DEFAULTS
    if entry.flat_key not in source:
        return entry
    return dataclasses.replace(entry, default=source[entry.flat_key])


def _validate(entries: Tuple[ConfigKeySchema, ...]) -> Dict[str, ConfigKeySchema]:
    table: Dict[str, ConfigKeySchema] = {}
    for entry in entries:
        if entry.flat_key in table:
            raise ValueError(f"duplicate registry entry {entry.flat_key!r}")
        if entry.group not in VALID_GROUP_IDS:
            raise ValueError(f"{entry.flat_key!r}: unknown group {entry.group!r}")
        if entry.provided_by is not None:
            if entry.provided_by not in VALID_GROUP_IDS:
                raise ValueError(
                    f"{entry.flat_key!r}: unknown provided_by group {entry.provided_by!r}"
                )
            if entry.provided_by == entry.group:
                raise ValueError(
                    f"{entry.flat_key!r}: provided_by must name a DIFFERENT group"
                )
        table[entry.flat_key] = entry

    expected = set(CONFIG_KEYS_SET) | NON_PIPELINE_DOC_KEYS
    registered = set(table)
    missing = expected - registered
    stray = registered - expected
    if missing or stray:
        raise ValueError(
            "config-key registry drift vs the live CONFIG_KEYS_SET: "
            f"missing={sorted(missing)} stray={sorted(stray)}"
        )
    return table


_REGISTRY: Dict[str, ConfigKeySchema] = _validate(
    tuple(_inject_default(e) for e in (_RUN + _MODEL + _CONVERSION + _PLATFORM))
)

REGISTRY: Mapping[str, ConfigKeySchema] = MappingProxyType(_REGISTRY)


def schema_for(flat_key: str) -> ConfigKeySchema:
    """The schema record for one flat key; raises KeyError for unknown keys."""
    return REGISTRY[flat_key]


def keys_in_category(category: Category) -> FrozenSet[str]:
    return frozenset(k for k, e in REGISTRY.items() if e.category is category)


def section_keys(section: str) -> FrozenSet[str]:
    return frozenset(k for k, e in REGISTRY.items() if e.section == section)


def serialize_registry() -> Dict[str, Any]:
    """JSON-safe registry payload for ``GET /api/config_schema``."""
    keys: Dict[str, Any] = {}
    for flat_key, entry in REGISTRY.items():
        record: Dict[str, Any] = {
            "key": flat_key,
            "group": entry.group,
            "owner": entry.owner,
            "section": entry.section,
            "type": entry.type.value,
            "category": entry.category.value,
            "label": entry.label,
            "doc": entry.doc,
            "effect": entry.effect,
            "unit": entry.unit,
            "options": (
                list(opts) if (opts := entry.resolved_options()) is not None else None
            ),
            "bounds": list(entry.bounds) if entry.bounds is not None else None,
            "relevant": entry.relevant.to_json(),
            "promote_when": (
                entry.promote_when.to_json() if entry.promote_when is not None else None
            ),
            "empty_means": entry.empty_means,
            "provided_by": entry.provided_by,
            "derived_from": list(entry.derived_from),
            "declarable": entry.declarable,
            "important": entry.important,
            "provenance": entry.provenance,
            "hidden": entry.hidden,
        }
        if entry.has_default():
            record["default"] = entry.default
        keys[flat_key] = record
    return {
        "groups": [dict(g) for g in CONCERN_GROUPS],
        "keys": keys,
    }
