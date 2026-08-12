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
from mimarsinan.config_schema.registry.entries_endpoint import ENTRIES as _ENDPOINT
from mimarsinan.config_schema.registry.entries_execution import ENTRIES as _EXECUTION
from mimarsinan.config_schema.registry.entries_model import ENTRIES as _MODEL
from mimarsinan.config_schema.registry.entries_physics import ENTRIES as _PHYSICS
from mimarsinan.config_schema.registry.entries_platform import ENTRIES as _PLATFORM
from mimarsinan.config_schema.registry.entries_pruning import ENTRIES as _PRUNING
from mimarsinan.config_schema.registry.entries_run import ENTRIES as _RUN
from mimarsinan.config_schema.registry.entries_semantics import (
    ENTRIES as _SEMANTICS,
)
from mimarsinan.config_schema.registry.entries_tuning import ENTRIES as _TUNING
from mimarsinan.config_schema.registry.groups import CONCERN_GROUPS, VALID_GROUP_IDS
from mimarsinan.config_schema.registry.relevance import Relevance
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


def _inject_domain_relevance(entry: ConfigKeySchema) -> ConfigKeySchema:
    """Domain-conditional EXISTENCE, injected generically: an event-domain key
    exists only under core_semantics='spiking', a value-domain key only under
    'mvm' — relevance controls existence, so the whole spiking surface leaves
    the wizard when the domain says it does not exist."""
    if entry.domain == "universal" or entry.flat_key == "core_semantics":
        return entry
    wanted = "spiking" if entry.domain == "event" else "mvm"
    gate = Relevance.when("core_semantics", in_=(wanted,))
    combined = (
        gate if entry.relevant.op == "always"
        else Relevance.all_of(gate, entry.relevant)
    )
    return dataclasses.replace(entry, relevant=combined)


def domain_keys(domain: str, section: str | None = None) -> Tuple[str, ...]:
    """Registry keys of one domain (optionally restricted to a section)."""
    return tuple(
        k for k, e in REGISTRY.items()
        if e.domain == domain and (section is None or e.section == section)
    )


def split_domain_dormant(dp: Mapping[str, Any], core_semantics: str):
    """Split a deployment-parameters mapping into (active, dormant) by domain:
    dormant keys belong to the OTHER core-semantics domain — the authoring
    surface keeps them in the draft but excludes them from resolution and
    emission (restored on switch-back)."""
    blocked = "value" if str(core_semantics) != "mvm" else "event"
    active: Dict[str, Any] = {}
    dormant: Dict[str, Any] = {}
    for key, value in dp.items():
        entry = _REGISTRY.get(key)
        if entry is not None and entry.domain == blocked:
            dormant[key] = value
        else:
            active[key] = value
    return active, dormant


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


def validate_registry(entries: Tuple[ConfigKeySchema, ...]) -> Dict[str, ConfigKeySchema]:
    """Registry BUILD-TIME invariants (defaults already injected)."""
    table: Dict[str, ConfigKeySchema] = {}
    for entry in entries:
        if entry.flat_key in table:
            raise ValueError(f"duplicate registry entry {entry.flat_key!r}")
        if entry.group not in VALID_GROUP_IDS:
            raise ValueError(f"{entry.flat_key!r}: unknown group {entry.group!r}")
        if entry.provenance is not None and not (
            entry.has_default()
            or entry.derived_default is not None
            or entry.category is Category.DERIVED
        ):
            raise ValueError(
                f"{entry.flat_key!r}: provenance {entry.provenance!r} without a way "
                "to produce a concrete value — declare derived_default (the wizard "
                "renders the VALUE, never prose about its source)"
            )
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


# The event-domain SHAPE rule the tags must satisfy (the drift guard that
# replaced the old hardcoded forbidden list): a key whose name or group is
# spike-shaped must declare domain='event'; the temporal platform grids are
# event; activation_bits is the value-domain boundary grid.
_EVENT_PREFIXES = ("lif_", "ttfs_", "ttfsq_", "casc_", "sync_", "spike_", "spiking_")
_EVENT_KEYS = frozenset({
    "simulation_steps", "target_tq", "encoding_layer_placement",
    "negative_value_shift", "per_channel_theta", "s_aware_theta_quantile",
    "s_allocation", "s_allocation_explicit", "s_allocation_budget",
})
_VALUE_KEYS = frozenset({"activation_bits", "value_parity_samples"})


def _assert_domain_tags(entries) -> None:
    for e in entries:
        event_shaped = (
            e.flat_key.startswith(_EVENT_PREFIXES)
            or e.flat_key in _EVENT_KEYS
            or (e.group == "spiking" and e.flat_key != "core_semantics")
        )
        if event_shaped and e.domain != "event":
            raise ValueError(
                f"{e.flat_key!r}: spike-shaped key must declare domain='event' "
                f"(got {e.domain!r})"
            )
        if e.flat_key in _VALUE_KEYS and e.domain != "value":
            raise ValueError(
                f"{e.flat_key!r}: value-grid key must declare domain='value' "
                f"(got {e.domain!r})"
            )


_REGISTRY: Dict[str, ConfigKeySchema] = validate_registry(
    tuple(
        _inject_domain_relevance(_inject_default(e))
        for e in (_RUN + _MODEL + _SEMANTICS + _CONVERSION + _PRUNING + _TUNING + _ENDPOINT + _EXECUTION + _PLATFORM + _PHYSICS)
    )
)
_assert_domain_tags(_REGISTRY.values())

REGISTRY: Mapping[str, ConfigKeySchema] = MappingProxyType(_REGISTRY)


def schema_for(flat_key: str) -> ConfigKeySchema:
    """The schema record for one flat key; raises KeyError for unknown keys."""
    return REGISTRY[flat_key]


def effective_value(config: Mapping[str, Any], flat_key: str) -> Any:
    """THE value a consumer sees for ``flat_key`` under ``config``.

    Explicit declaration (``None`` means absent) > schema default > the
    registry's ``derived_default`` for this config state. The one accessor for
    frozen fallbacks, so the number a consumer reads and the number the wizard
    renders as the derived value cannot drift.
    """
    entry = REGISTRY[flat_key]
    value = config.get(flat_key)
    if value is not None:
        return value
    if entry.has_default():
        return entry.default
    if entry.derived_default is not None:
        return entry.derived_default(config)
    return None


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
            "domain": entry.domain,
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
