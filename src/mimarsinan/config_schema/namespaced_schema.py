"""Namespaced config schema: concern groups + flat<->namespaced translation shim.

The per-key knowledge lives in ``config_schema.registry`` (the configurability
SSOT); this module derives the KeySpec provenance table and the byte-identical
flat<->namespaced bijection from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
)
from mimarsinan.config_schema.registry.build import REGISTRY
from mimarsinan.config_schema.registry.groups import CONCERN_GROUPS, VALID_GROUP_IDS

__all__ = [
    "CONCERN_GROUPS",
    "KEY_SPECS",
    "KeySpec",
    "LEGACY_KEY_TABLE",
    "NAMESPACED_KEY_TABLE",
    "keys_with_derivation",
    "keys_with_exposure",
    "provenance_table",
    "registered_flat_keys",
    "to_flat",
    "to_namespaced",
    "unregistered_default_keys",
]

_VALID_DERIVATIONS = frozenset({"default", "preset", "derived", "runtime"})

_VALID_EXPOSURES = frozenset({"user", "derived", "system", "runtime"})


@dataclass(frozen=True)
class KeySpec:
    """Provenance + namespacing for one flat deployment/platform key.

    ``flat_key`` is the runtime SSOT key; ``group.name`` is the namespaced
    path; ``owner``/``derivation`` record the consumer and value source.
    """

    flat_key: str
    group: str
    name: str
    owner: str
    derivation: str
    exposure: str = "system"

    def __post_init__(self) -> None:
        if self.group not in VALID_GROUP_IDS:
            raise ValueError(f"KeySpec {self.flat_key!r}: unknown group {self.group!r}")
        if self.derivation not in _VALID_DERIVATIONS:
            raise ValueError(
                f"KeySpec {self.flat_key!r}: unknown derivation {self.derivation!r}"
            )
        if self.exposure not in _VALID_EXPOSURES:
            raise ValueError(
                f"KeySpec {self.flat_key!r}: unknown exposure {self.exposure!r}"
            )

    @property
    def namespaced_path(self) -> str:
        return f"{self.group}.{self.name}"


_ALL_KEY_SPECS: Tuple[KeySpec, ...] = tuple(
    KeySpec(
        flat_key=entry.flat_key,
        group=entry.group,
        name=entry.flat_key,
        owner=entry.owner,
        derivation=entry.derivation,
        exposure=entry.exposure,
    )
    for entry in REGISTRY.values()
)

KEY_SPECS: Dict[str, KeySpec] = {s.flat_key: s for s in _ALL_KEY_SPECS}

LEGACY_KEY_TABLE: Dict[str, str] = {
    s.flat_key: s.namespaced_path for s in _ALL_KEY_SPECS
}

# The flat<->namespaced shim must be a bijection; the inverse is validated for uniqueness at import.
NAMESPACED_KEY_TABLE: Dict[str, str] = {}
for _flat, _path in LEGACY_KEY_TABLE.items():
    if _path in NAMESPACED_KEY_TABLE:
        raise ValueError(
            f"Namespaced path collision: {_path!r} from {_flat!r} and "
            f"{NAMESPACED_KEY_TABLE[_path]!r}"
        )
    NAMESPACED_KEY_TABLE[_path] = _flat


def to_namespaced(flat: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Project a flat config into nested concern groups.

    Registered keys land under ``group -> name``; unregistered keys pass
    through under ``run`` so nothing is dropped (round-trips via ``to_flat``).
    """
    nested: Dict[str, Dict[str, Any]] = {}
    for key, value in flat.items():
        spec = KEY_SPECS.get(key)
        if spec is None:
            nested.setdefault("run", {})[key] = value
        else:
            nested.setdefault(spec.group, {})[spec.name] = value
    return nested


def to_flat(nested: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Invert ``to_namespaced``: flatten concern groups back to legacy flat keys."""
    flat: Dict[str, Any] = {}
    for group, members in nested.items():
        if not isinstance(members, Mapping):
            raise ValueError(f"Group {group!r} must map to a dict, got {type(members)}")
        for name, value in members.items():
            path = f"{group}.{name}"
            flat_key = NAMESPACED_KEY_TABLE.get(path)
            if flat_key is None:
                if name in flat:
                    raise ValueError(f"Flatten collision on pass-through key {name!r}")
                flat[name] = value
            else:
                flat[flat_key] = value
    return flat


def provenance_table() -> Dict[str, Dict[str, str]]:
    """Return {flat_key: {group, name, owner, derivation, namespaced_path}}."""
    return {
        s.flat_key: {
            "group": s.group,
            "name": s.name,
            "owner": s.owner,
            "derivation": s.derivation,
            "exposure": s.exposure,
            "namespaced_path": s.namespaced_path,
        }
        for s in _ALL_KEY_SPECS
    }


def registered_flat_keys() -> frozenset:
    """All flat keys carrying a KeySpec (registered in the provenance table)."""
    return frozenset(KEY_SPECS)


def keys_with_derivation(derivation: str) -> frozenset:
    """Flat keys whose recorded provenance is ``derivation`` (see _VALID_DERIVATIONS)."""
    if derivation not in _VALID_DERIVATIONS:
        raise ValueError(f"unknown derivation {derivation!r}")
    return frozenset(k for k, s in KEY_SPECS.items() if s.derivation == derivation)


def keys_with_exposure(exposure: str) -> frozenset:
    """Flat keys with the requested persistence exposure."""
    if exposure not in _VALID_EXPOSURES:
        raise ValueError(f"unknown exposure {exposure!r}")
    return frozenset(k for k, s in KEY_SPECS.items() if s.exposure == exposure)


def unregistered_default_keys() -> frozenset:
    """Default flat keys (deployment + platform) that have no KeySpec.

    The registry covers every key by construction; this stays as the loud
    empty-set ratchet the architecture tests assert on.
    """
    defaults = set(DEFAULT_DEPLOYMENT_PARAMETERS) | set(DEFAULT_PLATFORM_CONSTRAINTS)
    return frozenset(defaults - set(KEY_SPECS))
