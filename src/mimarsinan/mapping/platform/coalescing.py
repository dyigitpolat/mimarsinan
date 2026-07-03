"""Single canonical coalescing flag for layout, packing, and scheduling."""

from __future__ import annotations

import math
from typing import Any, List, Mapping, MutableMapping, Tuple

CANONICAL_KEY = "allow_coalescing"

LEGACY_COALESCING_KEYS: Tuple[str, ...] = (
    "allow_core_coalescing",
    "allow_axon_coalescing",
    "allow_axon_tiling",
)


class CoalescingConfigError(ValueError):
    """Raised when platform_constraints use deprecated coalescing keys."""


def coalescing_config_errors(mapping: Mapping[str, Any]) -> List[str]:
    """Return human-readable errors for any deprecated coalescing keys in *mapping*."""
    errors: List[str] = []
    for k in LEGACY_COALESCING_KEYS:
        if k in mapping:
            errors.append(
                f"platform_constraints: key {k!r} is not supported; use {CANONICAL_KEY!r} only."
            )
    return errors


def resolve_allow_coalescing(mapping: Mapping[str, Any], *, default: bool = False) -> bool:
    """Return the effective coalescing bit from allow_coalescing only (default if absent); deprecated keys raise CoalescingConfigError."""
    errs = coalescing_config_errors(mapping)
    if errs:
        raise CoalescingConfigError(errs[0])
    return bool(mapping.get(CANONICAL_KEY, default))


def normalize_coalescing_config(pcfg: MutableMapping[str, Any]) -> bool:
    """Mutate pcfg in place so allow_coalescing is a bool (default False) and return it; raises CoalescingConfigError on deprecated keys."""
    errs = coalescing_config_errors(pcfg)
    if errs:
        raise CoalescingConfigError("; ".join(errs))
    resolved = bool(pcfg.get(CANONICAL_KEY, False))
    pcfg[CANONICAL_KEY] = resolved
    return resolved


def coalescing_fragment_count(input_count: int, max_axons: int) -> int:
    """Number of axon-coalescing fragments for *input_count* axons on one core type."""
    if max_axons <= 0:
        return 1
    if input_count <= max_axons:
        return 1
    return int(math.ceil(input_count / max_axons))
