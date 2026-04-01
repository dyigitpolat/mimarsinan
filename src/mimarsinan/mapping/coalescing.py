"""
Single canonical coalescing flag for layout, packing, and scheduling.

Only ``allow_coalescing`` is accepted in ``platform_constraints`` and merged
pipeline config. Deprecated names are rejected with a clear error.
"""

from __future__ import annotations

from typing import Any, List, Mapping, MutableMapping, Tuple

CANONICAL_KEY = "allow_coalescing"

# Rejected if present (migration: use CANONICAL_KEY only).
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
    """
    Return the effective coalescing bit from ``allow_coalescing`` only.

    If no key is present, returns *default*. Deprecated keys raise
    :class:`CoalescingConfigError`.
    """
    errs = coalescing_config_errors(mapping)
    if errs:
        raise CoalescingConfigError(errs[0])
    return bool(mapping.get(CANONICAL_KEY, default))


def normalize_coalescing_config(pcfg: MutableMapping[str, Any]) -> bool:
    """
    Mutate *pcfg* in place: ensure ``allow_coalescing`` is a bool (default ``False``).

    Returns the resolved boolean. Raises :class:`CoalescingConfigError` if
    deprecated coalescing keys are present.
    """
    errs = coalescing_config_errors(pcfg)
    if errs:
        raise CoalescingConfigError("; ".join(errs))
    resolved = bool(pcfg.get(CANONICAL_KEY, False))
    pcfg[CANONICAL_KEY] = resolved
    return resolved
