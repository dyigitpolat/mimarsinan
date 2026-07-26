"""Core-semantics taxonomy: the chip-domain axis (spiking vs value/MVM), queried by intent."""

from __future__ import annotations

from typing import Any, FrozenSet, Mapping

CORE_SEMANTICS_SPIKING = "spiking"
CORE_SEMANTICS_MVM = "mvm"
CORE_SEMANTICS_OPTIONS: tuple = (CORE_SEMANTICS_SPIKING, CORE_SEMANTICS_MVM)
CORE_SEMANTICS_VALUES: FrozenSet[str] = frozenset(CORE_SEMANTICS_OPTIONS)

# The spiking_mode a value-domain plan resolves: never in ALL_SPIKING_MODES,
# never falsy (a falsy mode would _norm back to "lif" in spiking_semantics).
INERT_SPIKING_MODE = "none"


def require_known_core_semantics(value: Any) -> str:
    """Return the normalized core_semantics, or raise ``ValueError`` if unknown."""
    semantics = str(value or CORE_SEMANTICS_SPIKING)
    if semantics in CORE_SEMANTICS_VALUES:
        return semantics
    raise ValueError(
        f"unknown core_semantics {semantics!r}; valid values: "
        f"{sorted(CORE_SEMANTICS_VALUES)}."
    )


def resolve_core_semantics(config: Mapping[str, Any]) -> str:
    """THE core_semantics resolution: the config key, defaulting to spiking."""
    return require_known_core_semantics(config.get("core_semantics"))


def is_mvm_core_semantics(value: Any) -> bool:
    """Whether ``value`` names the value-domain MVM family."""
    return require_known_core_semantics(value) == CORE_SEMANTICS_MVM
