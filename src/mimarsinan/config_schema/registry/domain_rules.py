"""Document-level core_semantics rules, derived from the registry domain tags."""

from __future__ import annotations

from typing import Any, List, Mapping

from mimarsinan.chip_simulation.activation_semantics import RETIRED_SPIKING_KEYS
from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics


def _registry():
    from mimarsinan.config_schema.registry import REGISTRY
    return REGISTRY


def _domain_of(key: str) -> str:
    entry = _registry().get(key)
    return entry.domain if entry is not None else "universal"


def mvm_document_errors(config: Mapping[str, Any]) -> List[str]:
    """Domain-key document rules for BOTH directions of core_semantics.

    The forbidden sets derive from the registry ``domain`` tags (the one SSOT
    the wizard existence rules and emission stripping also read); messages
    START with the offending key so the resolve channel keys them correctly.
    """
    dp = config.get("deployment_parameters")
    if not isinstance(dp, Mapping):
        return []
    pc = config.get("platform_constraints")
    pc = pc if isinstance(pc, Mapping) else {}
    if not is_mvm_core_semantics(dp.get("core_semantics")):
        return [
            f"{key} declares a value-domain boundary grid; an event-domain "
            f"(spiking) platform declares target_tq instead. Remove the key "
            f"or set core_semantics='mvm'."
            for key in sorted(set(dp) | set(pc))
            if _domain_of(key) == "value"
        ]
    errors = [
        f"{key} is event-domain (spiking) configuration and is not authorable "
        f"in a value-domain (core_semantics='mvm') deployment. Remove the key."
        for key in sorted(dp)
        if _domain_of(key) == "event" and key not in RETIRED_SPIKING_KEYS
    ]
    errors.extend(
        f"{key} declares a temporal (spiking) grid a value-domain "
        f"(core_semantics='mvm') deployment does not have. Remove the key."
        for key in sorted(pc) if _domain_of(key) == "event"
    )
    return errors
