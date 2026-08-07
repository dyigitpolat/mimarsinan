"""Document-level core_semantics rules: event-domain keys are not authorable under mvm."""

from __future__ import annotations

from typing import Any, List, Mapping

from mimarsinan.chip_simulation.activation_semantics import RETIRED_SPIKING_KEYS
from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics

# Keys whose SEMANTICS are event-domain (spike encoding, thresholds, cycle
# grids). Under core_semantics='mvm' a document declaring one is a
# contradiction, never silently inert.
MVM_FORBIDDEN_DEPLOYMENT_KEYS = frozenset({
    "spiking_family",
    "spiking_variant",
    "spiking_mode",
    "ttfs_cycle_schedule",
    "firing_mode",
    "spike_generation_mode",
    "thresholding_mode",
    "encoding_layer_placement",
    "cycle_accurate_lif_forward",
    "comparator_half_step",
    "negative_value_shift",
    "per_channel_theta",
    "s_aware_theta_quantile",
})

MVM_FORBIDDEN_KEY_PREFIXES = ("lif_", "ttfs_", "ttfsq_", "casc_", "sync_", "spike_")

# Temporal platform grids: a value core has no spike window / AQ level count.
MVM_FORBIDDEN_PLATFORM_KEYS = frozenset({"simulation_steps", "target_tq"})


def _forbidden(key: str) -> bool:
    return key in MVM_FORBIDDEN_DEPLOYMENT_KEYS or key.startswith(
        MVM_FORBIDDEN_KEY_PREFIXES
    )


def mvm_document_errors(config: Mapping[str, Any]) -> List[str]:
    """Domain-key document rules for BOTH directions of core_semantics."""
    dp = config.get("deployment_parameters")
    if not isinstance(dp, Mapping):
        return []
    if not is_mvm_core_semantics(dp.get("core_semantics")):
        pc_spiking = config.get("platform_constraints")
        if isinstance(pc_spiking, Mapping) and "activation_bits" in pc_spiking:
            return [
                "activation_bits declares a value-domain boundary grid; an "
                "event-domain (spiking) platform declares target_tq instead. "
                "Remove the key or set core_semantics='mvm'."
            ]
        return []
    errors = [
        f"core_semantics='mvm': {key} is event-domain (spiking) configuration "
        f"and is not authorable in a value-domain deployment. Remove the key."
        # Retired taxonomy keys report through the retired-key rule instead.
        for key in sorted(dp)
        if _forbidden(key) and key not in RETIRED_SPIKING_KEYS
    ]
    pc = config.get("platform_constraints")
    if isinstance(pc, Mapping):
        errors.extend(
            f"core_semantics='mvm': {key} declares a temporal (spiking) grid a "
            f"value-domain deployment does not have. Remove the key."
            for key in sorted(pc) if key in MVM_FORBIDDEN_PLATFORM_KEYS
        )
    return errors
