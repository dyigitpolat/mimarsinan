"""The authorable spiking axes: vocabulary, legality, and the retired-key bridge."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from mimarsinan.chip_simulation.spiking_semantics import (
    require_known_spiking_mode,
    ttfs_cycle_schedule,
)

SPIKING_FAMILY_KEY = "spiking_family"
SPIKING_VARIANT_KEY = "spiking_variant"

LIF_FAMILY = "lif"
TTFS_FAMILY = "ttfs"
SPIKING_FAMILIES: Tuple[str, ...] = (LIF_FAMILY, TTFS_FAMILY)

STREAMED_VARIANT = "streamed"
SYNCHRONIZED_VARIANT = "synchronized"
ANALYTICAL_VARIANT = "analytical"
QUANTIZED_VARIANT = "quantized"
CASCADED_VARIANT = "cascaded"

# The authorable vocabulary. (lif, streamed) is the end-to-end event-streamed
# discipline (P3); it becomes the derived default with the starter re-base (P4).
ALL_SPIKING_VARIANTS: Tuple[str, ...] = (
    STREAMED_VARIANT, SYNCHRONIZED_VARIANT, ANALYTICAL_VARIANT,
    QUANTIZED_VARIANT, CASCADED_VARIANT,
)
LIF_VARIANTS: Tuple[str, ...] = (STREAMED_VARIANT, SYNCHRONIZED_VARIANT)
LIF_VARIANTS_DESIGNED: Tuple[str, ...] = LIF_VARIANTS
TTFS_VARIANTS: Tuple[str, ...] = (
    ANALYTICAL_VARIANT, QUANTIZED_VARIANT, SYNCHRONIZED_VARIANT, CASCADED_VARIANT,
)

# Document keys retired by the family/variant taxonomy (P0). The derivation
# still folds the first two as internal values; a DOCUMENT declaring any of
# them gets a keyed migration error with one-click remedies.
RETIRED_SPIKING_KEYS: Tuple[str, ...] = (
    "spiking_mode", "ttfs_cycle_schedule",
    "lif_execution_discipline", "lif_per_hop_retiming",
)


def require_known_spiking_family(family: Any) -> str:
    value = str(family)
    if value in SPIKING_FAMILIES:
        return value
    raise ValueError(
        f"unknown spiking_family {family!r}; valid families: "
        f"{list(SPIKING_FAMILIES)}."
    )


def legal_spiking_families(cfg: Mapping[str, Any]) -> Tuple[str, ...]:
    return SPIKING_FAMILIES


def require_known_spiking_axes(family: Any, variant: Any) -> Tuple[str, str]:
    """Validate one (family, variant) point of the taxonomy."""
    fam = require_known_spiking_family(family)
    var = str(variant)
    legal = LIF_VARIANTS if fam == LIF_FAMILY else TTFS_VARIANTS
    if var not in legal:
        raise ValueError(
            f"spiking_variant {variant!r} is not legal for spiking_family="
            f"{fam!r}; legal variants: {list(legal)}."
        )
    return fam, var


def axes_from_legacy(spiking_mode: Any, schedule: Any = None) -> Tuple[str, str]:
    """The exact meaning-preserving reverse bridge (old lif was windowed)."""
    mode = require_known_spiking_mode(str(spiking_mode))
    if mode == "lif":
        return LIF_FAMILY, SYNCHRONIZED_VARIANT
    if mode == "ttfs":
        return TTFS_FAMILY, ANALYTICAL_VARIANT
    if mode == "ttfs_quantized":
        return TTFS_FAMILY, QUANTIZED_VARIANT
    return TTFS_FAMILY, (
        SYNCHRONIZED_VARIANT
        if ttfs_cycle_schedule(schedule) == "synchronized"
        else CASCADED_VARIANT
    )


def _contradiction(detail: str) -> ValueError:
    return ValueError(
        f"activation-semantics contradiction: {detail} Declare only "
        f"spiking_family/spiking_variant (the retired keys are derivation-"
        f"owned)."
    )
