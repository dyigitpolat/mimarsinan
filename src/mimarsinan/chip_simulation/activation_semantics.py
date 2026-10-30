"""Authored activation-semantics axes (family × variant) and their legacy-mode bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from mimarsinan.chip_simulation.activation_axes import (
    ALL_SPIKING_VARIANTS, ANALYTICAL_VARIANT, CASCADED_VARIANT, LIF_FAMILY,
    LIF_VARIANTS, QUANTIZED_VARIANT, SPIKING_FAMILY_KEY, SPIKING_VARIANT_KEY,
    STREAMED_VARIANT, SYNCHRONIZED_VARIANT, TTFS_FAMILY, TTFS_VARIANTS,
    _contradiction, axes_from_legacy, require_known_spiking_axes,
    require_known_spiking_family,
    LIF_VARIANTS_DESIGNED as LIF_VARIANTS_DESIGNED,
    RETIRED_SPIKING_KEYS as RETIRED_SPIKING_KEYS,
    SPIKING_FAMILIES as SPIKING_FAMILIES,
    legal_spiking_families as legal_spiking_families,
)
from mimarsinan.chip_simulation.spiking_semantics import (
    DEFAULT_TTFS_CYCLE_SCHEDULE,
    is_explicit_ttfs_cycle_schedule,
)


@dataclass(frozen=True)
class ActivationSemantics:
    """One resolved (family, variant) point of the spiking taxonomy."""

    family: str
    variant: str

    @property
    def mode_id(self) -> str:
        """Canonical six-point id: lif · lif_sync · ttfs · ttfs_quantized ·
        ttfs_sync · ttfs_cascaded."""
        if self.family == LIF_FAMILY:
            return "lif" if self.variant == STREAMED_VARIANT else "lif_sync"
        return {
            ANALYTICAL_VARIANT: "ttfs",
            QUANTIZED_VARIANT: "ttfs_quantized",
            SYNCHRONIZED_VARIANT: "ttfs_sync",
            CASCADED_VARIANT: "ttfs_cascaded",
        }[self.variant]

    @property
    def legacy_spiking_mode(self) -> str:
        """The internal mode string the pipeline dispatches on today."""
        if self.family == LIF_FAMILY:
            return "lif"
        return {
            ANALYTICAL_VARIANT: "ttfs",
            QUANTIZED_VARIANT: "ttfs_quantized",
            SYNCHRONIZED_VARIANT: "ttfs_cycle_based",
            CASCADED_VARIANT: "ttfs_cycle_based",
        }[self.variant]

    @property
    def legacy_ttfs_cycle_schedule(self) -> str:
        """The internal schedule string (inert historical default off-cycle)."""
        if self.family == TTFS_FAMILY and self.variant == SYNCHRONIZED_VARIANT:
            return "synchronized"
        return DEFAULT_TTFS_CYCLE_SCHEDULE

    @property
    def is_streamed(self) -> bool:
        """End-to-end event streaming: timing normalized only at encode/readout."""
        return (self.family, self.variant) in (
            (LIF_FAMILY, STREAMED_VARIANT), (TTFS_FAMILY, CASCADED_VARIANT),
        )

    @property
    def is_windowed(self) -> bool:
        """Windowed discipline: counts re-encoded at stage/hop boundaries."""
        return self.variant == SYNCHRONIZED_VARIANT

    @property
    def is_analytical(self) -> bool:
        """Closed-form evaluation, no cycle grid (analytical TTFS family)."""
        return self.family == TTFS_FAMILY and self.variant in (
            ANALYTICAL_VARIANT, QUANTIZED_VARIANT,
        )


def _effective_family(cfg: Mapping[str, Any]) -> Any:
    """The config's family, resolved through the legacy bridge when only the
    retired keys speak; an unresolvable config yields the raw value (its own
    keyed error is the single truth)."""
    family = cfg.get(SPIKING_FAMILY_KEY)
    if family is not None:
        return family
    try:
        return resolve_activation_semantics(cfg).family
    except ValueError:
        return cfg.get(SPIKING_FAMILY_KEY, LIF_FAMILY)


def legal_spiking_variants(cfg: Mapping[str, Any]) -> Tuple[str, ...]:
    """Legal variant set for the config's family; an unknown family rules
    nothing out (its own error is the single truth)."""
    family = _effective_family(cfg)
    if family == LIF_FAMILY:
        return LIF_VARIANTS
    if family == TTFS_FAMILY:
        return TTFS_VARIANTS
    return ALL_SPIKING_VARIANTS


def derived_spiking_variant(cfg: Mapping[str, Any]) -> str:
    """What an absent variant resolves to: lif → STREAMED (the event-driven
    default discipline, P4), ttfs → analytical. Legacy dicts keep their
    historical meaning through the bridge."""
    if cfg.get(SPIKING_VARIANT_KEY) is None and cfg.get("spiking_mode") is not None:
        try:
            return resolve_activation_semantics(cfg).variant
        except ValueError:
            return SYNCHRONIZED_VARIANT
    family = _effective_family(cfg)
    return ANALYTICAL_VARIANT if family == TTFS_FAMILY else STREAMED_VARIANT


def resolve_activation_semantics(config: Mapping[str, Any]) -> ActivationSemantics:
    """Resolve the authored axes, falling back to legacy keys with their exact
    historical meaning.

    Precedence: an explicit ``spiking_variant`` marks axes-authored intent —
    a disagreeing legacy key then raises. With the variant absent, a present
    legacy mode WINS outright (``spiking_family`` may be a merged schema
    default, which must never masquerade as a declaration; ``spiking_mode``
    has no schema default, so its presence is always meaningful).
    """
    variant_raw = config.get(SPIKING_VARIANT_KEY)
    legacy_mode = config.get("spiking_mode")
    legacy_schedule = config.get("ttfs_cycle_schedule")

    if variant_raw is None:
        if legacy_mode is not None:
            family, variant = axes_from_legacy(legacy_mode, legacy_schedule)
        else:
            family = require_known_spiking_family(
                config.get(SPIKING_FAMILY_KEY, LIF_FAMILY)
            )
            variant = derived_spiking_variant({SPIKING_FAMILY_KEY: family})
        return ActivationSemantics(family, variant)

    family = require_known_spiking_family(
        config.get(SPIKING_FAMILY_KEY, LIF_FAMILY)
    )
    family, variant = require_known_spiking_axes(family, variant_raw)
    semantics = ActivationSemantics(family, variant)
    if legacy_mode is not None and (
        str(legacy_mode) != semantics.legacy_spiking_mode
    ):
        raise _contradiction(
            f"spiking_mode={legacy_mode!r} contradicts (spiking_family="
            f"{family!r}, spiking_variant={variant!r}) which resolves to "
            f"{semantics.legacy_spiking_mode!r}."
        )
    if (
        semantics.legacy_spiking_mode == "ttfs_cycle_based"
        and is_explicit_ttfs_cycle_schedule(legacy_schedule)
        and str(legacy_schedule) != semantics.legacy_ttfs_cycle_schedule
    ):
        raise _contradiction(
            f"ttfs_cycle_schedule={legacy_schedule!r} contradicts "
            f"spiking_variant={variant!r}."
        )
    return semantics


def canonical_mode_id(config: Mapping[str, Any]) -> str:
    """The six-point canonical id of the config's resolved semantics."""
    return resolve_activation_semantics(config).mode_id


def fold_spiking_axes(dp: Any) -> None:
    """Fold the resolved axes and their legacy twins into ``dp`` (idempotent).

    Total over every historical shape: axes-only, legacy-only, both
    (consistency-checked), neither (defaults). After the fold every internal
    consumer of ``spiking_mode``/``ttfs_cycle_schedule`` reads the same value
    it always has.
    """
    semantics = resolve_activation_semantics(dp)
    dp[SPIKING_FAMILY_KEY] = semantics.family
    dp[SPIKING_VARIANT_KEY] = semantics.variant
    dp["spiking_mode"] = semantics.legacy_spiking_mode
    dp["ttfs_cycle_schedule"] = semantics.legacy_ttfs_cycle_schedule


def is_streamed_lif(cfg: Mapping[str, Any]) -> bool:
    """The per-segment streamed LIF discipline (total over any config).

    FALSE for value-domain (mvm) configs: the spiking axes are DORMANT there
    and their derived defaults must not leak legality — the P4 default flip
    silently locked allow_scheduling={False} on the mvm scheduling flagship
    (t0_44, caught by the 2026-08-09 re-baseline)."""
    if str(cfg.get("core_semantics", "spiking")) == "mvm":
        return False
    try:
        semantics = resolve_activation_semantics(cfg)
    except ValueError:
        return False
    return semantics.family == LIF_FAMILY and semantics.variant == STREAMED_VARIANT


def effective_legacy_spiking_mode(cfg: Mapping[str, Any]) -> str:
    """The legacy mode string for registry lambdas, TOTAL over every config
    shape: raw documents (pre-fold), resolved configs (post-fold), and
    pre-migration/invalid documents — an invalid axis rules nothing out here
    because its own keyed error is the single truth."""
    mode = cfg.get("spiking_mode")
    if mode is not None:
        return str(mode)
    try:
        return resolve_activation_semantics(cfg).legacy_spiking_mode
    except ValueError:
        return "lif"
