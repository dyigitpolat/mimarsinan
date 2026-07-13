"""Fold the ConversionPolicy recipe into deployment parameters (+ paired knobs)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Optional, Set

from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

# Registry keys the recipe owns OUTRIGHT (correctness mechanisms, never
# knobs): an explicit value is overwritten, so none is ever honored/stored.
RECIPE_OWNED_CORRECTNESS_KEYS = frozenset({"cycle_accurate_lif_forward"})


def _fold_sim_enables(
    dp: MutableMapping[str, Any],
    sim_enables: Mapping[str, bool],
    spiking_mode: str,
    explicit_keys: Set[str],
) -> None:
    """Backend enables: capability off is authoritative (an explicit ON is
    rejected loudly); a supported backend defaults ON per the recipe and a
    declared OFF is honored — a legitimate, stored override."""
    for key, supported in sim_enables.items():
        declared = dp.get(key) if key in explicit_keys else None
        if not supported:
            if declared is True:
                raise ValueError(
                    f"{key}=true contradicts the mode capability: the backend "
                    f"cannot run spiking_mode={spiking_mode!r} (schedule="
                    f"{dp.get('ttfs_cycle_schedule')!r}). Remove the explicit "
                    f"key to accept the capability derivation."
                )
            dp[key] = False
        else:
            dp[key] = declared is not False


def _pair_lif_exact_qat_retiming(
    dp: MutableMapping[str, Any], explicit: Set[str],
) -> None:
    """[lif_exact_qat_program §6.1(5)] ``lif_exact_qat`` arms per-hop re-timing:
    under exact-QAT the trained staircase IS the per-hop twin, and staircase
    deployment WITHOUT re-timing is the measured Goodhart hole (−2.5 pp) —
    an explicit contradiction fails loud."""
    if not bool(dp.get("lif_exact_qat", False)):
        return
    if not is_lif(str(dp.get("spiking_mode", "lif"))):
        raise ValueError(
            f"lif_exact_qat=true is only meaningful for spiking_mode='lif'; "
            f"got spiking_mode={dp.get('spiking_mode')!r}. Remove the key."
        )
    if str(dp.get("firing_mode", "Default")) != "Default":
        raise ValueError(
            "lif_exact_qat requires firing_mode='Default' (P-L5): Novena's zero "
            "reset breaks the Theorem-0 charge identity."
        )
    if "lif_per_hop_retiming" in explicit and not bool(
        dp.get("lif_per_hop_retiming", False)
    ):
        raise ValueError(
            "lif_exact_qat=true contradicts the explicit lif_per_hop_retiming="
            "false: the exact-QAT arm deploys as the per-hop RE-TIMED pair "
            "(raw-cascade staircase deployment measured −2.5 pp, "
            "lif_exact_qat_program.md §5). Drop the explicit key to accept the "
            "auto-pairing."
        )
    dp["lif_per_hop_retiming"] = True


def fold_conversion_recipe(
    dp: MutableMapping[str, Any],
    spiking_mode: str,
    explicit_keys: Optional[Iterable[str]] = None,
) -> None:
    """Fold the ConversionPolicy SSOT recipe for ``spiking_mode`` into ``dp``.

    Two-tier contract: INTERNAL recipe knobs (non-config keys) and the
    recipe-owned correctness keys are written authoritatively; REGISTRY
    knobs take the recipe value as their mode-aware default — an explicit
    document value wins (the registry's 'explicit value wins' contract).
    ``explicit_keys`` names the document-declared keys; ``None`` treats every
    present key as declared (dict-as-document callers).
    """
    explicit: Set[str] = set(dp) if explicit_keys is None else set(explicit_keys)
    recipe = ConversionPolicy.derive(spiking_mode, dp.get("ttfs_cycle_schedule"))
    _fold_sim_enables(dp, recipe.sim_enables, spiking_mode, explicit)
    dp["optimization_driver"] = recipe.driver
    for key, value in recipe.knobs.items():
        if (
            key in CONFIG_KEYS_SET
            and key in explicit
            and key not in RECIPE_OWNED_CORRECTNESS_KEYS
        ):
            continue
        dp[key] = value
    _pair_lif_exact_qat_retiming(dp, explicit)
