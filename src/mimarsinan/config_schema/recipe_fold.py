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
    deployment WITHOUT re-timing is the measured Goodhart hole (−2.5 pp).
    A RECIPE-DEFAULT arm yields to conflicts (Novena capability, an explicit
    retiming opt-out) by downgrading the pair — the ``_fold_sim_enables``
    contract; an EXPLICIT contradiction fails loud."""
    if not bool(dp.get("lif_exact_qat", False)):
        return
    armed_explicitly = "lif_exact_qat" in explicit
    novena = str(dp.get("firing_mode", "Default")) != "Default"
    retiming_opt_out = "lif_per_hop_retiming" in explicit and not bool(
        dp.get("lif_per_hop_retiming", False)
    )
    if not armed_explicitly and (novena or retiming_opt_out):
        dp["lif_exact_qat"] = False
        return
    if not is_lif(str(dp.get("spiking_mode", "lif"))):
        raise ValueError(
            f"lif_exact_qat=true is only meaningful for spiking_mode='lif'; "
            f"got spiking_mode={dp.get('spiking_mode')!r}. Remove the key."
        )
    if novena:
        raise ValueError(
            "lif_exact_qat requires firing_mode='Default' (P-L5): Novena's zero "
            "reset breaks the Theorem-0 charge identity."
        )
    if retiming_opt_out:
        raise ValueError(
            "lif_exact_qat=true contradicts the explicit lif_per_hop_retiming="
            "false: the exact-QAT arm deploys as the per-hop RE-TIMED pair "
            "(raw-cascade staircase deployment measured −2.5 pp, "
            "lif_exact_qat_program.md §5). Drop the explicit key to accept the "
            "auto-pairing."
        )
    dp["lif_per_hop_retiming"] = True


def _pair_lif_exact_qat_kd(
    dp: MutableMapping[str, Any], explicit: Set[str],
) -> None:
    """[lif_exact_qat_program §8] ``lif_exact_qat_kd`` rides the exact-QAT arm:
    it distils the exact endpoint to the post-structural float teacher. A
    RECIPE-DEFAULT KD arm downgrades with the exact arm (which itself downgrades
    on Novena / opt-out); an EXPLICIT KD arm without an active exact arm is a
    contradiction and fails loud. Runs AFTER ``_pair_lif_exact_qat_retiming``
    so it reads the resolved ``lif_exact_qat``."""
    if not bool(dp.get("lif_exact_qat_kd", False)):
        return
    if bool(dp.get("lif_exact_qat", False)):
        return
    if "lif_exact_qat_kd" in explicit:
        raise ValueError(
            "lif_exact_qat_kd=true requires lif_exact_qat to resolve on; it "
            "resolved off (non-lif mode, Novena's broken charge identity, or an "
            "explicit retiming opt-out). Drop the KD key or fix the exact-QAT "
            "precondition."
        )
    dp["lif_exact_qat_kd"] = False


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
    _pair_lif_exact_qat_kd(dp, explicit)
