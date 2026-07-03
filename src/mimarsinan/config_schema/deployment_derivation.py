"""Derive deployment_parameters flags from pipeline_mode and spiking_mode (wizard parity)."""

from __future__ import annotations

from typing import Any, MutableMapping

from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_cycle_based,
    requires_ttfs_firing,
)
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy


def _fold_conversion_recipe(dp: MutableMapping[str, Any], spiking_mode: str) -> None:
    """Fold the ConversionPolicy SSOT recipe for ``spiking_mode`` into ``dp``.

    sim_enables, driver, and per-mode knobs are written authoritatively
    (Pure SSOT): any user value for them is overwritten by the recipe.
    """
    recipe = ConversionPolicy.derive(spiking_mode, dp.get("ttfs_cycle_schedule"))
    for key, value in recipe.sim_enables.items():
        dp[key] = value
    dp["optimization_driver"] = recipe.driver
    for key, value in recipe.knobs.items():
        dp[key] = value


def derive_deployment_parameters(dp: MutableMapping[str, Any]) -> None:
    """Apply the same rules as ``gui/static/js/wizard.js`` ``buildConfig()`` in-place."""
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    pipeline_mode = str(dp.get("pipeline_mode", ""))
    float_weights = pipeline_mode == "vanilla" or not bool(dp.get("weight_quantization", True))

    _fold_conversion_recipe(dp, spiking_mode)

    if float_weights:
        dp["pipeline_mode"] = "vanilla"
        dp["weight_quantization"] = False
        dp["activation_quantization"] = False
        return

    act_quant = forces_activation_quantization(spiking_mode) or is_cycle_based(spiking_mode)
    wt_quant = bool(dp.get("weight_quantization", True))
    dp["activation_quantization"] = act_quant
    dp["weight_quantization"] = wt_quant

    if act_quant or wt_quant:
        dp.setdefault("pipeline_mode", "phased")
    else:
        dp.setdefault("pipeline_mode", "vanilla")


def derive_pipeline_runtime_parameters(dp: MutableMapping[str, Any]) -> None:
    """Fill runtime spiking fields that minimal persisted configs may omit."""
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    if requires_ttfs_firing(spiking_mode):
        dp.setdefault("firing_mode", "TTFS")
        dp.setdefault("spike_generation_mode", "TTFS")
        dp.setdefault("thresholding_mode", "<=")
        for key in ("firing_mode", "spike_generation_mode"):
            if dp[key] != "TTFS":
                raise ValueError(
                    f"spiking_mode='{spiking_mode}' requires {key}='TTFS', "
                    f"got '{dp[key]}'"
                )
    else:
        dp.setdefault("firing_mode", "Default")
        dp.setdefault("spike_generation_mode", "Uniform")
        dp.setdefault("thresholding_mode", "<=")
        dp.setdefault("cycle_accurate_lif_forward", True)
