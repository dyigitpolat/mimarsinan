"""Derive deployment_parameters flags from pipeline_mode and spiking_mode (wizard parity)."""

from __future__ import annotations

from typing import Any, MutableMapping

from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_cycle_based,
    is_synchronized_ttfs,
    requires_ttfs_firing,
)


def derive_deployment_parameters(dp: MutableMapping[str, Any]) -> None:
    """Apply the same rules as ``gui/static/js/wizard.js`` ``buildConfig()`` in-place."""
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    pipeline_mode = str(dp.get("pipeline_mode", ""))
    float_weights = pipeline_mode == "vanilla" or not bool(dp.get("weight_quantization", True))

    if float_weights:
        dp["pipeline_mode"] = "vanilla"
        dp["weight_quantization"] = False
        dp["activation_quantization"] = False
        return

    # Cycle-accurate LIF/TTFS tuning is preconditioned by shift/AQ before the
    # tuning ramp. Float-weight deployments keep activation quantization off.
    act_quant = forces_activation_quantization(spiking_mode) or is_cycle_based(spiking_mode)
    wt_quant = bool(dp.get("weight_quantization", True))
    dp["activation_quantization"] = act_quant
    dp["weight_quantization"] = wt_quant

    # nevresim has no genuine synchronized-window backend yet, so it is disabled
    # only for the synchronized schedule; cascaded runs genuinely on nevresim.
    if is_synchronized_ttfs(spiking_mode, dp.get("ttfs_cycle_schedule")):
        dp["enable_nevresim_simulation"] = False

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
    else:
        dp.setdefault("firing_mode", "Default")
        dp.setdefault("spike_generation_mode", "Uniform")
        dp.setdefault("thresholding_mode", "<=")
        dp.setdefault("cycle_accurate_lif_forward", True)
