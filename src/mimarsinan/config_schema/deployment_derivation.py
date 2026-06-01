"""Derive deployment_parameters flags from pipeline_mode and spiking_mode (wizard parity)."""

from __future__ import annotations

from typing import Any, MutableMapping

from mimarsinan.chip_simulation.spiking_semantics import forces_activation_quantization


def derive_deployment_parameters(dp: MutableMapping[str, Any]) -> None:
    """Apply the same rules as ``gui/static/js/wizard.js`` ``buildConfig()`` in-place."""
    spiking_mode = str(dp.get("spiking_mode", "lif"))
    float_weights = not bool(dp.get("weight_quantization", True))

    if float_weights:
        dp["pipeline_mode"] = "vanilla"
        dp["weight_quantization"] = False
        dp["activation_quantization"] = False
        return

    act_quant = forces_activation_quantization(spiking_mode)
    wt_quant = bool(dp.get("weight_quantization", True))
    dp["activation_quantization"] = act_quant
    dp["weight_quantization"] = wt_quant

    if spiking_mode == "lif":
        dp["activation_quantization"] = False

    if act_quant or wt_quant:
        dp.setdefault("pipeline_mode", "phased")
    else:
        dp.setdefault("pipeline_mode", "vanilla")
