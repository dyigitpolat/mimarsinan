"""The value-domain (mvm) conversion recipe — the domain sibling of ConversionPolicy."""

from __future__ import annotations

from mimarsinan.tuning.orchestration.conversion_policy import (
    _WQ_RECIPE_KNOBS,
    OPTIMIZATION_DRIVER_FAST,
    ConversionRecipe,
)

MVM_SIM_ENABLES = {
    "enable_nevresim_simulation": False,
    "enable_sanafe_simulation": False,
    "enable_loihi_simulation": False,
    "enable_odin_fpga_simulation": False,
    "enable_odin_hacc_export": False,
}

_MVM_RATIONALE = (
    "Value-domain MVM cores execute y = Wx (+b) with real/quantized I/O and no "
    "on-chip nonlinearity, so the whole event-conversion ladder (clamp/AQ/"
    "LIF/TTFS adaptation, activation alignment) is structurally inapplicable. "
    "The recipe is the target-agnostic weight-quantization family alone; the "
    "spiking simulators are capability-off and the in-process value twin "
    "carries the deployed read."
)


def derive_mvm_recipe() -> ConversionRecipe:
    """The proven recipe for ``core_semantics='mvm'`` (no spiking_mode axis)."""
    return ConversionRecipe(
        driver=OPTIMIZATION_DRIVER_FAST,
        knobs=dict(_WQ_RECIPE_KNOBS),
        sim_enables=dict(MVM_SIM_ENABLES),
        special_case="mvm_value_domain",
        rationale=_MVM_RATIONALE,
    )
