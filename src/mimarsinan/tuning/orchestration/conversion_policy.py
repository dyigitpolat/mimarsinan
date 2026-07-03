"""The ConversionPolicy SSOT — derive the proven recipe for a deployment mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

OPTIMIZATION_DRIVER_CONTROLLER = "controller"
OPTIMIZATION_DRIVER_FAST = "fast"

_LIF_RECIPE_KNOBS = {
    "lif_blend_fast": True,
    "lif_blend_fast_stabilize_steps": 600,
    "cycle_accurate_lif_forward": True,
    "fast_ladder_freeze_bn": True,
    "kd_ce_alpha": 0.5,
    "kd_temperature": 4.0,
}
_TTFS_QUANTIZED_RECIPE_KNOBS = {
    "activation_scale_quantile": 1.0,
    "manager_rate_fast_rates": [0.25, 0.5, 0.75, 1.0],
    "manager_rate_fast_steps_per_rate": 120,
}
_CASCADED_RECIPE_KNOBS = {
    "ttfs_genuine_blend_ramp": True,
    "ttfs_genuine_blend_fast": True,
    "ttfs_blend_fast_stabilize_steps": 300,
    "tuning_full_transform_probe": True,
}

_LIF_RATIONALE = (
    "BN-freeze makes the QAT train-forward bit-exact to the deployed eval-forward; "
    "the faithful LIF levers cap ~0.95-0.96 (mnist_mixer_fix_wave / per_channel_theta)."
)
_TTFS_QUANTIZED_RATIONALE = (
    "Full-quantile (q=1.0) per-perceptron decode helps the quantized timing path; it "
    "is harmful for LIF, whose decode scale is per-channel (per_channel_theta)."
)
_CASCADED_RATIONALE = (
    "The controller collapses on the deep genuine cascade (rate stalls then drops to "
    "chance); the fast blend ladder is the ec=0 survivor (0.9396 @ parity 0.9961) "
    "(per_channel_theta_deployment_fidelity)."
)
_SYNCHRONIZED_RATIONALE = (
    "synchronized IS ttfs_quantized at deploy: sync-deploy = ttfs_quantized-deploy + the "
    "free segment-input single-spike grid-snap (decision-level exact). So it TRAINS the "
    "well-conditioned ttfs_quantized floor recovery (floor+half-step-bias QAT + full "
    "recovery) rather than the harder ceil-staircase genuine QAT, and DEPLOYS the "
    "mode-derived single-spike ceil kernel + grid-snap (unchanged: SANA-FE parity 1.0, "
    "bit-exact). Its per-neuron NF↔SCM parity therefore uses the honest ttfs_quantized "
    "tolerance — excluded from the bit-exact per-neuron gate. nevresim has no "
    "synchronized-window backend (mnist_mixer_fix_wave / synchronized_floor_collapse)."
)

__all__ = [
    "OPTIMIZATION_DRIVER_CONTROLLER",
    "OPTIMIZATION_DRIVER_FAST",
    "ConversionRecipe",
    "ConversionPolicy",
]


@dataclass(frozen=True)
class ConversionRecipe:
    """The proven recipe derived for one deployment mode by ``ConversionPolicy.derive``.

    ``driver`` is always the fast ladder; ``knobs`` is the per-mode recipe knob set;
    ``sim_enables`` is the capability-derived backend-enable set; ``special_case`` /
    ``rationale`` mark and justify the rows that diverge from the generic flow.
    """

    driver: str
    knobs: Mapping[str, Any] = field(default_factory=dict)
    sim_enables: Mapping[str, bool] = field(default_factory=dict)
    special_case: Optional[str] = None
    rationale: str = ""


class ConversionPolicy:
    """The deterministic ``(spiking_mode, schedule) → ConversionRecipe`` SSOT table."""

    @classmethod
    def derive(cls, spiking_mode: str, schedule: Any = None) -> ConversionRecipe:
        """Derive the proven recipe for a deployment mode — the SSOT mode→recipe table.

        Maps ``(spiking_mode, schedule)`` to its empirically-proven recipe: the
        fast-ladder ``driver``, the per-mode ``knobs``, the capability-derived
        ``sim_enables``, and the ``special_case`` marker + ``rationale``.
        """
        # Lazy: chip_simulation has a fragile import cycle; a top-level import breaks
        # when this module loads before chip_simulation finishes initializing.
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )
        from mimarsinan.chip_simulation.spiking_semantics import (
            is_synchronized_ttfs,
            is_ttfs_cycle_based,
            require_known_spiking_mode,
        )

        mode = require_known_spiking_mode(spiking_mode)
        policy = policy_for_spiking_mode(mode, schedule)
        synchronized = is_synchronized_ttfs(mode, schedule)

        if mode == "lif":
            knobs, special_case, rationale = (
                _LIF_RECIPE_KNOBS, "bn_freeze", _LIF_RATIONALE,
            )
        elif mode == "ttfs_quantized":
            knobs, special_case, rationale = (
                _TTFS_QUANTIZED_RECIPE_KNOBS, "full_quantile_decode",
                _TTFS_QUANTIZED_RATIONALE,
            )
        elif is_ttfs_cycle_based(mode) and synchronized:
            knobs, special_case, rationale = (
                _TTFS_QUANTIZED_RECIPE_KNOBS, "synchronized_floor_collapse",
                _SYNCHRONIZED_RATIONALE,
            )
        elif is_ttfs_cycle_based(mode):
            knobs, special_case, rationale = (
                _CASCADED_RECIPE_KNOBS, "fast_only_never_controller",
                _CASCADED_RATIONALE,
            )
        else:
            knobs, special_case, rationale = {}, None, ""

        sim_enables = {
            "enable_nevresim_simulation": (
                policy.supports_backend("nevresim") and not synchronized
            ),
            "enable_sanafe_simulation": policy.supports_backend("sanafe"),
            "enable_loihi_simulation": policy.supports_backend("loihi"),
        }

        return ConversionRecipe(
            driver=OPTIMIZATION_DRIVER_FAST,
            knobs=dict(knobs),
            sim_enables=sim_enables,
            special_case=special_case,
            rationale=rationale,
        )
