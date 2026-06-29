"""The ConversionPolicy SSOT — derive the proven recipe for a deployment mode.

:meth:`ConversionPolicy.derive` is the enacted source of truth: a deterministic
``(spiking_mode, schedule) → ConversionRecipe`` table collapsing the fix-wave
proven-best recipes (driver + knob set + capability-derived sim-enable set + a
special-case marker for the divergences). It is what
:func:`config_schema.deployment_derivation.derive_deployment_parameters` folds
authoritatively into every config — the resolved config IS the materialized recipe.

The four marked rows (``bn_freeze`` / ``full_quantile_decode`` /
``fast_only_never_controller`` / ``genuine_qat_fidelity``) are the documented
divergences from the generic flow; each carries a one-line ``rationale`` citing the
finding so it stays studyable. Every proven recipe rides the fast ladder — the
controller path is the one that collapses on the deep cascade, so the SSOT never
yields it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

OPTIMIZATION_DRIVER_CONTROLLER = "controller"
OPTIMIZATION_DRIVER_FAST = "fast"

# ── the ConversionPolicy SSOT table (the collapsed fix-wave proven recipes) ───
# Each deployment mode derives ONE proven recipe via ``ConversionPolicy.derive``:
# the fast-ladder driver, the per-mode knob set (now internal constants, NOT user
# config keys), the capability-derived sim-enable set, and a special-case marker +
# rationale for the rows that DIVERGE from the generic flow (kept studyable). The
# four marked rows below are exactly those documented divergences.
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
_SYNCHRONIZED_RECIPE_KNOBS = {
    "ttfs_blend_fast": True,
    "ttfs_blend_fast_stabilize_steps": 300,
    "ttfs_sync_genuine_qat": True,
    "fast_ladder_freeze_bn": True,
    "kd_ce_alpha": 0.5,
    "kd_temperature": 4.0,
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
    "Fidelity comes from genuine QAT, NOT from relaxing the parity gate; ~0.96 at 3x "
    "latency. nevresim has no synchronized-window backend (mnist_mixer_fix_wave)."
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

    ``driver`` is the optimization-driver arm (always the fast ladder — the SSOT
    never yields the controller). ``knobs`` is the per-mode recipe knob set (internal
    constants, no longer user config keys). ``sim_enables`` is the capability-derived
    backend-enable set (a backend is enabled iff it can run the mode).
    ``special_case`` / ``rationale`` mark and justify the rows that DIVERGE from the
    generic flow, so each divergence stays studyable.
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
        fast-ladder ``driver`` (the controller path collapses on the deep cascade —
        the SSOT never yields it), the per-mode ``knobs`` (now internal constants),
        the capability-derived ``sim_enables`` (a backend is disabled ONLY where it
        cannot run the mode), and the ``special_case`` marker + ``rationale`` for the
        rows that diverge from the generic flow.
        """
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )
        from mimarsinan.chip_simulation.spiking_semantics import (
            is_synchronized_ttfs,
            is_ttfs_cycle_based,
        )

        mode = str(spiking_mode or "lif")
        policy = policy_for_spiking_mode(mode, schedule)
        synchronized = is_synchronized_ttfs(mode, schedule)

        if mode in ("lif", "rate"):
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
                _SYNCHRONIZED_RECIPE_KNOBS, "genuine_qat_fidelity",
                _SYNCHRONIZED_RATIONALE,
            )
        elif is_ttfs_cycle_based(mode):
            knobs, special_case, rationale = (
                _CASCADED_RECIPE_KNOBS, "fast_only_never_controller",
                _CASCADED_RATIONALE,
            )
        else:  # analytical ``ttfs`` — the generic reference column (plain fast).
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
