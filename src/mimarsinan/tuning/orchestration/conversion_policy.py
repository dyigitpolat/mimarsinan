"""The ConversionPolicy SSOT — derive the proven recipe for a deployment mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from mimarsinan.tuning.orchestration.tuning_policy import FAST_LADDER_STEPS_PER_RATE

OPTIMIZATION_DRIVER_CONTROLLER = "controller"
OPTIMIZATION_DRIVER_FAST = "fast"

# WQ demotion (mode-independent, theory 5g-v): NAPQ rides the same gated fixed
# ladder as everything else with projection-only rungs; its bounded recovery is
# the P1'' endpoint stage anchored at the D-hat high-water mark.
_WQ_RECIPE_KNOBS = {
    "wq_fast_rates": [0.5, 1.0],
    "wq_fast_steps_per_rate": 0,
    "wq_endpoint_recovery_steps": 600,
}

# [5u] endpoint target floor for bit-parity-lossless modes: every controller
# target anchors at a past deployed read, so for a lossless mode preservation
# is stagnation at the float envelope; the P1'' endpoint may instead chase the
# internal acceptance target (0.98 true ⇒ a reliable >=0.97 read at N=100).
# The budget is the measured ~180 s artifact-wall headroom (probe-validated:
# 16k steps at lr 2e-3 / cosine-over-budget lifted 0.9663 -> 0.9761 keep-best).
_BIT_PARITY_LOSSLESS_RECIPE_KNOBS = {
    "endpoint_target_floor": 0.98,
    "wq_endpoint_recovery_steps": 16000,
}

# [5u generalized] the endpoint target floor extends past the bit-parity family
# to every well-conditioned near-lossless conversion (lif/sync/cascaded): a
# gate-accepted, crater-free conversion whose deployed forward sits at the
# pretrain envelope (below the acceptance bar, not below by a conversion crater)
# may chase the bar. Scoped to the FINAL weight-quant endpoint alone
# (wq_endpoint_target_floor, never the intermediate mode endpoint via
# endpoint_target_floor) so the wall funds ONE lift; keep-best + the fp32 entry
# guard keep it monotone-safe and A1-A6 untouched (the endpoint is not a ramp).
_WELL_CONDITIONED_ENDPOINT_FLOOR_KNOBS = {
    "wq_endpoint_target_floor": 0.98,
    "wq_endpoint_recovery_steps": 16000,
}

_LIF_RECIPE_KNOBS = {
    "lif_blend_fast": True,
    "lif_tanneal": True,
    # P1'' budget, convergence-grounded (FAST respec 2026-07-08): healthy
    # endpoint reaches measure 250-930 steps; 600 = the shared endpoint cap
    # (the old 1560 was ladder-accounting, not convergence, and each armed
    # step pays the O(S) cycle-accurate LIF forward).
    "endpoint_recovery_steps": 600,
    "cycle_accurate_lif_forward": True,
    "fast_ladder_freeze_bn": True,
    "kd_ce_alpha": 0.5,
    "kd_temperature": 4.0,
    # [5v B3] the LIF analogue of the TTFS half-step bake (never existed for
    # LIF): +theta/(2T) folded as a TRAINABLE entry bias before the weight-quant
    # QAT turns the floor rate grid into nearest over the window and head-starts
    # every hop's first fire; the QAT owns and reconciles it so the float NF and
    # the quantized deployed sim stay bit-exact (mapping-time injection broke
    # that parity identity: t0_01 0.9336, t0_05 0.9883).
    "lif_half_step_bias": True,
    # [lif_deployment_exactness §7] the exact/statistical correction ladder on
    # top of the half-step: C2 membrane-augmented readout (Q = theta*c + m
    # recovers the sign-carrying unquantized logits at final-only output
    # cores, +2.6pp at S=4), C4 per-channel FULL affine fold (dead-zone bias
    # absorption, calibration-only; bias-only folds refuted at -4.2pp; Novena
    # arm gated S>=8 in the step), C5 depth-balancing relays (exact V6 join
    # fix, no-op on gap-free graphs). C3 per-hop re-timing stays a mapping
    # choice (mixers get it free at ComputeOp boundaries).
    "lif_membrane_readout": True,
    "lif_affine_fold": True,
    "lif_depth_balancing_relays": True,
    **_WELL_CONDITIONED_ENDPOINT_FLOOR_KNOBS,
}
_TTFS_QUANTIZED_RECIPE_KNOBS = {
    "activation_scale_quantile": 1.0,
    "manager_rate_fast_rates": [0.25, 0.5, 0.75, 1.0],
    "manager_rate_fast_steps_per_rate": FAST_LADDER_STEPS_PER_RATE,
    # [C4] granted 2026-07: the ttfsq proxy→deployed transfer is measured
    # sub-SE (t0_11 +0.0007 / t0_14 −0.0014 / t01_06 −0.0010 vs SE 0.0092),
    # so the floor-funded proxy climb survives to the deployed read; the C1
    # convergence stop bounds the funded burn.
    **_WELL_CONDITIONED_ENDPOINT_FLOOR_KNOBS,
}
_SYNCHRONIZED_RECIPE_KNOBS = {
    **_TTFS_QUANTIZED_RECIPE_KNOBS,
    "sync_exact_qat": True,
    # P1'' budget at the AQ endpoint (the sync conversion endpoint), replacing
    # the open-ended AQ stabilize; stalls add their freed ladder steps.
    "endpoint_recovery_steps": 600,
    # [5v B1] the sync crater's two measured scalar levers (t0_21 AQ entry
    # 0.10 -> 0.85 combined): the full-quantile theta deflates per-hop where
    # the A6 gauge reads starvation, and the exact-ceil endpoint is entered
    # through the +0.5/S half-step folded as trainable entry bias.
    "starvation_aware_scale_quantile": True,
    "sync_entry_half_step": True,
    # [5v B1] the P4 frontier below segments: when the A6 gauge fails on a
    # chain past the proven-recovery depth, the AQ ladder walks one hop level
    # per rung (per-hop keep-best re-affine) instead of a monolithic install.
    "sync_hop_staged_install": True,
    # [M2] the sync value-domain forward craters from the same bias-set shared
    # grid (wq_cascade_crater_repair.md §4.4 value-forward control), so the
    # whole ttfs_cycle_based family carries the two-scale projection.
    "wq_two_scale_projection": True,
    **_WELL_CONDITIONED_ENDPOINT_FLOOR_KNOBS,
}
_CASCADED_RECIPE_KNOBS = {
    "ttfs_genuine_blend_ramp": True,
    "ttfs_genuine_blend_fast": True,
    # T4/P4: multi-segment vehicles walk the converted-prefix frontier instead
    # of the output blend (single-segment vehicles keep the blend ramp).
    "ttfs_prefix_ramp": True,
    # [5v B2] the frontier below segments: a single-segment vehicle whose
    # chain is past the proven-recovery depth walks cascade hops (per-hop
    # keep-best re-affine + stage training) instead of the blend fallback —
    # the t0_16 crater was exactly that fallback. gamma(S) stays off (refuted).
    "ttfs_hop_prefix_ramp": True,
    # P1'' budget: W3 reinvests reclaimed eval wall here — the only endpoint
    # bound that BINDS while still improving (X3: t0_20/t0_16 FT 300/300,
    # t0_18 294/300, all climbing at cutoff); patience stops saturated cells.
    "endpoint_recovery_steps": 600,
    "tuning_full_transform_probe": True,
    # [M2] two-scale WQ projection: the shared max(|w|,|b|) grid is set BY THE
    # BIAS on the fc perceptrons (58-95% of weights round to exactly zero) and
    # craters the first-crossing forward; the weight grid from max|w| alone
    # with the bias on its own integer-ratio-snapped grid recovers float
    # exactly (wq_cascade_crater_repair.md §3-4).
    "wq_two_scale_projection": True,
    **_WELL_CONDITIONED_ENDPOINT_FLOOR_KNOBS,
}

_LIF_RATIONALE = (
    "BN-freeze makes the QAT train-forward bit-exact to the deployed eval-forward; "
    "the value-blend ramp measures near-zero transfer alignment (rho 0.014-0.167, X1) "
    "while the realizable T-anneal family is >= it everywhere and has no hidden debt "
    "at S=32 (X2b), so the LIF ramp anneals T on genuine LIF members; the Clamp/AQ "
    "ladders are behaviorally inert under the spiking node (X1) and train 0 steps "
    "(mnist_mixer_fix_wave / per_channel_theta / mbh_x2b_lif_tanneal_readout)."
)
_TTFS_QUANTIZED_RATIONALE = (
    "Full-quantile (q=1.0) per-perceptron decode helps the quantized timing path; it "
    "is harmful for LIF, whose decode scale is per-channel (per_channel_theta). Green "
    "family: stays on the floor+half-step proxy, and the final-WQ endpoint carries "
    "the well-conditioned floor (proxy->deployed transfer measured sub-SE: t0_11 "
    "+0.0007 / t0_14 -0.0014 / t01_06 -0.0010 vs SE 0.0092, so the funded climb "
    "survives to the deployed read); the exact-kernel endpoint promotion "
    "is an X4 follow-up (mbh_t6_sync_exact_kernel)."
)
_CASCADED_RATIONALE = (
    "The controller collapses on the deep genuine cascade (rate stalls then drops to "
    "chance); the fast blend ladder is the ec=0 survivor (0.9396 @ parity 0.9961) "
    "(per_channel_theta_deployment_fidelity). Multi-segment vehicles walk the "
    "converted-prefix frontier: boundary gradients are severed, so only the P4 "
    "frontier trains every layer once, at the moment its conversion damage is live "
    "(T4 shootout: 0.9629 vs 0.9277 post-FT at equal budget, mbh_t4_depth_law)."
)
_BIT_PARITY_LOSSLESS_RATIONALE = (
    "Analytical ttfs deploys bit-exactly (parity 0.0000% per-neuron mismatch), so "
    "sup(controller targets) <= float envelope + noise and the endpoint patience-stops "
    "at exactly patience x check_interval with the budget unspent (the stagnation "
    "theorem, mbh_analytical_ttfs_stagnation). The floor = the internal acceptance "
    "target lets the endpoint spend the measured wall headroom; keep-best and the "
    "entry guard keep the stage non-destructive, and reached=False stays legal."
)
_SYNCHRONIZED_RATIONALE = (
    "synchronized IS ttfs_quantized at deploy: sync-deploy = ttfs_quantized-deploy + the "
    "free segment-input single-spike grid-snap. It rides the ttfs_quantized ladder shape "
    "but TRAINS the exact deployed composition — the ceil TTFS kernel under STE + the "
    "per-stage entry grid snap — as the QAT endpoint (T6: parity 0.9180/0.8633-abort -> "
    "1.0000/256 on t0_22/t0_21), and the mapping-time +0.5/Tq bias compensation is "
    "skipped for models so trained (marker-asserted). Its per-neuron NF↔SCM parity "
    "stays excluded from the bit-exact per-neuron gate; nevresim has no "
    "synchronized-window backend (mbh_t6_sync_exact_kernel)."
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
            is_bit_parity_lossless_conversion,
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
                _SYNCHRONIZED_RECIPE_KNOBS, "sync_exact_endpoint",
                _SYNCHRONIZED_RATIONALE,
            )
        elif is_ttfs_cycle_based(mode):
            knobs, special_case, rationale = (
                _CASCADED_RECIPE_KNOBS, "fast_only_never_controller",
                _CASCADED_RATIONALE,
            )
        else:
            knobs, special_case, rationale = {}, None, ""
        knobs = {**_WQ_RECIPE_KNOBS, **knobs}
        if is_bit_parity_lossless_conversion(mode):
            # [5u] the floor rides EXACTLY the analytic BIT_PARITY family; the
            # predicate (not a mode literal) is the audit.
            knobs = {**knobs, **_BIT_PARITY_LOSSLESS_RECIPE_KNOBS}
            special_case = special_case or "endpoint_target_floor"
            rationale = rationale or _BIT_PARITY_LOSSLESS_RATIONALE

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
