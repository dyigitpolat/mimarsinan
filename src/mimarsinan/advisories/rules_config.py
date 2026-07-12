"""Config-time advisory rules: pure functions of the resolved DeploymentPlan."""

from __future__ import annotations

from typing import Any

from mimarsinan.advisories.advisory import (
    SEVERITY_INFO,
    SEVERITY_RISK,
    SEVERITY_UNSUPPORTED,
    Advisory,
    lossless_mandate_applies,
)
from mimarsinan.chip_simulation.spiking_semantics import (
    DEFAULT_FIRING_MODE,
    DEFAULT_THRESHOLDING_MODE,
    forces_activation_quantization,
    is_cycle_based,
    is_lif,
    is_novena_firing_mode,
)

ADV_CASC_UNSUPPORTED = "ADV-CASC-UNSUPPORTED"
ADV_NOVENA_CHARGE = "ADV-NOVENA-CHARGE"
ADV_STRICT_LT_LATTICE = "ADV-STRICT-LT-LATTICE"
ADV_ENVELOPE_GATE = "ADV-ENVELOPE-GATE"


def quantized_spiking_deployment(plan: Any) -> bool:
    """Deployment discretizes activations onto an S-level grid (cycle-based
    modes always; analytic modes only under activation quantization)."""
    return (
        is_cycle_based(plan.spiking_mode)
        or forces_activation_quantization(plan.spiking_mode)
        or bool(plan.activation_quantization)
    )


def rule_cascaded_ttfs_unsupported(plan: Any) -> list[Advisory]:
    if not plan.is_cascaded_ttfs:
        return []
    return [Advisory(
        id=ADV_CASC_UNSUPPORTED,
        severity=SEVERITY_UNSUPPORTED,
        title="Cascaded TTFS deployment is not fully supported",
        detail=(
            "The cascaded (pipelined) ttfs_cycle_based schedule fires each "
            "neuron ONCE at the first threshold crossing of a causally "
            "integrated ramp, before late/cancelling arrivals are in; "
            "premature firing dominates 79-94% of the cascaded rate error, "
            "GROWS with S (per-hop premature fractions 0.37-0.86), and every "
            "adversarially sound guard expressible in the existing parameter "
            "space costs a (1+Wpos) resolution factor per hop (premature-fire "
            "law; effective depth budget d_max ~= 0.56*sqrt(S) under the "
            "latched decode law). Measured: tier-0 casc_collapse cells read "
            "0.77-0.87 or collapse outright. This mode is not fully supported "
            "and may cause significant accuracy drops; a research program for "
            "closing this gap is planned. Memo: "
            "docs/research/findings/casc_first_crossing_transformation.md."
        ),
        tentative=False,
        mandate_violation=False,
        suggested_levers=(
            "ttfs_cycle_schedule=synchronized (complete-sum deferral is the lossless fix)",
            "spiking_mode=lif",
            "ttfs_gain_correction (partial, default-off)",
        ),
    )]


def rule_novena_charge(plan: Any) -> list[Advisory]:
    if not is_lif(plan.spiking_mode):
        return []
    firing_mode = plan.config.get("firing_mode", DEFAULT_FIRING_MODE)
    if not is_novena_firing_mode(firing_mode):
        return []
    return [Advisory(
        id=ADV_NOVENA_CHARGE,
        severity=SEVERITY_RISK,
        title="Novena zero-reset breaks LIF charge conservation",
        detail=(
            "The Novena firing mode zero-resets the membrane, discarding the "
            "residual m - theta at every fire and breaking the "
            "charge-conservation identity Q = theta*c + m (Theorem 0) under "
            "which the deployed LIF count equals the staircase of its exact "
            "integer charge. Measured cost on norm-free chain vehicles: "
            "-1.7pp at S=4 up to -10.4pp at S=8 (0.8640 vs 0.9680); the "
            "expectation-level affine repair recovers +8.1pp at S=8 but "
            "overfits the 5-level grid at S=4. Memo: "
            "docs/research/findings/lif_deployment_exactness.md (V7/C6)."
        ),
        tentative=True,
        mandate_violation=lossless_mandate_applies(plan),
        suggested_levers=(
            "firing_mode=Default (subtractive reset)",
            "lif_affine_fold (expectation repair; gate S>=8)",
        ),
    )]


def rule_strict_lt_lattice(plan: Any) -> list[Advisory]:
    thresholding = plan.config.get("thresholding_mode", DEFAULT_THRESHOLDING_MODE)
    if thresholding != "<":
        return []
    if not plan.weight_quantization:
        return []
    return [Advisory(
        id=ADV_STRICT_LT_LATTICE,
        severity=SEVERITY_INFO,
        title="Strict '<' comparator on an integer parameter lattice",
        detail=(
            "Under the strict '<' comparator a charge exactly equal to theta "
            "never fires, so a unit-weight identity relay is dead on any "
            "backend that snaps thresholds onto the weight-integer lattice — "
            "and weight quantization emits integer weights with threshold = "
            "scale, exactly that lattice. With float thresholds ties are "
            "measure-zero and no loss is measured ('<' == '<=' in every "
            "probed cell). A mapping-time guard exists: "
            "mapping/latency/depth_balancing.py assert_relays_alive raises "
            "DeadRelayError on silent relays, and relay weights carry a "
            "2^-20 identity margin. Memo: "
            "docs/research/findings/lif_deployment_exactness.md (V9)."
        ),
        tentative=False,
        mandate_violation=False,
        suggested_levers=(
            "thresholding_mode=<=",
        ),
    )]


CONFIG_RULES = (
    rule_cascaded_ttfs_unsupported,
    rule_novena_charge,
    rule_strict_lt_lattice,
)


def rule_envelope_gate(
    pretrain_acc: float, config: dict, acceptance_target=None
) -> list[Advisory]:
    """Post-pretraining rule: a pretrain read below the run's acceptance
    target makes the gate unreachable regardless of conversion quality."""
    target = acceptance_target
    if target is None:
        target = config.get("target_metric_override")
    if target is None:
        return []
    target = float(target)
    if float(pretrain_acc) >= target:
        return []
    return [Advisory(
        id=ADV_ENVELOPE_GATE,
        severity=SEVERITY_RISK,
        title="Pretrain accuracy below the run's acceptance target",
        detail=(
            f"The pretrain read ({float(pretrain_acc):.4f}) sits below the "
            f"run's acceptance target ({target:.4f}). Deployment steps are "
            "function-preserving transforms plus bounded recovery, so the "
            "deployed accuracy is capped by the float envelope the pretrain "
            "establishes and the gate is unreachable regardless of "
            "conversion quality. Measured: every mixer-class tier-0 cell "
            "pretrained at 0.9540-0.9703 — below the ~0.973 acceptance band "
            "— and every mode's final read landed on that envelope. This is "
            "a pretrain-side deficit, not a deployment loss: mandate_violation "
            "stays False because lossless deployment refinement cannot raise "
            "a float envelope. Memo: "
            "docs/research/findings/mixer_column_scale_pathology.md (§1)."
        ),
        tentative=True,
        mandate_violation=False,
        suggested_levers=(
            "training_epochs (more pretraining)",
            "pretrain recipe (lr / warmup / keep-best)",
            "model capacity",
            "preload_weights (a stronger envelope)",
        ),
    )]
