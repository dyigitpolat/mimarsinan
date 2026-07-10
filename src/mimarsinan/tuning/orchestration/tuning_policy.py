"""The TuningPolicy SSOT — the frozen tuning-loop behavior knobs (never user-configured)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mimarsinan.common.workload_profile import ResolvedWorkloadProfile

__all__ = [
    "FAST_LADDER_STEPS_PER_RATE",
    "TuningPolicy",
    "TUNING_POLICY",
    "ConvergenceStopGeometry",
    "effective_prefix_stage_lr",
    "effective_endpoint_floor_lr",
    "endpoint_convergence_geometry",
]


@dataclass(frozen=True)
class TuningPolicy:
    """The proven tuning-loop behavior: checkpointing, recovery, rollback,
    stabilization, and commit-gate settings. Field values carry the effective
    defaults of the former (never-set) ``tuning_*`` config reads."""

    checkpoint_scope: str = "full"
    checkpoint_location: str = "device"
    refind_lr_on_miss: bool = False
    recovery_lr_plateau: bool = False
    recovery_lr_plateau_factor: float = 0.3
    recovery_lr_plateau_reductions: int = 2
    recovery_check_divisor: int = 1
    rollback_ratchet: bool = False
    rollback_cumulative_bound: float = 0.05
    tight_plateau: bool = False
    keepbest_certified: bool = False
    stabilization_bounded: bool = False
    stabilization_ratio: float = 0.5
    use_paired_sensor: bool = False
    k_commit: float = 2.0
    global_budget: float = 0.0
    # DFQ keep-best: iterations without a new best deployed-probe read before
    # the calibration loop stops early (W-CAL-3).
    dfq_keepbest_patience: int = 5
    # P4 per-stage re-affine: short keep-best DFQ measured through the k-hybrid
    # (T4 arm B absorbed every boundary's damage at 4 iters/stage).
    prefix_stage_dfq_iters: int = 4
    # P4 stage recovery keep-best cadence (arm B probed every 25 steps).
    prefix_stage_keepbest_interval: int = 25
    # P4 stage LR ceiling (arm B trained at 1e-3; hotter pipeline LRs measured
    # destructive through the genuine k-hybrid on the first x3b wave).
    prefix_stage_lr: float = 1e-3
    # P4 stage per-step gradient clip through the genuine k-hybrid (arm B).
    prefix_stage_grad_clip_norm: float = 1.0
    # Hop-frontier rung budget (FAST respec 2026-07-08): the outcome is
    # measured budget-INSENSITIVE (stage budget x3: no effect, 0.8904 inside
    # the 0.88-0.90 band) while every rung step pays the O(S x depth) genuine
    # segment forward; 40 steps/rung holds the band at ~1/3 the FT wall.
    hop_stage_steps_per_rate: int = 40
    # [5u] floor-lifted endpoint LR ceiling: at the pipeline LR the first ~1.6k
    # steps dip below entry, so a floor-chasing stage trains the probe-validated
    # arm (lr 2e-3, cosine over the full funded budget).
    endpoint_floor_lr: float = 2e-3
    # [5u + reproducibility] the floor's RUN-total funding is a STEP budget
    # (the endpoint_steps ledger), never wall seconds: identical configs train
    # identical step counts on any hardware (same config + same seed => same
    # step trajectory, modulo GPU nondeterminism). 16,000 is the validated
    # full floor budget (t01_23: full 16k => the honest 0.97 fbu ceiling);
    # wall time is a pure measurement, judged per hardware context at harvest.
    endpoint_floor_steps: int = 16000
    # [C1] armed-endpoint convergence stop: min-cover = max(absolute lr-dip
    # cover — measured ~1.6k on the t0_21 dip — fraction of the funded budget)
    # and keep-best patience scaled to the budget instead of disarmed, so the
    # funded budget is a true CEILING, never a mandatory burn.
    endpoint_floor_min_cover_steps: int = 2000
    endpoint_floor_patience_fraction: float = 0.25
    # [C3] divergence guard + LR-backoff rescue on the armed floor (default
    # off until the Phase-3 measured graduation): a fired dead-run predicate
    # restores the live keep-best, rebuilds the optimizer, and restarts the
    # remaining funded budget once at lr*factor with a warmup ramp.
    endpoint_floor_divergence_rescue: bool = False
    endpoint_floor_rescue_lr_factor: float = 0.3
    endpoint_floor_rescue_warmup_fraction: float = 0.02


# The fast ladder's generic per-rung training budget (rate-invariant units;
# value proven on the tier-0/0.1 MNIST corpus — the former five scattered
# `120` CFG fallbacks). Workloads override via the *_fast_steps_per_rate keys.
FAST_LADDER_STEPS_PER_RATE = 120

TUNING_POLICY = TuningPolicy()


def effective_prefix_stage_lr(config) -> float:
    """P4 stage LR ceiling: explicit/model-registered workload override, else
    the frozen policy value (a trainability fact proven on the tier-0 corpus)."""
    override = ResolvedWorkloadProfile.from_config(config).prefix_stage_lr
    return TUNING_POLICY.prefix_stage_lr if override is None else float(override)


def effective_endpoint_floor_lr(config) -> float:
    """Floor-chasing endpoint LR: explicit/model-registered workload override,
    else the frozen probe-validated policy value."""
    override = ResolvedWorkloadProfile.from_config(config).endpoint_floor_lr
    return TUNING_POLICY.endpoint_floor_lr if override is None else float(override)


@dataclass(frozen=True)
class ConvergenceStopGeometry:
    """[C1] keep-best geometry for one funded endpoint/stabilize training leg."""

    min_steps: int
    patience: int


def endpoint_convergence_geometry(budget, check_interval) -> ConvergenceStopGeometry:
    """[C1] convergence-stop geometry for a funded leg: min-cover =
    max(absolute lr-dip cover, fraction of budget) — the absolute term wins at
    small residual budgets, keeping them full burns — and patience scaled to
    the budget instead of disarmed, so the budget stays a true ceiling."""
    policy = TUNING_POLICY
    budget = max(0, int(budget))
    interval = max(1, int(check_interval))
    fraction = float(policy.endpoint_floor_patience_fraction)
    return ConvergenceStopGeometry(
        min_steps=max(
            int(policy.endpoint_floor_min_cover_steps),
            math.ceil(fraction * budget),
        ),
        patience=max(1, math.ceil(fraction * budget / interval)),
    )
