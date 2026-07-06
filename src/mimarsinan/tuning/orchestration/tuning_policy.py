"""The TuningPolicy SSOT — the frozen tuning-loop behavior knobs (never user-configured)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TuningPolicy", "TUNING_POLICY"]


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


TUNING_POLICY = TuningPolicy()
