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


TUNING_POLICY = TuningPolicy()
