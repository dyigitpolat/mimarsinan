"""[C3] divergence guard + LR-backoff rescue plan for the armed endpoint floor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

TAKEOFF_CHECKS = 5
"""Checks the best read gets to beat entry+SE before the leg is declared dead."""

CRATER_CHECKS = 3
"""Consecutive reads below the pipeline hard floor that declare a crater."""


class DivergenceGuard:
    """Per-check dead-run predicate for one armed endpoint leg (one-shot).

    Fires when the best read never beat entry+SE after ``TAKEOFF_CHECKS``
    checks, OR when the current read sat below ``hard_floor`` for
    ``CRATER_CHECKS`` consecutive checks; ``hard_floor=None`` disables only
    the crater disjunct.
    """

    def __init__(self, *, accuracy_se: float, hard_floor: Optional[float] = None):
        self._se = float(accuracy_se)
        self._hard_floor = None if hard_floor is None else float(hard_floor)
        self._checks = 0
        self._crater_streak = 0
        self.fired = False

    def __call__(self, step, acc, best_acc, entry_acc) -> bool:
        if self.fired:
            return True
        self._checks += 1
        if self._hard_floor is not None:
            self._crater_streak = (
                self._crater_streak + 1 if acc < self._hard_floor else 0
            )
        never_took_off = (
            self._checks >= TAKEOFF_CHECKS
            and best_acc <= entry_acc + self._se
        )
        self.fired = never_took_off or self._crater_streak >= CRATER_CHECKS
        return self.fired


@dataclass(frozen=True)
class RescuePlan:
    """The restart leg: backed-off peak LR, warmup ramp, bounded train steps."""

    lr: float
    warmup_steps: int
    train_steps: int


def rescue_plan(remaining_budget, lr) -> Optional[RescuePlan]:
    """[C3] restart geometry over the remaining funded budget; ``None`` when
    fewer than 2 steps remain (warmup + at least one decay step must fit).
    warmup + train_steps never exceed the remainder (budgets stay ceilings)."""
    remaining = int(remaining_budget)
    if remaining < 2:
        return None
    warmup = max(1, math.ceil(
        TUNING_POLICY.endpoint_floor_rescue_warmup_fraction * remaining
    ))
    return RescuePlan(
        lr=float(lr) * float(TUNING_POLICY.endpoint_floor_rescue_lr_factor),
        warmup_steps=warmup,
        train_steps=remaining - warmup,
    )
