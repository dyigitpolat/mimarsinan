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


class CouplingGuard:
    """[C1'] decoupling stop for one armed endpoint leg (one-shot).

    The leg trains the SURROGATE the trainer validates, but the funded
    objective is the DEPLOYED read: at every ``cadence``-th check the guard
    reads the deployed currency and fires when the surrogate's best gained
    >= SE over the window while the deployed best gained < SE — further
    funding is buying surrogate quality the deployed composition cannot
    express (measured: a ViT WQ leg burned 3.5 h at flat deployed 0.396 while
    its surrogate climbed 0.43->0.61). Coupled legs never fire: their deployed
    read tracks the surrogate within SE, so the guard is inert.
    """

    def __init__(self, *, deployed_read, entry_deployed, accuracy_se, cadence):
        self._read = deployed_read
        self._se = float(accuracy_se)
        self._cadence = max(1, int(cadence))
        self._checks = 0
        self._anchor_deployed = float(entry_deployed)
        self._peak_deployed = float(entry_deployed)
        self._anchor_surrogate: Optional[float] = None
        self.fired = False

    def __call__(self, step, acc, best_acc, entry_acc) -> bool:
        if self.fired:
            return True
        self._checks += 1
        if self._anchor_surrogate is None:
            self._anchor_surrogate = float(entry_acc)
        if self._checks % self._cadence:
            return False
        self._peak_deployed = max(self._peak_deployed, float(self._read()))
        # Cumulative anchors: coupled progress (deployed +SE since the anchor)
        # re-anchors BOTH and stays silent; slow surrogate creep accumulates
        # across windows instead of being reset away, so a decoupled leg fires
        # once the surrogate runs >= SE ahead of a stalled deployed read.
        if self._peak_deployed - self._anchor_deployed >= self._se:
            self._anchor_deployed = self._peak_deployed
            self._anchor_surrogate = float(best_acc)
            return False
        self.fired = float(best_acc) - self._anchor_surrogate >= self._se
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
