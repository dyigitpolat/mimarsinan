"""Smooth adaptation via adaptive step sizing with rollback.

Given transformation T, evaluation A, and recovery R, drives rate from 0 to 1
using an adaptive step that grows on successful commits and shrinks on rollback:

    for each cycle:
        propose rate = t + step
        result = adaptation_fn(rate)  -- includes T, LR search, R, and rollback
        if committed: grow step
        if rolled back: shrink step

The adaptation function may return the *committed* rate (float).  When the
returned value is less than the proposed rate, the loop treats the cycle as
a rollback: ``t`` is reset to the committed rate and the step is halved.
This prevents ``t`` from diverging from the actual model state and guarantees
termination (step shrinks toward min_step).
"""

from __future__ import annotations

from typing import Callable, List, Optional


class SmartSmoothAdaptation:
    """Adaptive step-size loop for smooth rate-based adaptation."""

    def __init__(
        self,
        adaptation_fn: Callable,
        interpolators: List,
        get_target: Callable[[], float],
        min_step: float,
        before_cycle: Optional[Callable[[], None]] = None,
    ):
        self.adaptation_fn = adaptation_fn
        self.interpolators = interpolators
        self.get_target = get_target
        self.min_step = float(min_step)
        self.before_cycle = before_cycle

    def adapt_smoothly(self, max_cycles: Optional[int] = None) -> None:
        t = 0.0
        cycles = 0
        step = 1.0  # start aggressive; fast-fail in _adaptation handles bad rates cheaply

        while t < 1.0 - 1e-6 and (not max_cycles or cycles < max_cycles):
            if step < self.min_step:
                break
            if self.before_cycle is not None:
                self.before_cycle()

            t_proposed = min(t + step, 1.0)
            interpolated = [i(t_proposed) for i in self.interpolators]
            result = self.adaptation_fn(*interpolated)

            if result is not None and float(result) < t_proposed - 1e-9:
                # Rollback: adaptation_fn returned committed rate < proposed
                t = float(result)
                step /= 2.0
            else:
                # Commit
                t = t_proposed
                remaining = 1.0 - t
                step = min(step * 1.5, remaining) if remaining > 1e-6 else step
            cycles += 1
