"""Smooth adaptation via greedy step-size bisection.

Given transformation T, evaluation A, and recovery R, finds the schedule
r_1, r_2, ... with sum(r_i) = 1 by probing the largest tolerable step at
each cycle:

    for each cycle:
        find largest r such that A(T(r, M)) >= (1 - tol) * target
        apply T(r, M)  then  R(M, lr, budget)

The adaptation function may return the *committed* rate (float).  When the
returned value is less than the proposed rate, the loop treats the cycle as
a rollback: ``t`` is reset to the committed rate and the maximum allowed
step for the next cycle is halved.  This prevents ``t`` from diverging from
the actual model state and guarantees termination (max_step shrinks toward
min_step).
"""

from __future__ import annotations

from typing import Callable, List, Optional


class SmartSmoothAdaptation:
    """Greedy bisection loop for smooth rate-based adaptation."""

    def __init__(
        self,
        adaptation_fn: Callable,
        clone_state: Callable[[], object],
        restore_state: Callable[[object], None],
        evaluate_fn: Callable,
        interpolators: List,
        get_target: Callable[[], float],
        tolerance: float,
        min_step: float,
        before_cycle: Optional[Callable[[], None]] = None,
    ):
        self.adaptation_fn = adaptation_fn
        self.clone_state = clone_state
        self.restore_state = restore_state
        self.evaluate_fn = evaluate_fn
        self.interpolators = interpolators
        self.get_target = get_target
        self.tolerance = float(tolerance)
        self.min_step = float(min_step)
        self.before_cycle = before_cycle

    def _adjust_minimum_step(self, step_size: float, t: float) -> None:
        halfway = (1 - t) / 2
        if step_size < self.min_step and self.min_step < halfway:
            self.min_step *= 2.0

    def _find_step_size(self, t: float, max_step: float = float("inf")) -> float:
        step_size = min((1.0 - t) * 2, max_step)
        state = self.clone_state()

        current_metric = 0.0
        tolerable_metric = self.get_target() * (1.0 - self.tolerance)
        while current_metric < tolerable_metric and step_size > self.min_step:
            step_size /= 2
            next_t = t + step_size
            current_metric = self.evaluate_fn(
                *[i(next_t) for i in self.interpolators]
            )
            self.restore_state(state)

        self._adjust_minimum_step(step_size, t)

        step_size = min(step_size, max_step, 1.0)
        return step_size

    def adapt_smoothly(self, max_cycles: Optional[int] = None) -> None:
        t = 0.0
        cycles = 0
        rate_ceiling = 1.0
        while t < 1 and (not max_cycles or cycles < max_cycles):
            available = rate_ceiling - t
            if available < self.min_step:
                break
            if self.before_cycle is not None:
                self.before_cycle()
            step_size = self._find_step_size(t, available)
            t_proposed = min(t + step_size, rate_ceiling)
            interpolated = [i(t_proposed) for i in self.interpolators]
            result = self.adaptation_fn(*interpolated)

            if result is not None and float(result) < t_proposed - 1e-9:
                t = float(result)
                rate_ceiling = (t + t_proposed) / 2
            else:
                t = t_proposed
                rate_ceiling = 1.0
            cycles += 1
