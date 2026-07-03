"""RateScheduler — one greedy-to-1.0-then-bisect rate-search policy (spec §5.2)."""

from __future__ import annotations

from typing import Callable, Optional


class RateScheduler:
    """Greedy-to-1.0 then bisect-the-gap rate search (spec §5.2)."""

    def __init__(
        self,
        *,
        epsilon: float,
        alpha_tol: float = 1e-6,
        policy: str = "greedy_to_one",
        initial_step: Optional[float] = None,
        max_rounds: Optional[int] = None,
        rates: Optional[list] = None,
    ):
        if policy not in (
            "greedy_to_one", "uniform_ladder", "one_shot_only", "dense_grid",
            "fixed_ladder",
        ):
            raise ValueError(f"unknown rate policy: {policy!r}")
        self.epsilon = float(epsilon)
        self.alpha_tol = float(alpha_tol)
        self.policy = policy
        self.initial_step = initial_step
        self.max_rounds = max_rounds
        self.rates = list(rates) if rates is not None else None

    def _first_step(self, gap: float) -> float:
        if self.policy == "uniform_ladder" and self.initial_step is not None:
            return min(float(self.initial_step), gap)
        if self.policy == "dense_grid":
            step = float(self.initial_step) if self.initial_step else self.epsilon
            return min(step, gap)
        return gap

    def run(self, committed: float, attempt: Callable[[float], float]) -> float:
        """Drive ``committed`` toward 1.0; return the highest committed rate."""
        committed = float(committed)
        if self.policy == "fixed_ladder":
            for r in (self.rates or []):
                result = attempt(min(float(r), 1.0))
                if result is not None:
                    committed = float(result)
            return committed
        rounds = 0
        while committed < 1.0 - self.alpha_tol:
            if self.max_rounds is not None and rounds >= self.max_rounds:
                break
            rounds += 1
            gap = 1.0 - committed
            step = self._first_step(gap)
            accepted = False
            # epsilon bounds only the bisection refinement, never the first jump.
            while True:
                target = min(committed + step, 1.0)
                result = attempt(target)
                now = float(result) if result is not None else committed
                if now >= target - 1e-9:
                    committed = now
                    accepted = True
                    break
                committed = now
                if self.policy == "one_shot_only":
                    return committed
                step /= 2.0
                if step < self.epsilon:
                    break
            if not accepted:
                break
        return committed
