"""RateScheduler — one greedy-to-1.0-then-bisect rate-search policy (spec §5.2)."""

from __future__ import annotations

from typing import Callable, Optional

from mimarsinan.tuning.orchestration import adaptation_ledger


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
        ledger=None,
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
        self._ledger = ledger

    def _first_step(self, gap: float) -> float:
        if self.policy == "uniform_ladder" and self.initial_step is not None:
            return min(float(self.initial_step), gap)
        if self.policy == "dense_grid":
            step = float(self.initial_step) if self.initial_step else self.epsilon
            return min(step, gap)
        return gap

    def run(self, committed: float, attempt: Callable[[float], float]) -> float:
        """Drive ``committed`` toward 1.0; return the highest committed rate.

        Each exit names the path it left through on the ledger, so a sealed run
        answers "which path completed the rate search" without re-deriving it.
        """
        committed = float(committed)
        if self.policy == "fixed_ladder":
            for r in (self.rates or []):
                result = attempt(min(float(r), 1.0))
                if result is not None:
                    committed = float(result)
            self._complete(adaptation_ledger.FIXED_LADDER_EXHAUSTED)
            return committed
        rounds = 0
        exit_path = adaptation_ledger.REACHED_FULL_RATE
        while committed < 1.0 - self.alpha_tol:
            if self.max_rounds is not None and rounds >= self.max_rounds:
                exit_path = adaptation_ledger.ROUND_BUDGET
                self._stall(exit_path, f"{rounds} rounds spent at {committed:.6f}")
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
                    self._stall(
                        adaptation_ledger.ONE_SHOT_REFUSED,
                        f"one shot at {target:.6f} refused; no bisection follows",
                    )
                    self._complete(adaptation_ledger.ONE_SHOT_REFUSED)
                    return committed
                refined = step / 2.0
                if refined < self.epsilon:
                    self._stall(
                        adaptation_ledger.EPSILON_FLOOR,
                        f"step {refined:.6f} < epsilon {self.epsilon:.6f}",
                    )
                    break
                self._refine(f"{step:.6f} -> {refined:.6f}")
                step = refined
            if not accepted:
                exit_path = adaptation_ledger.EPSILON_FLOOR
                break
        self._complete(exit_path)
        return committed

    def _refine(self, detail: str) -> None:
        adaptation_ledger.record_refinement(
            self._ledger, adaptation_ledger.BISECT_STEP, detail
        )

    def _stall(self, path: str, detail: str) -> None:
        adaptation_ledger.record_escalation(self._ledger, path, detail)

    def _complete(self, path: str) -> None:
        adaptation_ledger.record_completion(self._ledger, path)
