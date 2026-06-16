"""RateScheduler — one rate-search policy replacing the three legacy loops.

Implements the spec's §5.2 search: each round greedily attempts the full jump to
1.0, then bisects the *remaining gap* on failure (``1.0`` → ``committed+gap/2`` →
…), committing the largest feasible increment and repeating. This subsumes the
legacy one-shot attempt, the grow/halve ramp, and the continue-to-full-rate
loop into a single policy. The ``fixed_ladder`` policy is the schedule-not-search
variant (a well-conditioned transformation walks an explicit rate list with no
bisection) — the home of the folded genuine-blend fast path.

``attempt(target)`` is the per-cycle callable (the tuner's ``_adaptation``): it
returns the committed rate after the attempt — ``target`` on commit, the prior
committed rate on rollback. Equivalence tiers: the one-shot single attempt is
Tier-A exact (== spec round 1); the ramp / continue-to-full-rate trajectories
are Tier-B (outcome equivalence: same final committed rate, monotone progress,
bounded probes) and their goldens are re-baselined with a documented diff.
"""

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
            # Safe mode for a non-monotone axis: never greedily jump to 1.0 (which
            # could vault a feasible region into an infeasible one and wrongly
            # conclude failure). Walk small uniform increments (the characterized
            # epsilon_hint) instead, bisecting only on a local failure.
            step = float(self.initial_step) if self.initial_step else self.epsilon
            return min(step, gap)
        return gap  # greedy_to_one / one_shot_only: jump to 1.0

    def run(self, committed: float, attempt: Callable[[float], float]) -> float:
        """Drive ``committed`` toward 1.0; return the highest committed rate."""
        committed = float(committed)
        if self.policy == "fixed_ladder":
            # Schedule-not-search: walk an explicit rate list in order, committing
            # whatever each attempt reports, never bisecting (the well-conditioned
            # transformation has no cliff to search). epsilon/max_rounds are unused.
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
            # The first (greedy / ladder) jump is always attempted — epsilon only
            # bounds the *bisection* refinement, never the initial jump, so a
            # degenerate epsilon >= gap still tries the full step (spec §5.2).
            while True:
                target = min(committed + step, 1.0)
                result = attempt(target)
                now = float(result) if result is not None else committed
                if now >= target - 1e-9:
                    committed = now
                    accepted = True
                    break
                committed = now  # rolled back to its committed rate
                if self.policy == "one_shot_only":
                    return committed
                step /= 2.0
                if step < self.epsilon:
                    break
            if not accepted:
                break
        return committed
