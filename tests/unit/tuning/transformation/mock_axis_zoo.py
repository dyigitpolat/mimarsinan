"""Analytic mock-axis surfaces for the RateScheduler (report IV.2).

Each archetype returns ``(attempt, expected_final)`` where ``attempt(target)`` is
the scheduler's per-cycle callable (returns the committed rate after the attempt:
``target`` on commit, the prior committed rate on rollback). The surfaces span
the profile space — smooth, cliff, plateau, recovery-limited, adversarial-timing,
non-monotone — so a single controller can be validated across them deterministically.
"""

from __future__ import annotations


def smooth_monotone():
    """Every increment is feasible → reaches 1.0."""
    state = {"c": 0.0}

    def attempt(target):
        state["c"] = target
        return state["c"]

    return attempt, 1.0


def cliff(alpha_star):
    """Feasible iff ``target <= alpha_star``; bisects to the edge, never past."""
    state = {"c": 0.0}

    def attempt(target):
        if target <= alpha_star + 1e-12:
            state["c"] = target
        return state["c"]

    return attempt, alpha_star


def plateau_then_drop(alpha_star):
    """Flat then a cliff at ``alpha_star`` — same feasibility shape as a cliff."""
    return cliff(alpha_star)


def recovery_limited(alpha_max):
    """The corrector recovers only up to ``alpha_max`` → partial result (I1)."""
    return cliff(alpha_max)


def adversarial_timing(max_advance=0.3):
    """A step infeasible early becomes feasible later as committed grows;
    each attempt advances at most ``max_advance`` from the current committed."""
    state = {"c": 0.0}

    def attempt(target):
        if target <= state["c"] + max_advance + 1e-12:
            state["c"] = target
        return state["c"]

    return attempt, 1.0


def non_monotone(low=0.4, high=0.6):
    """Feasible in ``[0, low]`` and ``[high, 1]`` (a gap in the middle).
    Bisection still terminates with a valid committed rate (I1/I2)."""
    state = {"c": 0.0}

    def attempt(target):
        if target <= low + 1e-12 or target >= high - 1e-12:
            state["c"] = target
        return state["c"]

    return attempt, 1.0  # the feasible endpoint at 1.0 is reachable greedily
