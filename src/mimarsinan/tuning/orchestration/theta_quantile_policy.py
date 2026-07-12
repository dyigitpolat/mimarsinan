"""[R4/S1] S-aware theta-quantile policy: grid levels -> calibration quantile."""

from __future__ import annotations

import math

# Measured optimum of the theta-loading quantile per grid resolution
# (sync_deployment_exactness.md §3.3, B1/A2e scans): the optimal quantile FALLS
# as the grid coarsens — S=4 hops selected 0.90–0.995 (0.95 representative),
# S=8 ~0.99, S=16 mostly 0.995–1.0. Between anchors the policy interpolates
# linearly in log2(levels); outside it clamps to the nearest anchor.
S_AWARE_QUANTILE_ANCHORS: dict[int, float] = {
    4: 0.95,
    8: 0.99,
    16: 0.995,
    32: 1.0,
}


def s_aware_quantile_for_levels(levels: int) -> float:
    """The memo-anchored quantile for a ``levels``-step value grid."""
    if int(levels) <= 0:
        raise ValueError(
            f"theta-quantile policy needs a positive grid level count, got {levels}"
        )
    log_levels = math.log2(int(levels))
    anchors = sorted(S_AWARE_QUANTILE_ANCHORS.items())
    if log_levels <= math.log2(anchors[0][0]):
        return anchors[0][1]
    if log_levels >= math.log2(anchors[-1][0]):
        return anchors[-1][1]
    for (s_lo, q_lo), (s_hi, q_hi) in zip(anchors, anchors[1:]):
        lo, hi = math.log2(s_lo), math.log2(s_hi)
        if lo <= log_levels <= hi:
            t = (log_levels - lo) / (hi - lo)
            return q_lo + (q_hi - q_lo) * t
    raise AssertionError("unreachable: anchor interval scan is exhaustive")


def effective_theta_quantile(levels: int | None, base_quantile: float) -> float:
    """``min(base, policy(levels))`` — the policy only ever deflates the configured
    base (a mode's proven quantile is never raised); continuous modes
    (``levels is None``) keep the base."""
    if levels is None:
        return float(base_quantile)
    return min(float(base_quantile), s_aware_quantile_for_levels(levels))
