"""[E85/E297] The residency planner: phase fit, the exact minimal rotating region r* and the live-set carry front."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Phases beyond the fit the planner may use: Q in [Q_fit, Q_fit + Q_SPAN] (law.md Q_hi).
Q_SPAN = 8
#: Unit count from which the vectorized front runs. Measured per cold call (07_engineering_refinements.md E296):
#: 21 units (ResNet-18) 0.01 s pure Python against 0.09 s vectorized; 100 units tie at C' 36 (0.33 s against
#: 0.34 s) and the vectorized front wins from C' 72 on (2.4x to 9.7x).
VECTORIZED_MIN_UNITS = 100

FrontFn = Callable[..., Optional[Tuple[int, int]]]


def _lowest_starts(n: int, forced: Sequence[int]) -> List[int]:
    """``starts[j]``: the first unit a phase ending at ``j`` may start at (no forced cut strictly inside it)."""
    cuts = sorted(int(p) for p in forced)
    starts, k, last = [0] * (n + 1), 0, 0
    for j in range(1, n + 1):
        while k < len(cuts) and cuts[k] < j:
            last, k = cuts[k], k + 1
        starts[j] = last
    return starts


def _python_fronts(sizes, live, C, r, qmax, forced, pointers: bool = False):
    """Pareto fronts over (excess, carry, phases), each kept tuple with a back-pointer (start, entry) when asked
    (the pure front keeps the RA's bare 3-tuples: its bisection calls it per probed r)."""
    n = len(sizes)
    prefix = [0]
    for s in sizes:
        prefix.append(prefix[-1] + s)
    budget = C - r
    starts = _lowest_starts(n, forced)
    fronts: List[list] = [[] for _ in range(n + 1)]
    fronts[0] = [(0, 0, 0, -1, -1) if pointers else (0, 0, 0)]
    for j in range(1, n + 1):
        cand = []
        cut = 2 * live[j - 1] if j < n else 0
        for i in range(j - 1, starts[j] - 1, -1):
            width = prefix[j] - prefix[i]
            if width > C:
                break
            ex = max(0, width - r)
            if pointers:
                for k, (e, c, q, _, _) in enumerate(fronts[i]):
                    if e + ex <= budget and q + 1 <= qmax:
                        cand.append((e + ex, c + cut, q + 1, i, k))
            else:
                for e, c, q in fronts[i]:
                    if e + ex <= budget and q + 1 <= qmax:
                        cand.append((e + ex, c + cut, q + 1))
        if pointers:
            cand.sort(key=_head)
        else:
            cand.sort()
        keep: list = []
        for t in cand:
            if not any(k[0] <= t[0] and k[1] <= t[1] and k[2] <= t[2] for k in keep):
                keep.append(t)
        fronts[j] = keep
    return fronts


def _head(t):
    return t[:3]


def _best(front):
    return min(range(len(front)), key=lambda k: (front[k][1], front[k][2]))


def front_mincarry_py(sizes, live, C, r, qmax, forced=()):
    """``x9_0_v3._front_mincarry`` with optional forced cuts: min (carry, phases) under every phase <= C and
    sum_q (R_q - r)^+ <= C - r, at most ``qmax`` phases, or None."""
    last = _python_fronts(sizes, live, C, r, qmax, forced)[len(sizes)]
    if not last:
        return None
    best = last[_best(last)]
    return best[1], best[2]


def front_mincarry_np(sizes, live, C, r, qmax, forced=()):
    """``v42a_plan.front_mincarry_np`` (E296, exact): the same fronts with numpy candidate arrays."""
    n = len(sizes)
    prefix = [0]
    for s in sizes:
        prefix.append(prefix[-1] + s)
    budget = C - r
    starts = _lowest_starts(n, forced)
    empty = np.zeros(0, dtype=np.int64)
    E: List[np.ndarray] = [np.zeros(1, dtype=np.int64)] + [empty] * n
    Cc: List[np.ndarray] = [np.zeros(1, dtype=np.int64)] + [empty] * n
    Q: List[np.ndarray] = [np.zeros(1, dtype=np.int64)] + [empty] * n
    big = np.iinfo(np.int64).max
    for j in range(1, n + 1):
        cut = 2 * live[j - 1] if j < n else 0
        es, cs, qs = [], [], []
        for i in range(j - 1, starts[j] - 1, -1):
            width = prefix[j] - prefix[i]
            if width > C:
                break
            if E[i].size == 0:
                continue
            e2, q2 = E[i] + max(0, width - r), Q[i] + 1
            ok = (e2 <= budget) & (q2 <= qmax)
            if ok.any():
                es.append(e2[ok])
                cs.append(Cc[i][ok] + cut)
                qs.append(q2[ok])
        if not es:
            continue
        e, c, q = np.concatenate(es), np.concatenate(cs), np.concatenate(qs)
        o = np.lexsort((q, c, e))
        e, c, q = e[o], c[o], q[o]
        keep = np.ones(e.size, dtype=bool)
        for qq in np.unique(q):
            # exclusive prefix minimum of carry over earlier candidates with phases <= qq
            pm = np.minimum.accumulate(np.where(q <= qq, c, big))
            prev = np.r_[big, pm[:-1]]
            sel = q == qq
            keep[sel] &= prev[sel] > c[sel]
        E[j], Cc[j], Q[j] = e[keep], c[keep], q[keep]
    if E[n].size == 0:
        return None
    o = np.lexsort((E[n], Q[n], Cc[n]))
    return int(Cc[n][o[0]]), int(Q[n][o[0]])


FRONTS: Dict[str, FrontFn] = {"python": front_mincarry_py, "vectorized": front_mincarry_np}


def select_front(unit_count: int) -> FrontFn:
    """The size dispatch: the pure-Python front below ``VECTORIZED_MIN_UNITS`` units, else the vectorized one."""
    return FRONTS["vectorized" if unit_count >= VECTORIZED_MIN_UNITS else "python"]


def q_fit(sizes: Sequence[int], C: int, forced: Sequence[int] = ()) -> Optional[int]:
    """Fewest contiguous phases within C, a phase never spanning a forced cut (greedy is optimal); None if a
    unit exceeds C."""
    if max(sizes) > C:
        return None
    cuts = set(int(p) for p in forced)
    q, acc = 1, 0
    for index, s in enumerate(sizes):
        if index in cuts or acc + s > C:
            q, acc = q + 1, s
        else:
            acc += s
    return q


@dataclass(frozen=True)
class ResidencyStepFunction:
    """``x9_0_v3.plan_steps``: the steps (r, carry, Q) from the smallest feasible r, the stateless plan."""

    steps: Tuple[Tuple[int, int, int], ...]
    stateless: Tuple[int, int]
    q_fit: int
    F: int
    fluid_r: float
    C: int
    qmax: int
    forced: Tuple[int, ...] = ()

    def as_ra_dict(self) -> dict:
        """The RA function's own return value, for equality against it."""
        return dict(steps=list(self.steps), stateless=self.stateless, q_fit=self.q_fit, F=self.F,
                    fluid_r=self.fluid_r)


def plan_steps(sizes, live, C, *, forced=(), front: Optional[FrontFn] = None) -> Optional[ResidencyStepFunction]:
    """The step function of mincarry(r), non-increasing in r; None if no layer-granular plan exists."""
    sizes = tuple(int(s) for s in sizes)
    live = tuple(int(v) for v in live)
    forced = tuple(sorted(int(p) for p in forced))
    F = sum(sizes)
    qf = q_fit(sizes, C, forced)
    if qf is None:
        return None
    qmax = qf + Q_SPAN
    run = front or select_front(len(sizes))
    memo: Dict[int, Optional[Tuple[int, int]]] = {}

    def f(r: int) -> Optional[Tuple[int, int]]:
        if r not in memo:
            memo[r] = run(sizes, live, C, r, qmax, forced)
        return memo[r]

    def g(r: int) -> Tuple[int, int]:
        """``f`` at an r at or above r* (feasibility is monotone in r)."""
        got = f(r)
        assert got is not None
        return got

    lo = max(0, math.ceil((F - C) / (qmax - 1)))
    if f(C) is None:
        return None
    a, b = lo, C                                   # smallest feasible r (feasibility monotone in r)
    while a < b:
        mid = (a + b) // 2
        if f(mid) is None:
            a = mid + 1
        else:
            b = mid
    r = a
    cur = g(r)
    steps = [(r, cur[0], cur[1])]
    while r < C:
        lo2, hi2 = r + 1, C
        if g(hi2)[0] >= cur[0]:
            break
        while lo2 < hi2:
            mid = (lo2 + hi2) // 2
            if g(mid)[0] < cur[0]:
                hi2 = mid
            else:
                lo2 = mid + 1
        r = lo2
        cur = g(r)
        steps.append((r, cur[0], cur[1]))
    sl = g(C)
    return ResidencyStepFunction(steps=tuple(steps), stateless=(sl[0], sl[1]), q_fit=qf, F=F,
                                 fluid_r=(F - C) / max(1, qf - 1), C=int(C), qmax=qmax, forced=forced)


def phase_partition(sizes, live, C, r, qmax, forced=()) -> Optional[Tuple[Tuple[int, int], ...]]:
    """The phases ``[(start, end), ...]`` of the plan the front answers at ``r`` (its minimal (carry, Q))."""
    sizes = tuple(int(s) for s in sizes)
    fronts = _python_fronts(sizes, tuple(int(v) for v in live), C, r, qmax, forced, pointers=True)
    j = len(sizes)
    if not fronts[j]:
        return None
    k = _best(fronts[j])
    phases: List[Tuple[int, int]] = []
    while j > 0:
        _, _, _, i, prev = fronts[j][k]
        phases.append((i, j))
        j, k = i, prev
    return tuple(reversed(phases))
