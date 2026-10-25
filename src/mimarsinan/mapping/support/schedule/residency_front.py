"""[E85/E297] The residency planner: phase fit, the exact minimal rotating region r* and the live-set carry front."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.support.residency_front_store import front_mincarry_store, front_store, lowest_starts

#: Phases beyond the fit the planner may use: Q in [Q_fit, Q_fit + Q_SPAN] (law.md Q_hi).
Q_SPAN = 8
#: Unit count from which the store front runs (``residency_front_store``) [W03]. Whole step functions, one cold call
#: each (RA/11_wave2/explore/w03/walls/front_walls.jsonl): below 30 units pure Python 5 ms against 21 ms per call;
#: from 100 units the store wins (DenseNet-BC-100 at C 144: 0.64 s against 1.78 s vectorized and 6.4 s Python;
#: t0_03's 640 tile units at C 162: 14.2 s against 41.2 s vectorized). The vectorized front (E296) stays a branch.
STORE_MIN_UNITS = 100

FrontFn = Callable[..., Optional[Tuple[int, int]]]


def _python_fronts(sizes, live, C, r, qmax, forced):
    """Pareto fronts over (excess, carry, phases), the RA's bare 3-tuples (its bisection calls it per probed r)."""
    n = len(sizes)
    prefix = [0]
    for s in sizes:
        prefix.append(prefix[-1] + s)
    budget = C - r
    starts = lowest_starts(n, forced)
    fronts: List[list] = [[] for _ in range(n + 1)]
    fronts[0] = [(0, 0, 0)]
    for j in range(1, n + 1):
        cand = []
        cut = 2 * live[j - 1] if j < n else 0
        for i in range(j - 1, starts[j] - 1, -1):
            width = prefix[j] - prefix[i]
            if width > C:
                break
            ex = max(0, width - r)
            for e, c, q in fronts[i]:
                if e + ex <= budget and q + 1 <= qmax:
                    cand.append((e + ex, c + cut, q + 1))
        cand.sort()
        keep: list = []
        for t in cand:
            if not any(k[0] <= t[0] and k[1] <= t[1] and k[2] <= t[2] for k in keep):
                keep.append(t)
        fronts[j] = keep
    return fronts


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
    starts = lowest_starts(n, forced)
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


FRONTS: Dict[str, FrontFn] = {"python": front_mincarry_py, "vectorized": front_mincarry_np,
                              "store": front_mincarry_store}


def select_front(unit_count: int) -> FrontFn:
    """The size dispatch: the pure-Python front below ``STORE_MIN_UNITS`` units, else the store front."""
    return FRONTS["store" if unit_count >= STORE_MIN_UNITS else "python"]


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
    """``x9_0_v3.plan_steps``: the steps (r, carry, Q) from the smallest feasible r, the stateless plan. [W03] The
    steps are found on demand (``step``; ``steps`` walks the whole function), each by bisection on r."""

    stateless: Tuple[int, int]
    q_fit: int
    F: int
    fluid_r: float
    C: int
    qmax: int
    forced: Tuple[int, ...] = ()
    walk: Optional["StepWalk"] = field(default=None, compare=False, repr=False)

    @property
    def steps(self) -> Tuple[Tuple[int, int, int], ...]:
        assert self.walk is not None
        return self.walk.every()

    def step(self, index: int) -> Optional[Tuple[int, int, int]]:
        """Step ``index`` of the function (0: the floor), or None past its last step."""
        assert self.walk is not None
        return self.walk.upto(int(index))

    def at(self, r: int) -> Optional[Tuple[int, int, int]]:
        """[W03] ``(r, carry, Q)`` of the minimal plan at ``r``, a step or not (None below r*)."""
        assert self.walk is not None
        got = self.walk.probe(int(r))
        return None if got is None else (int(r), int(got[0]), int(got[1]))

    def as_ra_dict(self) -> dict:
        """The RA function's own return value, for equality against it."""
        return dict(steps=list(self.steps), stateless=self.stateless, q_fit=self.q_fit, F=self.F,
                    fluid_r=self.fluid_r)


class StepWalk:
    """The steps of mincarry(r) from r*, each the smallest r whose carry falls below the last step's, probed once
    per r (feasibility and the carry are monotone in r)."""

    def __init__(self, probe: Callable[[int], Optional[Tuple[int, int]]], C: int, first: Tuple[int, int, int],
                 done: bool = False):
        self.probe, self.C, self.found, self.done = probe, int(C), [first], done

    def _at(self, r: int) -> Tuple[int, int]:
        got = self.probe(r)
        assert got is not None, "probed below r*"
        return got

    def _next(self) -> None:
        r, carry = self.found[-1][0], self.found[-1][1]
        if r >= self.C or self._at(self.C)[0] >= carry:
            self.done = True
            return
        lo, hi = r + 1, self.C
        while lo < hi:
            mid = (lo + hi) // 2
            if self._at(mid)[0] < carry:
                hi = mid
            else:
                lo = mid + 1
        self.found.append((lo, *self._at(lo)))

    def upto(self, index: int) -> Optional[Tuple[int, int, int]]:
        while len(self.found) <= index and not self.done:
            self._next()
        return self.found[index] if index < len(self.found) else None

    def every(self) -> Tuple[Tuple[int, int, int], ...]:
        while not self.done:
            self._next()
        return tuple(self.found)


def binding_cap(sizes: Sequence[int], C: int, forced: Sequence[int], qmax: int,
                cap: Optional[int]) -> Optional[int]:
    """[W03] A per-segment phase cap where it can bind: some segment (between forced cuts) could hold more phases
    than ``cap`` (its units, and at most ``qmax`` less the other segments' fewest); None where no plan exceeds it."""
    if cap is None:
        return None
    bounds = [0, *sorted(set(int(p) for p in forced)), len(sizes)]
    fewest = [q_fit(sizes[a:b], C) or 0 for a, b in zip(bounds, bounds[1:])]
    most = [min(b - a, int(qmax) - (sum(fewest) - k)) for (a, b), k in zip(zip(bounds, bounds[1:]), fewest)]
    return int(cap) if max(most) > int(cap) else None


def plan_steps(sizes, live, C, *, forced=(), front: Optional[FrontFn] = None,
               segment_cap: Optional[int] = None) -> Optional[ResidencyStepFunction]:
    """The step function of mincarry(r), non-increasing in r; None if no plan exists (a unit exceeds C, or a segment
    needs more phases than ``segment_cap``). A cap that can bind plans on the store front with it [W03]."""
    sizes = tuple(int(s) for s in sizes)
    live = tuple(int(v) for v in live)
    forced = tuple(sorted(int(p) for p in forced))
    F = sum(sizes)
    qf = q_fit(sizes, C, forced)
    if qf is None:
        return None
    qmax = qf + Q_SPAN
    cap = binding_cap(sizes, C, forced, qmax, segment_cap)
    if cap is not None and max(q_fit(sizes[a:b], C) or 0 for a, b in _segments(len(sizes), forced)) > cap:
        return None
    if F <= C and all(0 < p < len(sizes) for p in forced):
        return _fit(sizes, live, int(C), qf, qmax, forced)
    memo: Dict[int, Optional[Tuple[int, int]]] = {}
    run = front or select_front(len(sizes))

    def f(r: int) -> Optional[Tuple[int, int]]:
        if r not in memo:
            memo[r] = (run(sizes, live, C, r, qmax, forced) if cap is None else
                       front_mincarry_store(sizes, live, C, r, qmax, forced, segment_cap=cap))
        return memo[r]

    lo = max(0, math.ceil((F - C) / (qmax - 1)))
    stateless = f(C)
    if stateless is None:
        return None
    a, b = (lo, lo) if f(lo) is not None else (lo + 1, C)  # smallest feasible r (feasibility monotone in r)
    while a < b:
        mid = (a + b) // 2
        if f(mid) is None:
            a = mid + 1
        else:
            b = mid
    first = f(a)
    assert first is not None
    return ResidencyStepFunction(stateless=stateless, q_fit=qf, F=F, fluid_r=(F - C) / max(1, qf - 1), C=int(C),
                                 qmax=qmax, forced=forced, walk=StepWalk(f, int(C), (a, first[0], first[1])))


def _segments(n: int, forced: Sequence[int]) -> List[Tuple[int, int]]:
    bounds = [0, *sorted(set(int(p) for p in forced)), n]
    return list(zip(bounds, bounds[1:]))


def _fit(sizes, live, C, qf, qmax, forced) -> ResidencyStepFunction:
    """[W03] The step function of a fit (F <= C), in closed form: every r admits one phase per segment, which pays only
    the forced cuts' carry and runs the fewest phases (``q_fit``), so r* = 0 and the function is that one step."""
    carry = sum(2 * live[p - 1] for p in set(forced))
    return ResidencyStepFunction(stateless=(carry, qf), q_fit=qf, F=sum(sizes), fluid_r=(sum(sizes) - C) / max(1, qf - 1),
                                 C=C, qmax=qmax, forced=forced,
                                 walk=StepWalk(lambda r: (carry, qf), C, (0, carry, qf), done=True))


def phase_partition(sizes, live, C, r, qmax, forced=(), *,
                    segment_cap: Optional[int] = None) -> Optional[Tuple[Tuple[int, int], ...]]:
    """The phases ``[(start, end), ...]`` of the plan the front answers at ``r`` (its minimal (carry, Q)), from the
    store front with ``plan_steps``' cap and the pointer front's tie order."""
    sizes = tuple(int(s) for s in sizes)
    cap = binding_cap(sizes, C, forced, qmax, segment_cap)
    return front_store(sizes, tuple(int(v) for v in live), C, r, qmax, forced, segment_cap=cap).path()
