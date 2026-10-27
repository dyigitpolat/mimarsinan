"""[F3 round] Bounds that let the store front keep only the entries a minimal plan can pass through, exactly.

At one rotating region r, a phase [i, j) holds at most C tiles (and [F3] at most r operand tiles, no forced cut
inside) and costs carry cut(j) = 2 live[j - 1] (0 at the end) and excess (P_j - P_i - r)^+; a plan holds at most
``qmax`` phases and its excess is at most B = C - r. Over completions from position i in at most k phases:

- ``excess[k][i]`` is the least excess: an entry (e, c, q) at i with e + excess[qmax - q][i] > B has no feasible
  completion;
- for a weight lam, ``table[k][i]`` is the least carry + lam x excess: every feasible completion of the entry ends at a
  carry of at least c + table[qmax - q][i] - lam (B - e) (its own excess is at most B - e), and the path that attains
  ``table[qmax][0]`` is a plan whose carry bounds the minimal carry from above wherever it is feasible.

An entry whose least final carry exceeds that upper bound, or that has no feasible completion, is on no minimal plan.
Both tests are monotone in (e, c, q), so an entry the store's dominance rule removes in favour of another fails them
no sooner than the other, and an entry that passes has a predecessor that passes: dropping the failing entries keeps
every other entry's fate, and the store's best entry and phases are the full store's (``front_store(bound=...)``)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_INF = np.iinfo(np.int64).max // 8
#: Weights of the excess in the carry bound: 0, then powers of eight up to 8^10 (a carry is elements, an excess tiles).
LAMBDAS = (0,) + tuple(8 ** p for p in range(11))


@dataclass(frozen=True)
class PhaseWindows:
    """Per start position i, the last end ``near[i]`` of a phase without excess (at most r tiles) and the last end
    ``far[i]`` of any phase (at most C tiles, no forced cut inside, at most r operand tiles)."""

    near: np.ndarray
    far: np.ndarray
    prefix: np.ndarray
    cut: np.ndarray


def phase_windows(sizes: Sequence[int], live: Sequence[int], C: int, r: int, forced: Sequence[int] = (),
                  held: Optional[Sequence[int]] = None) -> PhaseWindows:
    n = len(sizes)
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + int(size))
    cuts, k = sorted(int(p) for p in forced), 0
    near, far = np.zeros(n + 1, dtype=np.int64), np.zeros(n + 1, dtype=np.int64)
    for i in range(n + 1):
        while k < len(cuts) and cuts[k] <= i:
            k += 1
        j = min(bisect.bisect_right(prefix, prefix[i] + int(C)) - 1, cuts[k] if k < len(cuts) else n)
        if held is not None:
            j = min(j, bisect.bisect_right(held, held[i] + int(r)) - 1)
        far[i] = j
        near[i] = min(j, bisect.bisect_right(prefix, prefix[i] + int(r)) - 1)
    cut = np.array([2 * int(live[j - 1]) if j < n else 0 for j in range(n + 1)], dtype=np.int64)
    return PhaseWindows(near=near, far=far, prefix=np.asarray(prefix, dtype=np.int64), cut=cut)


class _RangeQueries:
    """Queries [lo, hi] of one length class each (a sparse table's two overlapping blocks), shared by every table."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.size = lo.size
        self.valid = np.nonzero(lo <= hi)[0]
        lo, hi = lo[self.valid], hi[self.valid]
        span = np.floor(np.log2(np.maximum(hi - lo + 1, 1))).astype(np.int64)
        self.groups = [(int(p), sel, lo[sel], hi[sel] - (1 << int(p)) + 1)
                       for p in np.unique(span) for sel in (np.nonzero(span == p)[0],)]

    def argmin(self, values: np.ndarray) -> np.ndarray:
        """Per weight (row of ``values``) and query, an index of the least value in its range; -1 where empty."""
        rows, n = values.shape
        out = np.full((rows, self.size), -1, dtype=np.int64)
        if not self.valid.size:
            return out
        levels, width = [np.broadcast_to(np.arange(n), (rows, n))], 1
        top = max(p for p, *_ in self.groups)
        while len(levels) <= top:
            prev = levels[-1]
            a, b = prev[:, :n - 2 * width + 1], prev[:, width:n - width + 1]
            levels.append(np.where(np.take_along_axis(values, b, 1) < np.take_along_axis(values, a, 1), b, a))
            width *= 2
        pick = np.empty((rows, self.valid.size), dtype=np.int64)
        for p, sel, left, right in self.groups:
            a, b = levels[p][:, left], levels[p][:, right]
            pick[:, sel] = np.where(np.take_along_axis(values, b, 1) < np.take_along_axis(values, a, 1), b, a)
        out[:, self.valid] = pick
        return out


def completion_tables(w: PhaseWindows, r: int, qmax: int, lams: Sequence[int],
                      carry: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Per weight lam of ``lams``, ``table[l][k][i]``: the least carry x ``carry`` + lam x excess over the completions
    of position i in at most k phases (_INF: none), and ``after[l][k][i]`` the end of the first phase of one that
    attains it; one vectorized pass per k over every weight."""
    n = w.prefix.size - 1
    lam = np.asarray(lams, dtype=np.int64)[:, None]
    table = np.full((len(lams), qmax + 1, n + 1), _INF, dtype=np.int64)
    after = np.full((len(lams), qmax + 1, n + 1), -1, dtype=np.int32)
    table[:, :, n] = 0
    starts = np.arange(n, dtype=np.int64)
    near = _RangeQueries(starts + 1, w.near[:n])
    far = _RangeQueries(np.maximum(w.near[:n] + 1, starts + 1), w.far[:n])
    shift = lam * (w.prefix - int(r))[None, :]
    for k in range(1, qmax + 1):
        prev = table[:, k - 1, :]
        reach = prev < _INF
        near_v = np.where(reach, carry * w.cut[None, :] + prev, _INF)
        far_v = np.where(reach, carry * w.cut[None, :] + shift + prev, _INF)
        jn, jf = near.argmin(near_v), far.argmin(far_v)
        vn = np.where(jn >= 0, np.take_along_axis(near_v, np.maximum(jn, 0), 1), _INF)
        vf = np.where(jf >= 0, np.take_along_axis(far_v, np.maximum(jf, 0), 1), _INF)
        vf = np.where(vf < _INF, vf - lam * w.prefix[None, :n], _INF)
        table[:, k, :n] = np.minimum(vn, vf)
        after[:, k, :n] = np.where(vf < vn, jf, jn)
    return table, after


def _follow(w: PhaseWindows, after: np.ndarray, r: int, qmax: int) -> Optional[Tuple[int, int, List[int]]]:
    """(carry, excess, phase ends) of the completion from 0 that ``after`` records, or None."""
    n = w.prefix.size - 1
    i, k, carry, excess, ends = 0, qmax, 0, 0, []
    while i < n:
        if k == 0 or after[k][i] < 0:
            return None
        j = int(after[k][i])
        carry += int(w.cut[j])
        excess += max(0, int(w.prefix[j] - w.prefix[i]) - int(r))
        ends.append(j)
        i, k = j, k - 1
    return carry, excess, ends


@dataclass(frozen=True)
class CarryBound:
    """The tests of the module docstring at one r: per k the least excess to the end, the carry bounds' weights and
    tables, the budget B and the upper bound on the minimal carry (None: no feasible plan was found to bound it)."""

    budget: int
    qmax: int
    excess: np.ndarray
    weights: Tuple[int, ...]
    tables: Tuple[np.ndarray, ...]
    upper: Optional[int]
    #: The least carry of the weighted plans that keep the budget (a plan of that carry exists), or None.
    found: Optional[int] = None

    def keep(self, j: int, e: np.ndarray, c: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Which candidate entries (excess, carry, phases) at position ``j`` some minimal plan can pass through."""
        k = self.qmax - q
        ok = e + self.excess[k, j] <= self.budget
        if self.upper is not None:
            for lam, table in zip(self.weights, self.tables):
                ok &= c + table[k, j] - lam * (self.budget - e) <= self.upper
        return ok


def plan_exists(sizes: Sequence[int], C: int, r: int, qmax: int, forced: Sequence[int] = (),
                held: Optional[Sequence[int]] = None) -> bool:
    """Whether some plan at ``r`` keeps the budget (the least excess of a partition into at most ``qmax`` phases is at
    most C - r): exactly the store front's feasibility without a segment cap, from one excess table."""
    w = phase_windows(sizes, [0] * len(sizes), C, r, forced, held)
    excess, _ = completion_tables(w, r, qmax, (1,), carry=0)
    return bool(excess[0, qmax, 0] <= int(C) - int(r))


def carry_bound(sizes: Sequence[int], live: Sequence[int], C: int, r: int, qmax: int, forced: Sequence[int] = (),
                held: Optional[Sequence[int]] = None, segment_cap: Optional[int] = None,
                upper: Optional[int] = None) -> Optional[CarryBound]:
    """The bound at ``r``, or None where no plan exists (the least excess of any completion exceeds B) or, with
    ``upper``, where none carries at most ``upper`` (a weighted root bound exceeds it). The upper bound is the least
    carry of the weighted plans that keep the budget (and the segment cap where one is given), and at most ``upper``;
    the weights whose root bounds are highest are kept for the per-entry test."""
    w = phase_windows(sizes, live, C, r, forced, held)
    budget = int(C) - int(r)
    excess, _ = completion_tables(w, r, qmax, (1,), carry=0)
    if excess[0, qmax, 0] > budget:
        return None
    tables, after = completion_tables(w, r, qmax, LAMBDAS)
    found, roots = None, []
    for row, lam in enumerate(LAMBDAS):
        if tables[row, qmax, 0] >= _INF:
            continue
        roots.append((int(tables[row, qmax, 0]) - lam * budget, row))
        plan = _follow(w, after[row], r, qmax)
        if plan is not None and plan[1] <= budget and _within_cap(plan[2], forced, segment_cap):
            found = plan[0] if found is None else min(found, plan[0])
    if upper is not None:
        if any(root > int(upper) for root, _ in roots):
            return None
        upper = int(upper) if found is None else min(found, int(upper))
    else:
        upper = found
    best = [row for _, row in sorted(roots, key=lambda t: -t[0])[:3]]
    return CarryBound(budget=budget, qmax=int(qmax), excess=excess[0], weights=tuple(LAMBDAS[row] for row in best),
                      tables=tuple(tables[row] for row in best), upper=upper, found=found)


def _within_cap(ends: Sequence[int], forced: Sequence[int], cap: Optional[int]) -> bool:
    if cap is None:
        return True
    bounds = sorted(set(int(p) for p in forced))
    counts: dict = {}
    start = 0
    for end in ends:
        segment = bisect.bisect_right(bounds, start)
        counts[segment] = counts.get(segment, 0) + 1
        start = end
    return max(counts.values(), default=0) <= int(cap)
