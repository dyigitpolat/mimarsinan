"""[E85/E296] The residency front's reference branches: the pure-Python Pareto fronts (the RA's) and the vectorized
fronts (E296), each with optional forced cuts and [F3] at most r operand tiles per phase; the store front
(``residency_front_store``) is the planner's from 100 units (``residency_front.select_front``). Moved from
``schedule/residency_front.py``, which re-exports both answers."""

from __future__ import annotations

from typing import List

import numpy as np

from mimarsinan.mapping.support.residency_front_store import lowest_starts, operand_bound


def _python_fronts(sizes, live, C, r, qmax, forced, operand=()):
    """Pareto fronts over (excess, carry, phases), the RA's bare 3-tuples (its bisection calls it per probed r); [F3]
    a phase holds at most r operand tiles."""
    n, held = len(sizes), operand_bound(operand)
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
            if width > C or (held is not None and held[j] - held[i] > r):
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


def front_mincarry_py(sizes, live, C, r, qmax, forced=(), operand=()):
    """``x9_0_v3._front_mincarry`` with optional forced cuts: min (carry, phases) under every phase <= C and
    sum_q (R_q - r)^+ <= C - r, at most ``qmax`` phases ([F3] at most r operand tiles each), or None."""
    last = _python_fronts(sizes, live, C, r, qmax, forced, operand)[len(sizes)]
    if not last:
        return None
    best = last[_best(last)]
    return best[1], best[2]


def front_mincarry_np(sizes, live, C, r, qmax, forced=(), operand=()):
    """``v42a_plan.front_mincarry_np`` (E296, exact): the same fronts with numpy candidate arrays."""
    n, held = len(sizes), operand_bound(operand)
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
            if width > C or (held is not None and held[j] - held[i] > r):
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
