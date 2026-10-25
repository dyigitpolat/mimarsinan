"""[W03] The residency front over one store of every position's front: one vectorized step per position, phases and caps."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_BIG = np.iinfo(np.int64).max
#: Store columns: excess, carry, phases, position, predecessor entry, phases of the current segment.
_COLUMNS = 6


def lowest_starts(n: int, forced: Sequence[int]) -> List[int]:
    """``starts[j]``: the first unit a phase ending at ``j`` may start at (no forced cut strictly inside it)."""
    cuts = sorted(int(p) for p in forced)
    starts, k, last = [0] * (n + 1), 0, 0
    for j in range(1, n + 1):
        while k < len(cuts) and cuts[k] < j:
            last, k = cuts[k], k + 1
        starts[j] = last
    return starts


@dataclass(frozen=True)
class FrontStore:
    """Every position's Pareto front over (excess, carry, phases) in one store, in position order: per entry its
    excess, carry, phases, position and predecessor entry; ``off[j]`` is the first entry of position ``j``."""

    excess: np.ndarray
    carry: np.ndarray
    phases: np.ndarray
    position: np.ndarray
    source: np.ndarray
    off: Tuple[int, ...]

    def best(self) -> Optional[int]:
        """The last position's entry of minimal (carry, phases), the first in front order among equals."""
        a, b = self.off[-2], self.off[-1]
        if a == b:
            return None
        return a + int(np.lexsort((np.arange(a, b), self.phases[a:b], self.carry[a:b]))[0])

    def path(self) -> Optional[Tuple[Tuple[int, int], ...]]:
        """The phases ``[(start, end), ...]`` of the best entry, followed back through the predecessors."""
        g = self.best()
        if g is None:
            return None
        out: List[Tuple[int, int]] = []
        while int(self.position[g]) > 0:
            s = int(self.source[g])
            out.append((int(self.position[s]), int(self.position[g])))
            g = s
        return tuple(reversed(out))


def _pareto(e, c, q, s, keys) -> np.ndarray:
    """Indices, in (e, c, q, s) order then ``keys`` order, of the candidates no earlier one dominates in carry,
    phases and (when ``s`` is given) the segment's phases; of equal candidates the first is kept."""
    order = np.lexsort(tuple(keys) + ((q,) if s is None else (s, q)) + (c, e))
    c, q = c[order], q[order]
    s = None if s is None else s[order]
    keep = np.ones(order.size, dtype=bool)
    groups = np.unique(q)[:, None] if s is None else np.unique(np.stack([q, s]), axis=1).T
    for g in groups:
        below = q <= g[0] if s is None else (q <= g[0]) & (s <= g[1])
        mine = q == g[0] if s is None else (q == g[0]) & (s == g[1])
        prev = np.r_[_BIG, np.minimum.accumulate(np.where(below, c, _BIG))[:-1]]
        keep[mine] &= prev[mine] > c[mine]
    return order[keep]


def front_store(sizes: Sequence[int], live: Sequence[int], C: int, r: int, qmax: int,
                forced: Sequence[int] = (), *, segment_cap: Optional[int] = None) -> FrontStore:
    """The front DP of ``residency_front`` (phases within C, sum (R_q - r)^+ <= C - r, at most ``qmax`` phases, no
    phase across a forced cut), with its pointer front's tie order (sorted by (e, c, q), then the later start, then
    front order). With ``segment_cap`` a segment (between forced cuts) holds at most that many phases: inside a
    segment its phase count joins the state, and it is dropped at the segment's end."""
    n = len(sizes)
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + int(size))
    P = np.asarray(prefix, dtype=np.int64)
    budget, starts = int(C) - int(r), lowest_starts(n, forced)
    ends = set(int(p) for p in forced) | {n}
    cap = 1024
    cols = np.zeros((_COLUMNS, cap), dtype=np.int64)
    cols[4, 0] = -1                                            # entry 0: position 0, (0, 0, 0), no predecessor
    off, size = [0, 1], 1
    for j in range(1, n + 1):
        lo = max(starts[j], bisect.bisect_left(prefix, prefix[j] - int(C), 0, j))
        a, b = off[lo], off[j]
        if a < b:
            e0, c0, q0, pos, s0 = cols[0, a:b], cols[1, a:b], cols[2, a:b], cols[3, a:b], cols[5, a:b]
            e = e0 + np.maximum(0, prefix[j] - P[pos] - int(r))
            q, s = q0 + 1, s0 + 1
            ok = (e <= budget) & (q <= int(qmax))
            if segment_cap is not None:
                ok &= s <= int(segment_cap)
            idx = np.nonzero(ok)[0]
            if idx.size:
                cut = 2 * int(live[j - 1]) if j < n else 0
                interior = segment_cap is not None and j not in ends
                e, q, s, c, src = e[idx], q[idx], s[idx], c0[idx] + cut, idx + a
                pick = _pareto(e, c, q, s if interior else None, (src, -pos[idx]))
                m = int(pick.size)
                while size + m > cap:
                    cap *= 2
                    grown = np.zeros((_COLUMNS, cap), dtype=np.int64)
                    grown[:, :size] = cols[:, :size]
                    cols = grown
                cols[0, size:size + m], cols[1, size:size + m], cols[2, size:size + m] = e[pick], c[pick], q[pick]
                cols[3, size:size + m], cols[4, size:size + m] = j, src[pick]
                cols[5, size:size + m] = s[pick] if interior else 0
                size += m
        off.append(size)
    return FrontStore(excess=cols[0, :size], carry=cols[1, :size], phases=cols[2, :size], position=cols[3, :size],
                      source=cols[4, :size], off=tuple(off))


def front_mincarry_store(sizes, live, C, r, qmax, forced=(), *, segment_cap: Optional[int] = None):
    """``front_store``'s minimal (carry, phases), or None: the residency front's answer from one store."""
    store = front_store(sizes, live, C, r, qmax, forced, segment_cap=segment_cap)
    g = store.best()
    return None if g is None else (int(store.carry[g]), int(store.phases[g]))
