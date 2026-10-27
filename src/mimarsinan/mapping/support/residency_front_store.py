"""[W03] The residency front over one store of every position's front: one vectorized step per position, phases and
caps; [F3 round] each position's window of start positions kept as merged fronts, so a step merges a bounded number of
fronts instead of scanning the window; [fix round] a short window scanned (``scans``), the same store at less cost."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.support.residency_front_bounds import CarryBound, carry_bound
from mimarsinan.mapping.support.residency_front_window import EMPTY as _EMPTY, WindowFront, scans

_BIG = np.iinfo(np.int64).max


def operand_bound(operand: Sequence[int]) -> Optional[List[int]]:
    """[F3] The prefix sums of the units' operand tiles (a phase [i, j) holds ``bound[j] - bound[i]`` of them, at most
    r: they are never pinned, so they rotate), or None where no unit holds one (every r admits every phase)."""
    if not any(int(o) for o in operand):
        return None
    out = [0]
    for o in operand:
        out.append(out[-1] + int(o))
    return out


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
                forced: Sequence[int] = (), *, segment_cap: Optional[int] = None, operand: Sequence[int] = (),
                bound: Optional[CarryBound] = None, keep: bool = True, scan: Optional[bool] = None) -> FrontStore:
    """The front DP of ``residency_front`` (phases within C, sum (R_q - r)^+ <= C - r, at most ``qmax`` phases, no
    phase across a forced cut, [F3] at most r operand tiles per phase), with its pointer front's tie order (sorted by
    (e, c, q), then the later start, then front order). With ``segment_cap`` a segment (between forced cuts) holds at
    most that many phases: inside a segment its phase count joins the state, and it is dropped at the segment's
    end.

    [F3 round] The positions a phase ending at j may start at form a window [lo, j) that only moves right: the near
    ones (a phase of at most r tiles, no excess) and the far ones (excess (P_j - r) - P_i). Each is a queue of the
    positions' entries with its front (``WindowFront``; a far entry keyed by its excess less its position's
    prefix, so the queue's order does not move with j), and the end's entries are the store's rule on the two
    fronts: the same store, entry for entry, as scanning every position of the window. With ``bound``
    (``residency_front_bounds.carry_bound`` at this r) an entry no minimal plan passes through is not stored: the
    store's best entry and phases are the full store's (``planned_store``). With ``keep`` false only the best entry is
    read: a position's entries are released once it leaves both queues, and the store holds the end's entries alone
    (``best`` only, no ``path``). [Fix round] ``scan``: the window scan (``_scan``), the sliding fronts, or (None)
    the scan where ``scans`` holds."""
    if scans(sizes, C, segment_cap is not None) if scan is None else scan:
        return _scan(sizes, live, C, r, qmax, forced, segment_cap, operand, bound, keep)
    n = len(sizes)
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + int(size))
    budget, starts, held = int(C) - int(r), lowest_starts(n, forced), operand_bound(operand)
    ends = set(int(p) for p in forced) | {n}
    fine = segment_cap is not None
    blocks: List[np.ndarray] = [np.zeros((6, 1), dtype=np.int64)]   # entry 0: position 0, (0, 0, 0)
    sources: List[np.ndarray] = [np.full(1, -1, dtype=np.int64)]
    near, far, far_positions = WindowFront(fine), WindowFront(fine), []
    pushed = moved = far_head = 0
    off, size = [0, 1], 1
    for j in range(1, n + 1):
        lo = max(starts[j], bisect.bisect_left(prefix, prefix[j] - int(C), 0, j))
        if held is not None:
            lo = max(lo, bisect.bisect_left(held, held[j] - int(r), 0, j))
        mid = max(lo, bisect.bisect_left(prefix, prefix[j] - int(r), 0, j))
        while pushed < j:
            near.push(blocks[pushed])
            pushed += 1
        while moved < mid:
            near.pop()
            if moved >= lo:
                shifted = blocks[moved].copy()
                shifted[0] -= prefix[moved]
                far.push(shifted)
                far_positions.append(moved)
            if not keep:
                blocks[moved], sources[moved] = _EMPTY, sources[0][:0]
            moved += 1
        while far_head < len(far_positions) and far_positions[far_head] < lo:
            far.pop()
            far_head += 1
        nearest, farthest = near.query(), far.query()
        if farthest.shape[1]:
            farthest = farthest.copy()
            farthest[0] += prefix[j] - int(r)
        cand = np.concatenate([nearest, farthest], axis=1)
        e, c0, q, s, pos, src = cand[0], cand[1], cand[2] + 1, cand[3] + 1, cand[4], cand[5]
        ok = (e <= budget) & (q <= int(qmax))
        if segment_cap is not None:
            ok &= s <= int(segment_cap)
        idx = np.nonzero(ok)[0]
        cut = 2 * int(live[j - 1]) if j < n else 0
        if bound is not None and idx.size:
            idx = idx[bound.keep(j, e[idx], c0[idx] + cut, q[idx])]
        entries = _EMPTY
        if idx.size:
            interior = segment_cap is not None and j not in ends
            e, q, s, c, src = e[idx], q[idx], s[idx], c0[idx] + cut, src[idx]
            pick = _pareto(e, c, q, s if interior else None, (src, -pos[idx]))
            m = int(pick.size)
            entries = np.stack([e[pick], c[pick], q[pick], s[pick] if interior else np.zeros(m, dtype=np.int64),
                                np.full(m, j, dtype=np.int64), np.arange(size, size + m, dtype=np.int64)])
            sources.append(src[pick])
            size += m
        if not entries.shape[1]:
            sources.append(sources[0][:0])
        blocks.append(entries)
        off.append(size)
    if not keep:
        last = blocks[-1]
        return FrontStore(excess=last[0], carry=last[1], phases=last[2], position=last[4], source=sources[-1][:0],
                          off=(0, 0, last.shape[1]))
    store = np.concatenate(blocks, axis=1)
    return FrontStore(excess=store[0], carry=store[1], phases=store[2], position=store[4],
                      source=np.concatenate(sources), off=tuple(off))


def _scan(sizes, live, C, r, qmax, forced, segment_cap, operand, bound, keep) -> FrontStore:
    """``front_store`` by the window scan [W03]: every entry of the window [lo, j) is a candidate of the end j, the
    carry bound's test first where one is given, kept by the store's rule (``_pareto``); with ``keep`` false the end's
    entries alone, as the sliding fronts return them."""
    n = len(sizes)
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + int(size))
    P = np.asarray(prefix, dtype=np.int64)
    budget, starts, held = int(C) - int(r), lowest_starts(n, forced), operand_bound(operand)
    ends = set(int(p) for p in forced) | {n}
    cols, off, size = np.zeros((6, 1024), dtype=np.int64), [0, 1], 1
    cols[4, 0] = -1  # entry 0: position 0, (0, 0, 0), no predecessor; rows e, c, q, position, source, segment's q
    for j in range(1, n + 1):
        lo = max(starts[j], bisect.bisect_left(prefix, prefix[j] - int(C), 0, j))
        if held is not None:
            lo = max(lo, bisect.bisect_left(held, held[j] - int(r), 0, j))
        a, b = off[lo], off[j]
        if a == b:
            off.append(size)
            continue
        e0, c0, q0, pos, s0 = cols[0, a:b], cols[1, a:b], cols[2, a:b], cols[3, a:b], cols[5, a:b]
        e, q, s = e0 + np.maximum(0, prefix[j] - P[pos] - int(r)), q0 + 1, s0 + 1
        ok = (e <= budget) & (q <= int(qmax))
        if segment_cap is not None:
            ok &= s <= int(segment_cap)
        idx = np.nonzero(ok)[0]
        cut = 2 * int(live[j - 1]) if j < n else 0
        if bound is not None and idx.size:
            idx = idx[bound.keep(j, e[idx], c0[idx] + cut, q[idx])]
        if idx.size:
            interior = segment_cap is not None and j not in ends
            e, q, s, c, src = e[idx], q[idx], s[idx], c0[idx] + cut, idx + a
            pick = _pareto(e, c, q, s if interior else None, (src, -pos[idx]))
            m = int(pick.size)
            if size + m > cols.shape[1]:
                grown = np.zeros((6, max(2 * cols.shape[1], size + m)), dtype=np.int64)
                grown[:, :size] = cols[:, :size]
                cols = grown
            cols[0, size:size + m], cols[1, size:size + m], cols[2, size:size + m] = e[pick], c[pick], q[pick]
            cols[3, size:size + m], cols[4, size:size + m] = j, src[pick]
            cols[5, size:size + m] = s[pick] if interior else 0
            size += m
        off.append(size)
    a = 0 if keep else off[-2]
    return FrontStore(excess=cols[0, a:size], carry=cols[1, a:size], phases=cols[2, a:size], position=cols[3, a:size],
                      source=cols[4, a:size] if keep else cols[4, :0], off=tuple(off) if keep else (0, 0, size - a))


def planned_store(sizes: Sequence[int], live: Sequence[int], C: int, r: int, qmax: int, forced: Sequence[int] = (), *,
                  segment_cap: Optional[int] = None, operand: Sequence[int] = (), upper: Optional[int] = None,
                  keep: bool = True) -> Optional[FrontStore]:
    """[F3 round] The store the planner reads at ``r``: ``front_store`` with the carry bound at r, whose best entry and
    phases are the full store's; None where no plan exists (no completion keeps the budget). The bound is computed
    where it costs less than the windows it prunes: its tables hold qmax rows of the program per weight, the store
    merges a window of about len(sizes) x C / F positions per position, so a program whose windows are shorter than
    qmax (a small C against many phases) reads the full store. ``upper``: only a plan of carry at most ``upper`` is
    asked for (None where there is none); ``keep`` as ``front_store``'s."""
    if len(sizes) * int(C) <= int(qmax) * max(1, sum(int(t) for t in sizes)):
        return front_store(sizes, live, C, r, qmax, forced, segment_cap=segment_cap, operand=operand, keep=keep)
    return bounded_store(sizes, live, C, r, qmax, forced, segment_cap=segment_cap, operand=operand, upper=upper,
                         keep=keep)


def bounded_store(sizes: Sequence[int], live: Sequence[int], C: int, r: int, qmax: int, forced: Sequence[int] = (), *,
                  segment_cap: Optional[int] = None, operand: Sequence[int] = (), upper: Optional[int] = None,
                  keep: bool = True, bound: Optional[CarryBound] = None) -> Optional[FrontStore]:
    """``front_store`` with the carry bound at r (``residency_front_bounds.carry_bound``, at most ``upper`` where
    given; or the ``bound`` given), or None where no completion keeps the budget (or none carries at most ``upper``):
    the full store's best entry and phases (where the full store's best carries at most ``upper``)."""
    if bound is None:
        bound = carry_bound(sizes, live, C, r, qmax, forced, operand_bound(operand), segment_cap, upper)
    if bound is None:
        return None
    return front_store(sizes, live, C, r, qmax, forced, segment_cap=segment_cap, operand=operand, bound=bound,
                       keep=keep)


def front_mincarry_store(sizes, live, C, r, qmax, forced=(), *, segment_cap: Optional[int] = None, operand=()):
    """``front_store``'s minimal (carry, phases), or None: the residency front's answer from one (planned) store."""
    return front_mincarry_below(sizes, live, C, r, qmax, forced, None, segment_cap=segment_cap, operand=operand)


def carries_below(sizes, live, C, r, qmax, forced=(), ceiling: int = 0, *, segment_cap: Optional[int] = None,
                  operand=()) -> Optional[bool]:
    """[F3 round] Whether the minimal plan at ``r`` carries less than ``ceiling`` where the carry bound decides it
    alone: False where no plan carries at most ``ceiling - 1`` (none keeps the budget, or a weighted root bound
    exceeds it), True where a weighted plan that keeps the budget carries less; None where only the store decides
    (or where the program reads the full store: ``planned_store``)."""
    if len(sizes) * int(C) <= int(qmax) * max(1, sum(int(t) for t in sizes)):
        return None
    bound = carry_bound(sizes, live, C, r, qmax, forced, operand_bound(operand), segment_cap, int(ceiling) - 1)
    if bound is None:
        return False
    return True if bound.found is not None and bound.found < int(ceiling) else None


def front_mincarry_below(sizes, live, C, r, qmax, forced=(), ceiling: Optional[int] = None, *,
                         segment_cap: Optional[int] = None, operand=()):
    """[F3 round] The minimal (carry, phases) at ``r`` where its carry is below ``ceiling`` (None: any), else None:
    the planned store asked only for plans of carry at most ``ceiling - 1``, its end's entries alone kept."""
    store = planned_store(sizes, live, C, r, qmax, forced, segment_cap=segment_cap, operand=operand,
                          upper=None if ceiling is None else int(ceiling) - 1, keep=False)
    if store is None:
        return None
    g = store.best()
    if g is None or (ceiling is not None and int(store.carry[g]) >= int(ceiling)):
        return None
    return int(store.carry[g]), int(store.phases[g])
