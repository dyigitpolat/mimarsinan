"""[F3 round] The sliding window fronts of the store front (``residency_front_store.front_store``): a FIFO of positions'
entry blocks and the Pareto front of its content, merged by the store's joint dominance rule. [Fix round] Moved from
``residency_front_store`` unchanged, beside the rule that picks the window scan where the window is short."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_BIG = np.iinfo(np.int64).max

#: [Fix round] The longest mean phase window, in positions (len(sizes) x C / F), whose store ``front_store`` builds by
#: the window scan, without and with a segment cap: there one Pareto pass over the window's entries per position costs
#: less than the sliding fronts' merges, whose fine rule (a segment cap) costs more (MEASURED on 400 random programs of
#: 20 to 4,000 units, 11_wave2/resid/report.md, Fix Round). Either build is the same store, entry for entry.
SCAN_WINDOW, SCAN_WINDOW_CAPPED = 64, 256


def scans(sizes: Sequence[int], C: int, capped: bool = False) -> bool:
    """Whether the store is built by the window scan: the mean window, len(sizes) x C / F positions, is at most
    ``SCAN_WINDOW`` (``SCAN_WINDOW_CAPPED`` under a segment cap)."""
    limit = SCAN_WINDOW_CAPPED if capped else SCAN_WINDOW
    return len(sizes) * int(C) <= limit * max(1, sum(int(t) for t in sizes))


#: An entry block: rows key (the excess, or a far queue's excess less its position's prefix), carry, phases, the
#: segment's phases, position, entry index; one column per entry.
EMPTY = np.zeros((6, 0), dtype=np.int64)


def window_rule(block: np.ndarray, fine: bool) -> np.ndarray:
    """The entries of ``block`` no other one dominates: lower or equal in (key, carry, phases[, segment's phases])
    and earlier in (key, carry, phases, the later position[, segment's phases]). With ``fine`` (a segment cap) the
    order keeps each entry that either of the store's rules keeps at an end position (``_pareto``: by phases at a
    segment's end, with the segment's phases inside it), so either rule on the window's front is that rule on the
    window; the relation is transitive, so the window's front is an associative merge."""
    if block.shape[1] <= 1:
        return block
    k, c, q, s, pos = block[0], block[1], block[2], block[3], block[4]
    order = np.lexsort((s, -pos, q, c, k) if fine else (-pos, q, c, k))
    keep = pareto_keep(c[order], q[order], s[order] if fine else None)
    return block[:, order[keep]]


def pareto_keep(c: np.ndarray, q: np.ndarray, s: Optional[np.ndarray]) -> np.ndarray:
    """Over candidates already in order: which no earlier one dominates in carry, phases (and ``s``)."""
    keep = np.ones(c.size, dtype=bool)
    groups = np.unique(q)[:, None] if s is None else np.unique(np.stack([q, s]), axis=1).T
    for g in groups:
        below = q <= g[0] if s is None else (q <= g[0]) & (s <= g[1])
        mine = q == g[0] if s is None else (q == g[0]) & (s == g[1])
        prev = np.r_[_BIG, np.minimum.accumulate(np.where(below, c, _BIG))[:-1]]
        keep[mine] &= prev[mine] > c[mine]
    return keep


def merge(a: np.ndarray, b: np.ndarray, fine: bool) -> np.ndarray:
    if not a.shape[1] or not b.shape[1]:
        return b if not a.shape[1] else a
    return window_rule(np.concatenate([a, b], axis=1), fine)


class WindowFront:
    """A FIFO of positions' entry blocks and the front of its content, kept by two stacks of merged fronts: each
    block is merged into the back's front once and into one suffix front of the front stack, amortized O(1)."""

    def __init__(self, fine: bool):
        self.fine, self.back, self.back_front, self.front = fine, [], EMPTY, []

    def push(self, block: np.ndarray) -> None:
        self.back.append(block)
        self.back_front = merge(self.back_front, block, self.fine)

    def pop(self) -> None:
        if not self.front:
            acc = EMPTY
            for block in reversed(self.back):
                acc = merge(block, acc, self.fine)
                self.front.append(acc)
            self.back, self.back_front = [], EMPTY
        self.front.pop()

    def query(self) -> np.ndarray:
        return merge(self.front[-1] if self.front else EMPTY, self.back_front, self.fine)
