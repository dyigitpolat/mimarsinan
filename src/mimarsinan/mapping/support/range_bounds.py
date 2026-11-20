"""[U-BRANGE2, third form] Bounds of every window's error over all calls at once, from one histogram of a site's values.

A signed site (a seam's window: its low end below 0 searched with its span) has the low ends times the spans as
candidates (3,074 x 3,073 windows at the rule's interval). Evaluating each window on every call costs their number
times the calls: the second form took 99 s for one seam-shaped site of four calls of 2.2M values at 15 levels. Here the
search's second pass reads each call into a histogram of ``BINS`` bins over the site's extremes (per bin the count, the
sum and the sum of squares in float64, the least and the largest value), and every window's error over all calls is
bounded at once:

* a bin whose values all take one grid value (one cell, or the clamp below the low end, or above the top) contributes
  its squared departure from that value exactly (``S2 - 2 q S1 + q^2 n``), summed over runs of such bins by prefix sums;
* a bin holding a cell edge contributes between its count times the least and the largest squared distance to the grid
  over its value interval (the distance is piecewise linear: read at the interval's ends, 0 at a grid value inside it,
  at most half a step plus the edge's rounding at the edge).

The cell edges and the top are the search's own floats (``range_search._Sorted``: edges rounded to the values' dtype,
a value at an edge in the upper cell, a value at the top inside the window), so every bound holds the error the search
computes up to float64's rounding, which ``Histogram.margin`` bounds. Only the windows whose lower bound is within twice
the margin of the least upper bound stay open; the search reads them exactly on every call, so its choice is the full
search's. The bounds run in three stages: a coarse lattice's full bounds give an upper bound of the least error, the
clamp parts alone (two ranks a window, whatever the levels) close every window above it, and the windows left take
full bounds.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

#: Bins of a signed site's histogram over its extremes (the bounds' resolution: a bin holding a cell edge is the gap).
BINS = 1 << 16
#: (window, boundary) pairs bounded together: the transient's bound, whatever the windows and levels.
_BLOCK_CELLS = 1 << 18
#: The coarse lattice's stride in low ends and in spans.
_COARSE = 16
_EPS = 2.0 ** -52


class Histogram:
    """One signed site's values over the calls: per bin of ``BINS`` over ``[lo, hi]`` the count, the sum and the sum of
    squares (float64, each call summed into its own buffer, then added in call order), the least and the largest value;
    and the dtype the values arrive in (the search compares its edges in it)."""

    def __init__(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(lo), float(hi)
        self.width = (self.hi - self.lo) / BINS if self.hi > self.lo else 1.0
        # the bins live on the host, as the search's own arithmetic (range_search._Sorted)
        self.count, self.first, self.second = (torch.zeros(BINS, dtype=torch.float64) for _ in range(3))
        self.least = torch.full((BINS,), math.inf, dtype=torch.float64)
        self.largest = torch.full((BINS,), -math.inf, dtype=torch.float64)
        self.dtype: Optional[torch.dtype] = None
        self.mixed = False
        self.calls = self.largest_call = self.values = 0

    def add(self, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1).cpu()
        if not flat.is_floating_point():
            flat = flat.float()
        if flat.numel() == 0:
            return
        if self.dtype is None:
            self.dtype = flat.dtype
        elif flat.dtype != self.dtype:
            self.mixed = True
        v = flat.to(device=self.count.device, dtype=torch.float64)
        index = torch.floor((v - self.lo) / self.width).clamp_(0, BINS - 1).long()
        for total, term in ((self.count, torch.ones_like(v)), (self.first, v), (self.second, v * v)):
            total += torch.zeros_like(total).index_add_(0, index, term)
        self.least = torch.minimum(self.least, torch.full_like(self.least, math.inf).scatter_reduce_(
            0, index, v, "amin", include_self=True))
        self.largest = torch.maximum(self.largest, torch.full_like(self.largest, -math.inf).scatter_reduce_(
            0, index, v, "amax", include_self=True))
        self.calls += 1
        self.largest_call = max(self.largest_call, v.numel())
        self.values += v.numel()

    def margin(self) -> float:
        """A bound of float64's rounding in a window's bounds and in the search's own error of it: every term of an
        error is at most ``4 M^2`` a value (``M`` the largest magnitude), and each prefix sum, bin sum or sum over the
        calls rounds by at most its length times eps of the terms it adds."""
        m = max(abs(self.lo), abs(self.hi))
        return 16.0 * _EPS * (self.largest_call + BINS + self.calls + 8) * self.values * 4.0 * m * m


class _Bins:
    """The histogram's non-empty bins, value-ordered (a bin's values all precede the next bin's), with prefix sums."""

    def __init__(self, h: Histogram) -> None:
        keep = h.count > 0
        self.n, s1, s2 = h.count[keep], h.first[keep], h.second[keep]
        self.lo, self.hi = h.least[keep].contiguous(), h.largest[keep].contiguous()
        z = torch.zeros(1, dtype=torch.float64, device=self.n.device)
        self.pn, self.p1, self.p2 = (torch.cat((z, x.cumsum(0))) for x in (self.n, s1, s2))
        assert h.dtype is not None, "a histogram is bounded only after a call with values (open_windows)"
        self.dtype = h.dtype
        self.round = 2.0 * torch.finfo(h.dtype).eps  # an edge's rounding to the values' dtype, relative
        self.size = int(self.n.numel())

    def run(self, start: torch.Tensor, end: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """The exact squared departure from ``q`` of the bins ``[start, end)`` (none where end <= start), at least 0."""
        end = torch.maximum(end, start)
        n = self.pn[end] - self.pn[start]
        s1 = self.p1[end] - self.p1[start]
        s2 = self.p2[end] - self.p2[start]
        return (s2 - 2.0 * q * s1 + q * q * n).clamp(min=0.0)


def _top(b: _Bins, lows: torch.Tensor, spans: torch.Tensor):
    """The search's top (``low + span``, or the span at low 0) and the first value above it in the values' dtype."""
    top = torch.where(lows != 0.0, lows + spans, spans)
    inf = torch.tensor(math.inf, dtype=b.dtype, device=spans.device)
    return top, torch.nextafter(top.to(b.dtype), inf).to(torch.float64)


def _clamp_lower_bounds(b: _Bins, lows: torch.Tensor, spans: torch.Tensor, levels: int) -> torch.Tensor:
    """Per window: the bins wholly below its first cell edge, departing from the low end, plus the bins wholly above its
    top, departing from the top (exact; a lower bound of the window's error, every other part being at least 0)."""
    step = spans / levels
    first = torch.where(lows != 0.0, lows + 0.5 * step, 0.5 * step).to(b.dtype).to(torch.float64)
    top, above_top = _top(b, lows, spans)
    below = torch.searchsorted(b.hi, first.contiguous())
    above = torch.searchsorted(b.lo, above_top.contiguous())
    return b.run(torch.zeros_like(below), below, lows) + b.run(above, torch.full_like(above, b.size), top)


def _full_bounds(b: _Bins, lows: torch.Tensor, spans: torch.Tensor, levels: int):
    """Per window: (lower, upper) bounds of its error over every call."""
    step = spans / levels
    codes = torch.arange(1, levels + 1, dtype=torch.float64, device=spans.device)
    edges = (codes - 0.5)[None, :] * step[:, None]
    edges = torch.where((lows != 0.0)[:, None], lows[:, None] + edges, edges)
    top, above_top = _top(b, lows, spans)
    bounds = torch.cat((edges.to(b.dtype).to(torch.float64), above_top[:, None]), dim=1).contiguous()
    values = torch.arange(levels + 1, dtype=torch.float64, device=spans.device)[None, :] * step[:, None]
    values = torch.where((lows != 0.0)[:, None], lows[:, None] + values, values)
    grid = torch.cat((values, top[:, None]), dim=1)                       # region r's grid value, r = 0 .. levels + 1
    w, k = bounds.shape
    rank = torch.searchsorted(b.hi, bounds)                               # first bin whose largest is at the boundary
    at = rank.clamp(max=max(b.size - 1, 0))
    straddle = (rank < b.size) & (b.lo[at] < bounds)                      # that bin holds values on both sides
    start = torch.cat((torch.zeros(w, 1, dtype=rank.dtype, device=rank.device), rank + straddle.long()), dim=1)
    end = torch.cat((rank, torch.full((w, 1), b.size, dtype=rank.dtype, device=rank.device)), dim=1)
    exact = b.run(start.reshape(-1), end.reshape(-1), grid.reshape(-1)).reshape(w, k + 1).sum(dim=1)
    again = torch.zeros_like(straddle)                                    # a bin over several boundaries counted once
    again[:, 1:] = straddle[:, :-1] & (rank[:, 1:] == rank[:, :-1])
    counted = straddle & ~again
    lo_v, hi_v, n = b.lo[at], b.hi[at], b.n[at]
    r_lo = torch.searchsorted(bounds, lo_v.contiguous(), right=True)      # each end's region in its window
    r_hi = torch.searchsorted(bounds, hi_v.contiguous(), right=True)
    q_lo, q_hi = grid.gather(1, r_lo), grid.gather(1, r_hi)
    d_lo, d_hi = (lo_v - q_lo).abs(), (hi_v - q_hi).abs()
    one_edge = (r_hi == r_lo + 1) & (r_hi <= levels)                      # one cell edge, not the top's boundary
    no_grid_inside = (q_lo < lo_v) & (q_hi > hi_v)
    d_min = torch.where(one_edge & no_grid_inside, torch.minimum(d_lo, d_hi), torch.zeros_like(d_lo))
    half = (step / 2.0 + (lows.abs() + top.abs()) * b.round)[:, None]
    d_max = torch.maximum(torch.maximum(d_lo, d_hi), half)
    zero = torch.zeros_like(d_min)
    lower = exact + torch.where(counted, n * d_min * d_min, zero).sum(dim=1)
    upper = exact + torch.where(counted, n * d_max * d_max, zero).sum(dim=1)
    return lower, upper


def open_windows(h: Histogram, lows: torch.Tensor, spans_of: Callable[[int], torch.Tensor], levels: int,
                 forced: Sequence[Tuple[int, int]]) -> Dict[int, torch.Tensor]:
    """{low index: mask over its spans} of the windows that can be the search's least, the ``forced`` ones (the
    census's) always open; every window open when the calls' dtypes differ (no bound is read then)."""
    spans = [spans_of(i) for i in range(len(lows))]
    if h.dtype is None or h.mixed:
        return {i: torch.ones(len(s), dtype=torch.bool) for i, s in enumerate(spans)}
    dev = h.count.device
    b = _Bins(h)
    lows_d = lows.to(device=dev, dtype=torch.float64)
    spans_d = [s.to(dev) for s in spans]
    margin = 2.0 * h.margin()

    def blocks(index: List[Tuple[int, torch.Tensor]], per: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor,
                                                                                  torch.Tensor, torch.Tensor]]:
        """(low index, span index, low, span) vectors over ``index``'s windows, at most ``per`` a block."""
        li_parts, si_parts, size = [], [], 0
        for li, si in index:
            for first in range(0, len(si), per):
                part = si[first:first + per]
                li_parts.append(torch.full((len(part),), li, dtype=torch.long))
                si_parts.append(part)
                size += len(part)
                if size >= per:
                    yield _windows(li_parts, si_parts)
                    li_parts, si_parts, size = [], [], 0
        if size:
            yield _windows(li_parts, si_parts)

    def _windows(li_parts, si_parts):
        li_t, si_t = torch.cat(li_parts), torch.cat(si_parts)
        sv = torch.cat([spans_d[int(l[0])][s.to(dev)] for l, s in zip(li_parts, si_parts)])
        return li_t, si_t, lows_d[li_t.to(dev)], sv

    per_full = max(1, _BLOCK_CELLS // (levels + 2))
    coarse = [(i, torch.unique(torch.cat((torch.arange(0, len(s), _COARSE), torch.tensor([len(s) - 1])))))
              for i, s in enumerate(spans) if i % _COARSE == 0 or i == len(spans) - 1]
    least_upper = math.inf
    for _li, _si, lv, sv in blocks(coarse, per_full):
        least_upper = min(least_upper, float(_full_bounds(b, lv, sv, levels)[1].min()))
    kept: Dict[int, List[torch.Tensor]] = {}
    for li_t, si_t, lv, sv in blocks([(i, torch.arange(len(s))) for i, s in enumerate(spans)], _BLOCK_CELLS):
        stay = (_clamp_lower_bounds(b, lv, sv, levels) <= least_upper + margin).cpu()
        for l in torch.unique(li_t[stay]).tolist():
            kept.setdefault(int(l), []).append(si_t[stay & (li_t == l)])
    bounded = []
    for li_t, si_t, lv, sv in blocks([(l, torch.cat(p)) for l, p in sorted(kept.items())], per_full):
        lower, upper = _full_bounds(b, lv, sv, levels)
        least_upper = min(least_upper, float(upper.min()))
        bounded.append((li_t, si_t, lower.cpu()))
    out = {i: torch.zeros(len(s), dtype=torch.bool) for i, s in enumerate(spans)}
    for li_t, si_t, lower in bounded:
        stay = lower <= least_upper + margin
        for l in torch.unique(li_t[stay]).tolist():
            out[int(l)][si_t[stay & (li_t == l)]] = True
    for li, si in forced:
        out[li][si] = True
    return {i: m for i, m in out.items() if bool(m.any())}
