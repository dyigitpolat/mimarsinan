"""[U-BRANGE2] The range search's per-call arithmetic (``range_search`` holds the search and states its exactness).

Each call's values are sorted once with float64 prefix sums of the values and their squares (a call's transient,
released after it). A candidate window ``[a, a + s]`` has ``levels`` levels of step ``s / levels``; its error on a call
is its clamp part (the values below ``a`` and above ``a + s``, one rank each, each departing by its excess) plus its
rounding part (the values inside, one rank per cell edge, each cell read from the prefix sums, each cell's squared
error taken at least 0 and the cells summed in code order), read in blocks of at most ``_BLOCK_CELLS`` (candidate,
cell) pairs. A candidate's error is never below its clamp part as computed, so in the second pass a call evaluates only
the candidates whose lower bound (the calls before, this call's clamp part, and each later call's excess of its
extremes over the window) is within a generous upper bound of the least error; under the full search every candidate
is evaluated on every call (``_Site.full``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from mimarsinan.mapping.support.range_bounds import Histogram

#: (candidate, cell) pairs evaluated together: the transient's bound, whatever the candidates and levels.
_BLOCK_CELLS = 1 << 18


class _Sorted:
    """One call's values sorted, with float64 prefix sums of the values and of their squares (a zero first): a call's
    transient. The sorted copy is the rule's own; the input is only read."""

    def __init__(self, values: torch.Tensor) -> None:
        # [third form] the search's arithmetic runs on the host whatever device the values come from: one arithmetic
        # (the same range for the same values on any device), and no device round trip per candidate block.
        flat = values.detach().reshape(-1).cpu()
        if not flat.is_floating_point():
            flat = flat.float()
        self.values = torch.sort(flat).values
        self.n = self.values.numel()
        dev = self.values.device
        # the prefix sums built in their own buffers (20 bytes a value with the sorted copy, revision 1's footprint)
        self.first = torch.zeros(self.n + 1, dtype=torch.float64, device=dev)
        self.second = torch.zeros(self.n + 1, dtype=torch.float64, device=dev)
        self.first[1:].copy_(self.values)
        torch.mul(self.first[1:], self.first[1:], out=self.second[1:])
        self.first[1:].cumsum_(0)
        self.second[1:].cumsum_(0)

    def _rank(self, at: torch.Tensor, above: bool) -> torch.Tensor:
        """Values strictly below ``at`` (``above`` False) or at most ``at`` (True), compared in the values' dtype."""
        dt, dev = self.values.dtype, self.values.device
        edge = at.to(dt)
        if above:
            edge = torch.nextafter(edge, torch.tensor(float("inf"), dtype=dt))
        return torch.searchsorted(self.values, edge.to(dev))

    def _tail(self, rank: torch.Tensor, at: torch.Tensor, upper: bool) -> torch.Tensor:
        """The squared excess of the values beyond ``at`` (above it from ``rank`` up, or below it up to ``rank``)."""
        if upper:
            n = (self.n - rank).to(torch.float64)
            s1, s2 = self.first[-1] - self.first[rank], self.second[-1] - self.second[rank]
        else:
            n = rank.to(torch.float64)
            s1, s2 = self.first[rank], self.second[rank]
        return (s2 - 2.0 * at * s1 + at * at * n).clamp(min=0.0)

    def low_parts(self, lows: torch.Tensor):
        """Per low end: the count below it and its clamp part (every value below it departing by its excess)."""
        a = lows.to(self.values.device)
        below = self._rank(lows, above=False)
        return below, self._tail(below, a, upper=False).cpu()

    def top_parts(self, low: float, spans: torch.Tensor):
        """Per span above ``low``: the count at or below the top, the clamped count, the clamped squared magnitude
        and the upper clamp part."""
        tops = (low + spans) if low != 0.0 else spans
        kept = self._rank(tops, above=True)
        t = tops.to(self.values.device)
        clamped = (self.n - kept).to(torch.float64)
        return kept, clamped.cpu(), (self.second[-1] - self.second[kept]).cpu(), self._tail(kept, t, upper=True).cpu()

    def rounding_parts(self, low: float, below: int, spans: torch.Tensor, levels: int, kept: torch.Tensor):
        """Per span of a block: the rounding part (every value from ``low`` to the top, in its code's cell), each cell's
        squared error at least 0, the cells summed in code order."""
        dev, dt = self.values.device, self.values.dtype
        s = spans.to(dev)
        codes = torch.arange(1, levels + 1, dtype=torch.float64, device=dev)
        edges = (codes - 0.5)[None, :] * (s / levels)[:, None]
        if low != 0.0:
            edges = low + edges
        ranks = torch.searchsorted(self.values, edges.to(dt).reshape(-1)).reshape(edges.shape)
        start = torch.full((len(s), 1), int(below), dtype=ranks.dtype, device=dev)
        bounds = torch.cat((start, torch.maximum(ranks, start), kept[:, None]), dim=1)
        lo, hi = bounds[:, :-1], bounds[:, 1:]
        count = (hi - lo).to(torch.float64)
        s1 = self.first[hi] - self.first[lo]
        s2 = self.second[hi] - self.second[lo]
        value = torch.arange(levels + 1, dtype=torch.float64, device=dev)[None, :] * (s / levels)[:, None]
        if low != 0.0:
            value = low + value
        cells = (s2 - 2.0 * value * s1 + count * value * value).clamp(min=0.0)
        return torch.cumsum(cells, dim=1)[:, -1].cpu()


@dataclass
class _Low:
    """One low end's candidates on a site: its spans and their running sums over the calls (the lower bound, every
    call's error where it was evaluated, else its clamp part, else its low clamp part alone; the error over the calls
    where evaluated on every one so far), and the census sums."""

    low: float
    spans: torch.Tensor
    bound: torch.Tensor
    error: torch.Tensor
    full: torch.Tensor
    clamped: torch.Tensor
    clamp_energy: torch.Tensor
    calls: int = 0


@dataclass
class _Site:
    """One site's search state over its calls (every call holding values: an empty call is no call). ``candidates``
    maps a window's widest span to its spans; ``full`` makes every candidate evaluated on every call."""

    spec: object
    minima: List[float]
    maxima: List[float]
    counts: List[int]
    candidates: Callable[[float], torch.Tensor]
    lows: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.float64))
    low_only: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.float64))
    by_low: Dict[int, _Low] = field(default_factory=dict)
    energy: float = 0.0
    entries: int = 0
    full: bool = False
    #: [third form] A signed site's second pass reads a histogram (``range_bounds``), its windows bounded at once.
    histogram: Optional[Histogram] = None

    @property
    def levels(self) -> int:
        return int(getattr(self.spec, "levels"))

    def spans_for(self, low: float) -> torch.Tensor:
        return self.candidates(max(self.maxima) - low)

    def state(self, i: int, call: int = 0) -> _Low:
        """Low end ``i``'s state, made at ``call``: the calls before it read only its low clamp part (``low_only``),
        so no span of it counts as evaluated on every call."""
        if i not in self.by_low:
            low = float(self.lows[i])
            spans = self.spans_for(low)
            c = len(spans)
            self.by_low[i] = _Low(low, spans, torch.full((c,), float(self.low_only[i]), dtype=torch.float64),
                                  torch.zeros(c, dtype=torch.float64), torch.full((c,), call == 0, dtype=torch.bool),
                                  torch.zeros(c, dtype=torch.float64), torch.zeros(c, dtype=torch.float64), calls=call)
        return self.by_low[i]


def _later_excess(site: _Site, call: int, low: float, spans: torch.Tensor) -> torch.Tensor:
    """The first pass's bound on the clamp part of every later call: its extremes' excess beyond the window."""
    out = torch.zeros_like(spans)
    for c in range(call + 1, len(site.maxima)):
        out = out + (site.maxima[c] - (low + spans)).clamp(min=0.0) ** 2
        if site.minima[c] < low:
            out = out + (low - site.minima[c]) ** 2
    return out


def _upper(site: _Site, call: int, top_error: float) -> float:
    """A generous upper bound of the least error over every call: the widest window's error on the calls so far, and
    four times its rounding bound on each later one; infinite under the full search (nothing skipped)."""
    if site.full:
        return math.inf
    wide = max(site.maxima) - float(site.lows[-1])
    upper = top_error
    for c in range(call + 1, len(site.maxima)):
        upper += site.counts[c] * (wide / site.levels) ** 2
    return upper


def _selection(site: _Site, call: int, low_index: int, bound: torch.Tensor, clamp: torch.Tensor,
               upper: float) -> torch.Tensor:
    """The spans of one low end evaluated on one call of the second pass: those whose lower bound is within the
    upper bound, and the two ends of the search (a selection, never the rule: the final check reads what it skips)."""
    state = site.by_low[low_index]
    chosen = bound + clamp + _later_excess(site, call, state.low, state.spans) <= upper
    if low_index == 0:
        chosen[0] = chosen[-1] = True
    if low_index == len(site.lows) - 1:
        chosen[-1] = True
    return chosen


def _evaluate_low(site: _Site, batch: _Sorted, below: torch.Tensor, low_part: torch.Tensor, li: int,
                  chosen: Optional[torch.Tensor], upper: Optional[float], call: int, errors: Optional[dict]):
    """One low end on one call: the clamp parts of all its spans, the rounding parts of the chosen ones; the running
    sums updated (second pass), or the chosen spans' errors returned per call (third pass, ``errors``)."""
    state = site.state(li, call)
    kept, clamped, energy, upper_part = batch.top_parts(state.low, state.spans)
    clamp = (float(low_part[li]) + upper_part)
    if chosen is None:
        chosen = _selection(site, call, li, state.bound, clamp, upper if upper is not None else math.inf)
    idx = torch.nonzero(chosen).reshape(-1)
    error = torch.full((len(state.spans),), float("nan"), dtype=torch.float64)
    block = max(1, _BLOCK_CELLS // (site.levels + 2))
    start_rank = int(below[li])
    for first in range(0, len(idx), block):
        sel = idx[first:first + block]
        error[sel] = clamp[sel] + batch.rounding_parts(state.low, start_rank, state.spans[sel], site.levels,
                                                       kept[sel.to(kept.device)])
    if errors is not None:
        errors[li] = (error, clamped, energy)
        return
    seen = ~torch.isnan(error)
    state.bound = state.bound + torch.where(seen, error, clamp)
    state.full = state.full & seen
    state.error = state.error + torch.where(seen, error, torch.zeros_like(error))
    state.clamped += clamped
    state.clamp_energy += energy
    state.calls += 1


def _call(site: _Site, values: torch.Tensor, call: int, wanted: Optional[Dict[int, torch.Tensor]],
          errors: Optional[dict]) -> None:
    """One call of the second pass (``wanted`` None) or of the third (exactly the open spans of the open low ends).
    The caller never hands an empty call: a call with no values is no call (``range_search``)."""
    batch = _Sorted(values)
    below, low_part = batch.low_parts(site.lows)
    if wanted is not None:
        if site.histogram is not None:  # a bounded site's census sums, read here (its second pass read a histogram)
            site.energy += float(batch.second[-1])
            site.entries += batch.n
        assert errors is not None, "the third pass hands its error table with its windows"
        for li, chosen in wanted.items():
            _evaluate_low(site, batch, below, low_part, li, chosen, None, call, errors.setdefault(call, {}))
        return
    site.energy += float(batch.second[-1])
    site.entries += batch.n
    widest = len(site.lows) - 1
    top_state = site.state(widest)
    kept, _c, _e, up = batch.top_parts(top_state.low, top_state.spans[-1:])
    top_error = float(low_part[widest] + up[0]) + float(batch.rounding_parts(
        top_state.low, int(below[widest]), top_state.spans[-1:], site.levels, kept)[0])
    upper = _upper(site, call, float(top_state.error[-1]) + top_error)
    for li in range(len(site.lows)):
        later_low = sum((site.lows[li] - site.minima[c]).clamp(min=0.0).item() ** 2
                        for c in range(call + 1, len(site.minima)) if site.minima[c] < float(site.lows[li]))
        floor = float(site.low_only[li] if li not in site.by_low else site.by_low[li].bound.min())
        if li in site.by_low or li == 0 or li == widest or floor + float(low_part[li]) + later_low <= upper:
            _evaluate_low(site, batch, below, low_part, li, None, upper, call, None)
        else:
            site.low_only[li] = site.low_only[li] + low_part[li]
    for li, state in site.by_low.items():  # a low end skipped on this call adds its low clamp part to every span
        if state.calls < call + 1:
            state.bound = state.bound + low_part[li]
            state.full = torch.zeros_like(state.full)
            state.calls = call + 1
