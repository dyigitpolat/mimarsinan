"""[U-BRANGE2] The range rule's search: exact, with a transient bounded independent of candidates times levels.

A site's candidate grid is a window ``[a, a + s]`` with ``levels`` levels of step ``s / levels``: ``a`` is 0 on a site
of magnitudes (the rule's every value grid), or a low end below 0 on a signed site (a signed seam's window). The spans
are ``RANGE_CANDIDATES_PER_OCTAVE`` geometric values per octave over ``RANGE_OCTAVES`` octaves below the largest value
above ``a``; a signed site's low ends are 0 and the same candidates on its negative side. The per-call arithmetic is
``range_calls``'s; a signed site's windows are bounded at once from one histogram (``range_bounds``).

Three passes over the site's source: the first reads each call's extremes and count; the second evaluates the
candidates that can be the least (a signed site: fills its histogram); the final check reads, in a third pass on every
call, any candidate not evaluated everywhere whose summed lower bound is within the arithmetic's margin of the least
error found. A call with no values is no call: it is read by no pass, so it changes neither the choice nor the cost.

What "exact" means, the proven property: the window chosen and its census are the full search's bit for bit, where
the full search (``full=True``) is this same arithmetic with every candidate (every low end and every span) read on
every call; ties go to the least span, then to the low end nearest 0. The error values are that arithmetic's: each
cell's squared error is a difference of float64 prefix sums over every smaller value, so a census error departs from
the grid's own error read value by value by cancellation, measured at most 6.7e-13 of the site's squared magnitude
(``ERROR_RESOLUTION`` states 1e-11), which on a tight cluster far from 0 is a few percent of that cluster's own error.
[U-BRANGE2 stage 1b] The resolution's scope: float32 and float64 values. Each cell edge is read in the values' own
dtype, so on a float16 seam the census departed from the grid's own error by 1.47e-7 of the squared magnitude (the
stage verify's m-8); a site whose values are narrower than float32 is refused by name (``RangeSearchError``).

Cost law (per call of n values, C candidates, L levels): O(n log n) to sort, O(C log n) for every clamp part, and
O(S (L + 2) log n) for the S candidates evaluated; S is C at small L and shrinks as L grows, since a candidate's clamp
part prunes it once it exceeds the least error, which falls as L squared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.support.range_bounds import Histogram, open_windows
from mimarsinan.mapping.support.range_calls import _BLOCK_CELLS, _Low, _Site, _Sorted, _call

__all__ = ["ERROR_RESOLUTION", "RANGE_CANDIDATES", "RANGE_CANDIDATES_PER_OCTAVE", "RANGE_OCTAVES", "RangeCensus",
           "RangeSearchError", "SiteSpec", "search_ranges", "whole_interval_candidates"]

#: The search: geometric candidates per octave below the largest value, over this many octaves.
RANGE_CANDIDATES_PER_OCTAVE = 256
RANGE_OCTAVES = 12
RANGE_CANDIDATES = RANGE_CANDIDATES_PER_OCTAVE * RANGE_OCTAVES + 1
#: The final check's relative margin: float64 sums of non-negative terms over a few hundred calls are exact to far less.
_MARGIN = 1e-9
#: The resolution of a census error relative to the site's squared magnitude (the prefix sums' cancellation).
ERROR_RESOLUTION = 1e-11
#: [third form] A signed site's windows are bounded at once from its histogram (``range_bounds``).
SIGNED_BY_BOUNDS = True
_EXPORTED = (_BLOCK_CELLS, _Sorted)  # the per-call arithmetic, read by the tests through this module


class RangeSearchError(ToolchainRefusal, RuntimeError):
    """A site the search refuses by name: its grid's error minimum lies at the search's low end (never returned
    silently), or its values are narrower than float32 (the stated resolution's scope)."""


@dataclass(frozen=True)
class SiteSpec:
    """A site's grid: its levels, and whether its low end is searched (a signed site reads its values themselves)."""

    levels: int
    signed: bool = False


@dataclass(frozen=True)
class RangeCensus:
    """A site's chosen range and its clamp census over the calibration batches: the entries outside the window, their
    share of the squared magnitude, the largest magnitude over the span, and the grid's squared error (the search's
    arithmetic, module docstring) at the chosen window and at both ends of the search (``low_span``: the search's
    least span). ``low`` is the window's low end (0 on a site of magnitudes)."""

    scale: float
    low_span: float
    maximum: float
    candidates: int
    entries: int
    clamped: int
    clamped_share: float
    largest_over_scale: float
    squared_error: float
    low_span_error: float
    maximum_error: float
    low: float = 0.0
    minimum: float = 0.0

    @property
    def clamped_fraction(self) -> float:
        return self.clamped / self.entries if self.entries else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"scale": self.scale, "low_span": self.low_span, "maximum": self.maximum,
                "candidates": self.candidates, "entries": self.entries, "clamped": self.clamped,
                "clamped_fraction": self.clamped_fraction, "clamped_share": self.clamped_share,
                "largest_over_scale": self.largest_over_scale, "squared_error": self.squared_error,
                "low_span_error": self.low_span_error, "maximum_error": self.maximum_error, "low": self.low,
                "minimum": self.minimum}


def whole_interval_candidates(high: float) -> torch.Tensor:
    """The search's spans, ascending fp32 values in float64: ``high 2^(-k / per_octave)`` for ``k`` from
    ``octaves x per_octave`` down to 0, the maximum exact."""
    k = torch.arange(RANGE_OCTAVES * RANGE_CANDIDATES_PER_OCTAVE, -1, -1, dtype=torch.float64)
    scales = (float(high) * torch.pow(2.0, -k / RANGE_CANDIDATES_PER_OCTAVE)).to(torch.float32).to(torch.float64)
    scales[-1] = float(torch.tensor(high, dtype=torch.float32))
    return torch.unique(scales)


def _open(site: _Site) -> Tuple[float, Dict[int, torch.Tensor]]:
    """The final check: the least error among candidates evaluated on every call, and per low end the spans not
    evaluated everywhere whose lower bound is within the margin of it (the low ends never evaluated included)."""
    best = math.inf
    for state in site.by_low.values():
        if bool(state.full.any()):
            best = min(best, float(state.error[state.full].min()))
    limit = best * (1.0 + _MARGIN)
    wanted: Dict[int, torch.Tensor] = {}
    for li in range(len(site.lows)):
        if li in site.by_low:
            state = site.by_low[li]
            open_ = (~state.full) & (state.bound <= limit)
        elif float(site.low_only[li]) <= limit:
            open_ = torch.ones(len(site.spans_for(float(site.lows[li]))), dtype=torch.bool)
        else:
            continue
        if bool(open_.any()):
            wanted[li] = open_
    return best, wanted


def _bounded_windows(site: _Site) -> Dict[int, torch.Tensor]:
    """A signed site's open windows from its histogram's bounds, each one's state made unread (no call read yet, its
    bound unknown until the third pass); every other low end's bound infinite (closed by the bounds)."""

    histogram = site.histogram
    assert histogram is not None, "a bounded site reads its histogram"
    last = len(site.lows) - 1
    forced = [(0, 0), (0, len(site.spans_for(float(site.lows[0]))) - 1),
              (last, len(site.spans_for(float(site.lows[last]))) - 1)]
    wanted = open_windows(histogram, site.lows, lambda i: site.spans_for(float(site.lows[i])),
                          site.levels, forced)
    site.low_only = torch.full((len(site.lows),), math.inf, dtype=torch.float64)
    for li in wanted:
        spans = site.spans_for(float(site.lows[li]))
        c = len(spans)
        site.by_low[li] = _Low(float(site.lows[li]), spans, torch.full((c,), math.inf, dtype=torch.float64),
                               torch.zeros(c, dtype=torch.float64), torch.zeros(c, dtype=torch.bool),
                               torch.zeros(c, dtype=torch.float64), torch.zeros(c, dtype=torch.float64))
    return wanted


def _third_pass(run_pass: Callable, sites: Dict[int, _Site], reopened: Dict[int, Dict[int, torch.Tensor]]) -> None:
    """Every open candidate read exactly on every call, its error summed in call order."""
    errors: Dict[int, dict] = {i: {} for i in reopened}
    seen = {i: 0 for i in reopened}

    def third(i: int, values: torch.Tensor) -> None:
        if values.numel():  # an empty call is no call
            _call(sites[i], values, seen[i], reopened[i], errors[i])
            seen[i] += 1

    run_pass(third, list(reopened))
    for i, per_call in errors.items():
        site = sites[i]
        for li, open_ in reopened[i].items():
            state = site.state(li)
            total = torch.zeros(len(state.spans), dtype=torch.float64)
            clamped = torch.zeros_like(total)
            energy = torch.zeros_like(total)
            for call in range(len(site.counts)):
                error, c_n, c_e = per_call[call][li]
                total = total + torch.nan_to_num(error, nan=0.0)
                clamped, energy = clamped + c_n, energy + c_e
            state.error = torch.where(open_, total, state.error)
            state.bound = torch.where(open_, total, state.bound)
            state.clamped = torch.where(open_, clamped, state.clamped)
            state.clamp_energy = torch.where(open_, energy, state.clamp_energy)
            state.full = state.full | open_


def _census(i: int, site: _Site, hi: float, lo: float) -> RangeCensus:
    """The site's choice by the final check, ties to the least span, then to the low end nearest 0."""
    best, still = _open(site)
    if still:
        raise RuntimeError(f"range site {i}: a candidate stayed open after the third pass")
    choice = None
    for li in sorted(site.by_low, key=lambda k: (abs(float(site.lows[k])), k)):
        state = site.by_low[li]
        hits = torch.nonzero(state.full & (state.error == best)).reshape(-1)
        if len(hits):
            cand = (float(state.spans[int(hits[0])]), abs(state.low), li, int(hits[0]))
            choice = cand if choice is None or cand < choice else choice
    assert choice is not None, "the zero low end's spans are read on every call"
    span, _absl, li, si = choice
    state = site.by_low[li]
    zero = site.by_low[0]
    if li == 0 and si == 0 and len(state.spans) > 1:
        raise RangeSearchError(
            f"range site {i}: the grid's error minimum lies at the search's low end {float(state.spans[0]):.6g} "
            f"({RANGE_OCTAVES} octaves below the calibration maximum {hi:.6g}): its values span more octaves "
            "than the search covers")
    return RangeCensus(
        scale=span, low_span=float(zero.spans[0]), maximum=hi, candidates=len(zero.spans),
        entries=site.entries, clamped=int(state.clamped[si]),
        clamped_share=float(state.clamp_energy[si]) / site.energy if site.energy > 0.0 else 0.0,
        largest_over_scale=(hi - state.low) / span, squared_error=float(state.error[si]),
        low_span_error=float(zero.error[0]), maximum_error=float(zero.error[-1]), low=state.low, minimum=lo)


def search_ranges(run_pass: Callable, specs: Sequence[SiteSpec], *, full: bool = False) -> List[RangeCensus]:
    """The rule's search for every site of a source (``range_rule.least_squares_recorded``). ``full`` reads every
    candidate on every call (the reference the exactness property names; the choice is the same, the cost is not)."""
    calls: List[List[Tuple[float, float, int]]] = [[] for _ in specs]
    narrow: Dict[int, torch.dtype] = {}

    def first(i: int, values: torch.Tensor) -> None:
        v = values.detach()
        if v.numel():  # an empty call is no call
            calls[i].append((float(v.min()), float(v.max()), v.numel()))
            if v.is_floating_point() and torch.finfo(v.dtype).bits < 32:
                narrow.setdefault(i, v.dtype)

    run_pass(first, None)
    if narrow:
        dtypes = sorted({str(d).replace("torch.", "") for d in narrow.values()})
        raise RangeSearchError(
            f"range site(s) {sorted(narrow)}: values of {dtypes}, narrower than float32: the search reads each cell "
            f"edge in its values' own dtype and states its resolution ({ERROR_RESOLUTION:g} of the site's squared "
            "magnitude) for float32 and float64 values only; calibrate the site in float32")
    sites: Dict[int, _Site] = {}
    for i, (spec, site_calls) in enumerate(zip(specs, calls)):
        lo = min((c[0] for c in site_calls), default=0.0)
        hi = max((c[1] for c in site_calls), default=0.0)
        lows = torch.zeros(1, dtype=torch.float64)
        if spec.signed and lo < 0.0:
            lows = torch.cat((lows, -whole_interval_candidates(-lo)))
        if spec.levels > 0 and hi - float(lows[-1]) > 0.0 and (spec.signed or hi > 0.0):
            sites[i] = _Site(spec, [c[0] for c in site_calls], [c[1] for c in site_calls], [c[2] for c in site_calls],
                             whole_interval_candidates, lows=lows, low_only=torch.zeros(len(lows), dtype=torch.float64),
                             full=full)
            if len(lows) > 1 and SIGNED_BY_BOUNDS and not full:  # [third form] low ends times spans: bounded at once
                sites[i].histogram = Histogram(lo, hi)
    seen = {i: 0 for i in sites}

    def second(i: int, values: torch.Tensor) -> None:
        if i in sites and values.numel():  # an empty call is no call
            histogram = sites[i].histogram
            if histogram is not None:
                histogram.add(values)
            else:
                _call(sites[i], values, seen[i], None, None)
            seen[i] += 1

    if sites:
        run_pass(second, list(sites))
    reopened = {i: _open(s)[1] for i, s in sites.items() if s.histogram is None}
    reopened.update({i: _bounded_windows(s) for i, s in sites.items() if s.histogram is not None})
    reopened = {i: w for i, w in reopened.items() if w}
    if reopened:
        _third_pass(run_pass, sites, reopened)
    out: List[RangeCensus] = []
    for i, spec in enumerate(specs):
        hi = max((c[1] for c in calls[i]), default=0.0)
        lo = min((c[0] for c in calls[i]), default=0.0)
        site = sites.get(i)
        if site is None:  # never reached (maximum 0), or a grid without levels: unsearched
            out.append(RangeCensus(0.0, 0.0, hi, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, minimum=lo))
            continue
        out.append(_census(i, site, hi, lo))
    return out
