"""[U-BRANGE2, third form] A signed site's windows are bounded at once from one histogram, and only the windows that can
be the least are read exactly: the choice is the full search's, bit for bit, at a cost independent of low ends times
spans times calls.

The second form read every (low end, span) window on every call where its per-call bound let it: on a seam-shaped
site (four calls of 2.2M signed values, 15 levels) it read every low end's 3,073 spans on every call, 99 s for one
site, and S4's 54 seams did not finish an Activation Quantization install in 20 minutes. The third form's second pass
reads each call into a histogram; the bounds of every window's error over all calls close every window that cannot be
the least, and the third pass reads the open ones exactly.
"""

from __future__ import annotations

import pytest
import torch

try:  # each clause reads the engine itself, so a head without it fails clause by clause, never at collection
    import mimarsinan.mapping.support.range_search as engine
except ImportError:  # R5 (the 0.999 quantile) and revision 1 (one module, no search engine)
    engine = None


def _need_engine():
    if engine is None:
        pytest.fail("no range search engine on this head: no signed window is searched")


def _bounds():
    try:
        import mimarsinan.mapping.support.range_bounds as bounds
    except ImportError:
        pytest.fail("no window bounds on this head: a signed site reads every window on every call")
    return bounds


def _seam(seed: int, n: int, *, scale: float = 0.9, shift: float = 0.2, tail: float = 3.0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g) * scale + shift
    heavy = torch.rand(n, generator=g) < 0.01
    v[heavy] = v[heavy].abs() * tail
    return v


def _search(batches, levels: int):
    def run_pass(record, only):
        for b in batches:
            record(0, b)

    return engine.search_ranges(run_pass, [engine.SiteSpec(levels, signed=True)])[0]


def _full(batches, levels: int, monkeypatch):
    """The full search: the second form's arithmetic with every window read on every call (the engine's own,
    ``full=True``; [stage 1] a head without it fails here by name)."""
    import inspect

    del monkeypatch
    if "full" not in inspect.signature(engine.search_ranges).parameters:
        pytest.fail("the engine holds no full search of its own")

    def run_pass(record, only):
        for b in batches:
            record(0, b)

    return engine.search_ranges(run_pass, [engine.SiteSpec(levels, signed=True)], full=True)[0]


def _every_window_error(batches, levels):
    """Every window's error over the calls by the search's own arithmetic (``_Sorted``), summed in call order."""
    lo = min(float(b.min()) for b in batches)
    hi = max(float(b.max()) for b in batches)
    lows = torch.cat((torch.zeros(1, dtype=torch.float64), -engine.whole_interval_candidates(-lo)))
    spans = [engine.whole_interval_candidates(hi - float(a)) for a in lows]
    totals = [torch.zeros(len(s), dtype=torch.float64) for s in spans]
    for b in batches:
        batch = engine._Sorted(b)
        below, low_part = batch.low_parts(lows)
        for li, s in enumerate(spans):
            kept, _c, _e, up = batch.top_parts(float(lows[li]), s)
            totals[li] = totals[li] + (float(low_part[li]) + up + batch.rounding_parts(
                float(lows[li]), int(below[li]), s, levels, kept))
    return lo, hi, lows, spans, totals


@pytest.mark.parametrize("levels", [4, 15])
def test_every_window_s_bounds_hold_the_error_the_search_computes(levels, monkeypatch):
    _need_engine()
    bounds = _bounds()
    monkeypatch.setattr(engine, "RANGE_OCTAVES", 2)
    batches = [_seam(1, 1 << 12), _seam(2, 1 << 12)]
    batches[1][:64] = torch.round(batches[1][:64] * 4) / 4  # values on a lattice: many at cell edges and at bin edges
    lo, hi, lows, spans, totals = _every_window_error(batches, levels)
    h = bounds.Histogram(lo, hi)
    for b in batches:
        h.add(b)
    bins = bounds._Bins(h)
    margin = 2.0 * h.margin()
    for li in range(0, len(lows), 7):
        lv = torch.full((len(spans[li]),), float(lows[li]), dtype=torch.float64)
        lower, upper = bounds._full_bounds(bins, lv, spans[li], levels)
        clamp = bounds._clamp_lower_bounds(bins, lv, spans[li], levels)
        assert bool((lower - margin <= totals[li]).all()), li
        assert bool((totals[li] <= upper + margin).all()), li
        assert bool((clamp - margin <= totals[li]).all()), li


@pytest.mark.parametrize("levels", [4, 15])
@pytest.mark.parametrize("form", ["seam", "lattice", "far negative outlier", "late minimum", "one sided"])
def test_the_bounded_choice_is_the_full_search_s_bit_for_bit(form, levels, monkeypatch):
    """A guard: the second form is exact too (this clause holds on it); the third must keep it."""
    _need_engine()
    monkeypatch.setattr(engine, "RANGE_OCTAVES", 2)
    batches = [_seam(3, 1 << 13), _seam(4, 1 << 13)]
    if form == "lattice":
        batches = [torch.round(b * 2) / 2 for b in batches]
    elif form == "far negative outlier":
        batches[0][17] = -60.0
    elif form == "late minimum":
        batches[1][5] = -9.0
    elif form == "one sided":
        batches = [b.abs() + 0.5 for b in batches]
        batches[1][3] = -0.25
    row = _search(batches, levels)
    assert row == _full(batches, levels, monkeypatch)


def test_a_seam_shaped_signed_site_reads_few_windows_exactly(monkeypatch):
    _need_engine()
    read = []
    rounding = engine._Sorted.rounding_parts

    def counted(self, low, below, spans, levels, kept):
        read.append(len(spans))
        return rounding(self, low, below, spans, levels, kept)

    monkeypatch.setattr(engine._Sorted, "rounding_parts", counted)
    batches = [_seam(5 + c, 1 << 18) for c in range(4)]
    row = _search(batches, 15)
    assert row.low < 0.0
    assert sum(read) <= 4 * 50_000, f"{sum(read)} (window, call) reads exactly, of about 9.4M windows times 4 calls"
