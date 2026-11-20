"""[U-BRANGE2 stage 1, the review's m-6] What "exact" means, as the proven property, pinned.

The search's choice is exact: the window it returns, and its census, are the full search's bit for bit, where the full
search is the same arithmetic with every candidate (every low end and every span) read on every call. That is the
lemma of design2.md (Item 1) and the property every test of the search reads. The error values themselves are that
arithmetic's: each cell's squared error is a difference of float64 prefix sums over every smaller value, so it
departs from the grid's own error read value by value by cancellation (design2_review.md m-6: 2.96% of the offset
cluster's own error, 4e-4 at 32,767 levels on saturated and bimodal seams; measured here at most 6.7e-13 of the seam's
squared magnitude). The engine holds its full search as a reference of its own (``full=True``), and states the
resolution of its error values relative to the squared magnitude (``ERROR_RESOLUTION``).
"""

from __future__ import annotations

import inspect

import pytest
import torch

try:
    import mimarsinan.mapping.support.range_search as engine
    import mimarsinan.mapping.support.range_rule as range_rule
except ImportError:  # R5 and revision 1: no search engine
    engine = range_rule = None


def _need_full():
    if engine is None:
        pytest.fail("no range search engine on this head")
    if "full" not in inspect.signature(engine.search_ranges).parameters:
        pytest.fail("the engine holds no full search of its own: 'exact' names no reference a test can read")
    return engine


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _seams():
    b, d = 16, 4096
    out = {
        "offset cluster": ([1000.0 + 1.0e-3 * torch.randn(b, d, generator=_gen(800 + c)) for c in range(2)], False),
        "saturated": ([(2.0 + 3.0 * torch.randn(b, d, generator=_gen(900 + c))).clamp(0.0, 6.0) for c in range(2)],
                      False),
        "ties": ([torch.randint(0, 10, (b, d), generator=_gen(600 + c)).float() for c in range(2)], False),
        "constant": ([torch.ones(b, d) for _ in range(2)], False),
        "late outlier": ([torch.randn(b, d, generator=_gen(500)), torch.randn(b, d, generator=_gen(501))], False),
        "signed skew": ([1.0 + torch.randn(8, 2048, generator=_gen(2200 + c)) for c in range(3)], True),
        "signed ties": ([torch.randint(-4, 5, (8, 2048), generator=_gen(2500 + c)).float() for c in range(3)], True),
    }
    out["late outlier"][0][1][3, 11] = 1.0e4
    bimodal = []
    for c in range(2):
        x = 0.01 * torch.randn(b, d, generator=_gen(1700 + c))
        x[:, : d // 2] += 5.0
        bimodal.append(x)
    out["bimodal"] = (bimodal, False)
    return out


SEAMS = _seams()


def _search(calls, levels, signed, full=False):
    eng = _need_full()

    def run_pass(record, only):
        for c in calls:
            record(0, c if signed else c.abs())

    return eng.search_ranges(run_pass, [eng.SiteSpec(levels, signed=signed)], full=full)[0]


CASES = [(name, levels) for name, (_c, signed) in sorted(SEAMS.items()) if not signed for levels in (7, 127, 2047)]
# [U-BRANGE2 stage 1d, the check's n-2] a signed site's full search reads every low end with every span (55 s for
# signed skew at 15 levels on a sura worker at load 180, over the default 60 s budget under heavier load): the signed
# cases take a time budget of their own
SIGNED = [pytest.param(name, levels, marks=pytest.mark.timeout(240))
          for name, (_c, signed) in sorted(SEAMS.items()) if signed for levels in (7, 15)]


@pytest.mark.parametrize("name, levels", CASES + SIGNED)
def test_exact_means_the_choice_and_census_are_the_full_search_s_bit_for_bit(name, levels):
    calls, signed = SEAMS[name]
    assert _search(calls, levels, signed) == _search(calls, levels, signed, full=True)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("name", sorted(n for n, (_c, s) in SEAMS.items() if not s))
def test_at_16_bits_the_choice_and_census_are_the_full_search_s_bit_for_bit(name):
    calls, signed = SEAMS[name]
    assert _search(calls, 32767, signed) == _search(calls, 32767, signed, full=True)


def test_the_full_search_reads_every_candidate_on_every_call(monkeypatch):
    eng = _need_full()
    read = []
    rounding = eng._Sorted.rounding_parts

    def counted(self, low, below, spans, levels, kept):
        read.append(len(spans))
        return rounding(self, low, below, spans, levels, kept)

    monkeypatch.setattr(eng._Sorted, "rounding_parts", counted)
    calls, _signed = SEAMS["saturated"]
    _search(calls, 127, False, full=True)
    candidates = len(eng.whole_interval_candidates(max(float(c.abs().max()) for c in calls)))
    # every candidate on every call, and the widest window once more a call (the second pass's upper bound reads it)
    assert sum(read) == len(calls) * (candidates + 1)


def _direct_error(calls, span, levels):
    """The grid's own error read value by value: each value's code by the search's dtype-rounded cell edges."""
    step = span / levels
    edges = ((torch.arange(1, levels + 1, dtype=torch.float64) - 0.5) * step).to(torch.float32)
    top = torch.tensor(span, dtype=torch.float64).to(torch.float32)
    total = 0.0
    for v in calls:
        a = v.abs().contiguous()
        code = torch.searchsorted(edges, a, right=True)
        q = torch.where(a > top, torch.full(a.shape, span, dtype=torch.float64), code.to(torch.float64) * step)
        total += float(((a.to(torch.float64) - q) ** 2).sum())
    return total


@pytest.mark.parametrize("name", ["offset cluster", "saturated", "bimodal", "ties"])
@pytest.mark.parametrize("levels", [127, 2047])
def test_the_error_value_departs_from_the_grid_s_own_by_at_most_the_stated_resolution(name, levels):
    eng = _need_full()
    resolution = getattr(eng, "ERROR_RESOLUTION", None)
    if resolution is None:
        pytest.fail("the engine states no resolution of its error values")
    calls, _signed = SEAMS[name]
    row = _search(calls, levels, False)
    energy = sum(float((c.double() ** 2).sum()) for c in calls)
    departure = abs(row.squared_error - _direct_error(calls, row.scale, levels))
    assert departure <= resolution * energy, (departure, resolution * energy)


def _tied_census(rows):
    """The census of a signed site whose windows' errors are given (every window read on every call): ``rows`` is
    ``[(low end, [(span, error), ...]), ...]``, the first low end 0."""
    eng = _need_full()
    from mimarsinan.mapping.support.range_calls import _Low, _Site

    by_low = {}
    for i, (low, cells) in enumerate(rows):
        spans = torch.tensor([s for s, _e in cells], dtype=torch.float64)
        error = torch.tensor([e for _s, e in cells], dtype=torch.float64)
        c = len(cells)
        by_low[i] = _Low(float(low), spans, error.clone(), error, torch.ones(c, dtype=torch.bool),
                         torch.zeros(c, dtype=torch.float64), torch.zeros(c, dtype=torch.float64), calls=1)
    lows = torch.tensor([low for low, _cells in rows], dtype=torch.float64)
    site = _Site(eng.SiteSpec(7, signed=True), [float(lows.min())], [4.0], [8], lambda high: by_low[0].spans,
                 lows=lows, low_only=torch.zeros(len(rows), dtype=torch.float64), by_low=by_low, energy=1.0,
                 entries=8)
    census = eng._census(0, site, 4.0, float(lows.min()))
    return census.scale, census.low


def test_ties_go_to_the_least_span_then_to_the_low_end_nearest_zero():
    """The tie rule the engine and its full search share (mutant M11 sent ties to the widest window): of the windows
    whose error is the least, the least span, then the low end nearest 0, read on the census itself."""
    assert _tied_census([(0.0, [(0.5, 9.0), (1.0, 5.0), (2.0, 8.0)]),
                         (-1.0, [(0.5, 9.0), (2.0, 5.0), (3.0, 8.0)])]) == (1.0, 0.0)
    assert _tied_census([(0.0, [(0.5, 9.0), (2.0, 5.0), (3.0, 8.0)]),
                         (-1.0, [(0.5, 9.0), (1.0, 5.0), (3.0, 8.0)])]) == (1.0, -1.0)
    assert _tied_census([(0.0, [(0.5, 9.0), (2.0, 5.0), (3.0, 8.0)]),
                         (-1.0, [(0.5, 9.0), (2.0, 5.0), (3.0, 8.0)])]) == (2.0, 0.0)
