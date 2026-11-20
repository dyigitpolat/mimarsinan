"""[U-BRANGE2 stage 1, the review's m-3] An empty calibration call is no call.

A source may hand a site a call with no values (a batch with no sample, a hook that reads an empty tensor). The rule's
choice depends only on the values it reads, so such a call must change neither the choice nor the census, and it must
not change the search's cost either. The second form returned from an empty call before its bookkeeping, so the next
call found every candidate one call short and reopened all of them: at 32,767 levels one empty call took a site from
0.11 s to 21 s and from 425,984 to 201,818,112 evaluated (candidate, cell) pairs (design2_review.md, q6_empty.py).
Here the cost is the number of (candidate, cell) pairs the search's rounding parts read, counted on the engine itself.
"""

from __future__ import annotations

import pytest
import torch

try:  # each clause reads the engine itself, so a head without it fails clause by clause, never at collection
    import mimarsinan.mapping.support.range_search as engine
    import mimarsinan.mapping.support.range_rule as range_rule
except ImportError:  # R5 (the 0.999 quantile) and revision 1 (one module, no search engine)
    engine = range_rule = None


def _need():
    if engine is None:
        pytest.fail("no range search engine on this head: no empty call to read")
    return engine


def _calls(signed: bool = False):
    g = torch.Generator().manual_seed(17)
    calls = [torch.randn(64, 4096, generator=g) for _ in range(2)]
    if signed:
        return [c * 1.3 + 0.2 for c in calls]
    return [c.abs() for c in calls]


def _insert_empty(calls, where: str):
    empty = torch.empty(0)
    if where == "first":
        return [empty] + list(calls)
    if where == "between":
        return [calls[0], empty] + list(calls[1:])
    return list(calls) + [empty]


def _sorted_type():
    """The per-call sorted form whose rounding parts the search reads (the engine's own class)."""
    return _need()._Sorted


def _search_counted(calls, levels: int, signed: bool, monkeypatch):
    """The site's census and the (candidate, cell) pairs its rounding parts read."""
    eng = _need()
    sorted_type = _sorted_type()
    rounding = sorted_type.rounding_parts
    pairs = []

    def counted(self, low, below, spans, lv, kept):
        pairs.append(len(spans) * (lv + 1))
        return rounding(self, low, below, spans, lv, kept)

    with monkeypatch.context() as m:
        m.setattr(sorted_type, "rounding_parts", counted)

        def run_pass(record, only):
            for c in calls:
                record(0, c)

        (row,) = eng.search_ranges(run_pass, [eng.SiteSpec(levels, signed=signed)])
    return row, sum(pairs)


@pytest.mark.parametrize("where", ["first", "between", "last"])
@pytest.mark.parametrize("levels", [2047, 32767])
def test_an_empty_call_changes_neither_the_choice_nor_the_cost(where, levels, monkeypatch):
    base = _calls()
    row0, pairs0 = _search_counted(base, levels, False, monkeypatch)
    row1, pairs1 = _search_counted(_insert_empty(base, where), levels, False, monkeypatch)
    assert row1 == row0, (row1, row0)  # the window, its census and its three errors, bit for bit
    assert pairs1 == pairs0, f"an empty call {where}: {pairs1} pairs read against {pairs0} without it"


@pytest.mark.parametrize("where", ["first", "between"])
def test_an_empty_call_on_a_signed_site_changes_neither_the_choice_nor_the_cost(where, monkeypatch):
    base = _calls(signed=True)
    row0, pairs0 = _search_counted(base, 15, True, monkeypatch)
    row1, pairs1 = _search_counted(_insert_empty(base, where), 15, True, monkeypatch)
    assert row1 == row0, (row1, row0)
    assert pairs1 == pairs0, f"an empty call {where}: {pairs1} pairs read against {pairs0} without it"


def test_an_empty_batch_through_a_module_site_is_no_call(monkeypatch):
    """The module source: a batch with no sample runs the model, its hook reads an empty tensor."""
    _need()
    probe = torch.nn.Identity()
    site = range_rule.RangeSite(probe, 2047)
    base = [c[:16] for c in _calls()]
    with_empty = [base[0], torch.empty(0, 4096), base[1]]
    (row0,) = range_rule.least_squares_ranges(probe, [site], base)
    (row1,) = range_rule.least_squares_ranges(probe, [site], with_empty)
    assert row1 == row0


def test_the_census_of_systematic_clamps_counts_no_empty_call():
    _need()
    base = _calls()

    def run_pass(calls):
        def run(record, only):
            for c in calls:
                record(0, c)
        return run

    (row0,) = range_rule.positions_above(run_pass(base), [(0.0, 3.0)])
    (row1,) = range_rule.positions_above(run_pass([torch.empty(0, 4096)] + base), [(0.0, 3.0)])
    assert row1 == row0
