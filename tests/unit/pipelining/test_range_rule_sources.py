"""[U-BRANGE2, second form] One source for every range site: a module's input or output, or any pass a caller runs,
every value read; a signed site's window searched at both ends; the census of systematic clamps.

The spiking route's range sites (theta, the entry gauge, the seam covers, the fan-in statistic) read their values from
a module's output or from the segment walk's host values, and a signed seam's window has a low end below 0. They read
the rule through these entries, never a quantile and never a sample.
"""

from __future__ import annotations

import pytest
import torch

try:
    import mimarsinan.mapping.support.range_rule as range_rule
except ImportError:
    range_rule = None


def _rule():
    if range_rule is None:
        pytest.fail("no range rule module on this head")
    return range_rule


def _need(name):
    value = getattr(_rule(), name, None)
    if value is None:
        pytest.fail(f"the rule has no {name}: a range site reads only a module's input")
    return value


def _seam(seed=3, n=4096, hot=8.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.relu(torch.randn(16, n, generator=g))
    x[:, 7] = hot * (1.0 + 0.05 * torch.rand(16, generator=g))  # one position at 8 times the bulk on every sample
    return x


def _grid_error(values, low, span, levels):
    """The window's own squared error, read directly: codes round half away from the low end, saturating."""
    v = values.reshape(-1).double()
    step = span / levels
    codes = torch.clamp(torch.floor((v - low) / step + 0.5), 0, levels)
    return float(((v - (low + codes * step)) ** 2).sum())


class _Double(torch.nn.Module):
    """A module whose output is twice its input: an output site and an input site of it read different values."""

    def forward(self, x):
        return 2.0 * x


def test_a_site_reads_its_module_s_output():
    """[U-BRANGE2 stage 1c, the delta verify's n-1, corrected] A site declared ``reads="output"`` reads the module's
    output, not its input. The first form read a ReLU on non-negative values, whose output equals its input, so it
    could not tell the two apart (X61, an output site reading the input, passed it). On a doubling module the output
    site's calibration maximum is twice the input site's, and so is its range (the search's candidates follow the
    maximum, and doubling is exact). Red on X61."""
    site_type = _rule().RangeSite
    if "reads" not in getattr(site_type, "__dataclass_fields__", {}):
        pytest.fail("a range site reads only its module's input")
    x = [_seam(1) + 0.5, _seam(2) + 0.5]
    double = _Double()
    model = torch.nn.Sequential(torch.nn.Identity(), double)
    by_output = _rule().least_squares_ranges(model, [site_type(double, 15, torch.abs, reads="output")], x)[0]
    by_input = _rule().least_squares_ranges(model, [site_type(double, 15)], x)[0]
    assert by_output.maximum == 2.0 * by_input.maximum, (by_output.maximum, by_input.maximum)
    assert by_output.scale == 2.0 * by_input.scale, (by_output.scale, by_input.scale)


def test_a_recorded_source_reads_every_value_as_a_module_site_does():
    recorded = _need("least_squares_recorded")
    from mimarsinan.mapping.support.range_search import SiteSpec

    x = [_seam(1), _seam(2)]
    passes = []

    def run_pass(record, only):
        passes.append(only)
        for b in x:
            record(0, b.abs())

    via_source = recorded(run_pass, [SiteSpec(15)])[0]
    probe = torch.nn.Identity()
    via_module = _rule().least_squares_ranges(probe, [_rule().RangeSite(probe, 15)], x)[0]
    assert via_source == via_module
    assert len(passes) in (2, 3)


@pytest.mark.parametrize("levels", [15, 127])
def test_t1_form_a_systematic_position_is_clamped_only_where_the_error_minimum_says(levels):
    """T-1's form on a staircase of ``levels``: one position at 8 times the bulk on every sample. The chosen top is the
    least error of the search (read directly), and at most both ends' errors."""
    x = [_seam(1), _seam(2)]
    probe = torch.nn.Identity()
    row = _rule().least_squares_ranges(probe, [_rule().RangeSite(probe, levels)], x)[0]
    chosen = sum(_grid_error(b, 0.0, row.scale, levels) for b in x)
    assert chosen <= row.low_span_error * (1 + 1e-6) and chosen <= row.maximum_error * (1 + 1e-6)
    from mimarsinan.mapping.support.range_search import whole_interval_candidates

    for s in whole_interval_candidates(row.maximum)[::64].tolist() + [row.maximum]:
        assert chosen <= sum(_grid_error(b, 0.0, s, levels) for b in x) * (1 + 1e-6) + 1e-12


def _signed(seed=5, n=1 << 14):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(8, n, generator=g) * 1.5 + 0.4
    v[:, 3] = -6.0  # a systematic negative position on every sample
    return v


def test_a_signed_site_s_window_is_the_least_error_of_its_low_ends_and_spans():
    from mimarsinan.mapping.support.range_search import SiteSpec, whole_interval_candidates

    recorded = _need("least_squares_recorded")
    x = [_signed(5), _signed(6)]

    def run_pass(record, only):
        for b in x:
            record(0, b)

    row = recorded(run_pass, [SiteSpec(15, signed=True)])[0]
    chosen = sum(_grid_error(b, row.low, row.scale, 15) for b in x)
    assert abs(chosen - row.squared_error) <= 1e-6 * chosen
    lows = [0.0] + (-whole_interval_candidates(6.0)).tolist()
    top = max(float(b.max()) for b in x)
    for low in lows[::97] + [row.low]:
        for span in whole_interval_candidates(top - low).tolist()[::97]:
            assert chosen <= sum(_grid_error(b, low, span, 15) for b in x) * (1 + 1e-6)
    assert row.low < 0.0


def test_a_signed_site_is_the_full_search_s(monkeypatch):
    """Every (low end, span) pair evaluated on every call against the search as it runs (two octaves per end here, so
    the full search reads 263,169 windows in a unit test's time)."""
    import mimarsinan.mapping.support.range_search as engine

    recorded = _need("least_squares_recorded")
    monkeypatch.setattr(engine, "RANGE_OCTAVES", 2)
    x = [_signed(7), _signed(8)]

    def run_pass(record, only):
        for b in x:
            record(0, b)

    row = recorded(run_pass, [engine.SiteSpec(15, signed=True)])[0]
    import inspect

    if "full" not in inspect.signature(recorded).parameters:  # [stage 1] the engine's own full search
        pytest.fail("the rule holds no full search of its own")
    full = recorded(run_pass, [engine.SiteSpec(15, signed=True)], full=True)[0]  # the reference reads every window
    assert row == full


def test_the_census_counts_positions_clamped_on_every_sample():
    census = _need("positions_above")
    x = [_seam(1), _seam(2)]

    def run_pass(record, only):
        for b in x:
            record(0, b)

    (row,) = census(run_pass, [(0.0, 4.0)])
    assert row["positions_above"] == 1 and row["samples"] == 32 and row["positions"] == x[0].shape[1]
