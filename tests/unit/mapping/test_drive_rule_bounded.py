"""[U-BRANGE2, second form] The always-on row's drive search is exact with a bounded transient at every declared width.

R3's first form evaluated every code at or above ``max|b| / max|w|`` against every bias at once: its transient was codes
times biases (the design review's M-1: 2.95 GB and 3.0 s on one 4,096-output perceptron at 16 bits). The second form
reads codes in ascending blocks and biases in chunks, each code's error summed in bias order, and drops a code once its
running sum exceeds the least error before it (every term is a square). The full search here is the same arithmetic on
every code with nothing dropped: the choice must be its, bit for bit.
"""

from __future__ import annotations

import math

import pytest
import torch

try:
    import mimarsinan.mapping.support.drive_rule as drive_rule
except ImportError:  # R5 and revision 1: the drive is the top code value, no drive rule module
    drive_rule = None


def _rule():
    if drive_rule is None:
        pytest.fail("no drive rule module on this head (the drive is its reader grid's top code value)")
    return drive_rule


def _one_row():
    """[stage 1] The drive at one deployed row (the drive site's row budget), on a head that reads one."""
    budget = getattr(_rule(), "RowBudget", None)
    return {} if budget is None else {"budget": budget(split=False)}


def _split():
    """[stage 1] The drive with the rows its bias needs (split rows, no cores declared), on a head that reads one."""
    budget = getattr(_rule(), "RowBudget", None)
    return {} if budget is None else {"budget": budget(split=True)}


def _perceptron(outputs: int, fan_in: int, seed: int, bias_sd: float = 0.02, lattice: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(outputs, fan_in, generator=g) * 0.05
    b = torch.randn(outputs, generator=g) * bias_sd
    if lattice:  # every bias on a lattice: a code whose row grid holds it exactly has error 0
        b = torch.round(b / lattice) * lattice
    return w, b


def _full_search(w, b, grid, q_max, fractional):
    """Every code at or above the ratio, each error the sequential sum of its squared departures in bias order."""
    rule = _rule()
    bb = b.reshape(-1).double()
    w_max, b_max = float(w.abs().max()), float(bb.abs().max())
    ratio = b_max / w_max
    codes = [k for k in range(1, grid.code_max + 1) if grid.value(k) > 0.0 and grid.value(k) >= ratio]
    if not codes:
        return grid.top, None
    drives = torch.tensor([grid.value(k) for k in codes], dtype=torch.float64)
    peak = torch.clamp(torch.maximum(torch.full_like(drives, w_max), b_max / drives), min=1e-12)
    scale = q_max / peak if fractional else torch.clamp(torch.floor(q_max / peak), min=1.0)
    errors = []
    for i in range(0, len(codes), 256):
        sl = slice(i, i + 256)
        rows = torch.round(bb[None, :] * (scale[sl] / drives[sl])[:, None]).clamp(-(q_max + 1), q_max)
        terms = (bb[None, :] - rows * (drives[sl] / scale[sl])[:, None]) ** 2
        errors.append(torch.cumsum(terms, dim=1)[:, -1])
    errors = torch.cat(errors)
    at = int(torch.argmin(errors))
    del rule
    return grid.value(codes[at]), errors


CASES = [(bits, wbits, offset, fractional, seed)
         for bits, wbits in ((8, 8), (12, 8), (12, 12), (16, 8), (16, 16))
         for offset in (0.0, 0.37)
         for fractional in (True, False)
         for seed in (1, 2)]


def _grid(bits, offset, top=8.0):
    levels = (1 << (bits - 1)) - 1
    return _rule().DriveGrid(step=(top + offset) / levels, offset=offset, code_max=levels)


@pytest.mark.parametrize("bits, wbits, offset, fractional, seed", [c for c in CASES if c[0] < 16])
def test_the_drive_is_the_full_search_s_bit_for_bit(bits, wbits, offset, fractional, seed):
    w, b = _perceptron(512, 64, seed)
    grid, q_max = _grid(bits, offset), (1 << (wbits - 1)) - 1
    expected, _ = _full_search(w, b, grid, q_max, fractional)
    assert _rule().least_error_drive(w, b, grid, q_max, fractional=fractional, **_one_row()) == expected


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("bits, wbits, offset, fractional, seed", [c for c in CASES if c[0] == 16])
def test_at_16_bits_the_drive_is_the_full_search_s_bit_for_bit(bits, wbits, offset, fractional, seed):
    w, b = _perceptron(4096, 64, seed)
    grid, q_max = _grid(bits, offset), (1 << (wbits - 1)) - 1
    expected, _ = _full_search(w, b, grid, q_max, fractional)
    assert _rule().least_error_drive(w, b, grid, q_max, fractional=fractional, **_one_row()) == expected


@pytest.mark.parametrize("bits", [8, 12])
def test_a_lattice_bias_finds_its_exact_code_far_above_the_least(bits):
    """Every bias on the lattice of one code's row grid: that code's error is 0 and it is the drive, however far above
    the ratio it lies (no window of codes is assumed)."""
    rule = _rule()
    grid, q_max = _grid(bits, 0.0), 127
    w, _ = _perceptron(256, 64, 5)
    w_max = float(w.abs().max())
    target = grid.code_max - 3
    step = grid.value(target) / (q_max / w_max)  # the row grid's step at that code under the fractional rule
    b = torch.round(torch.randn(256, generator=torch.Generator().manual_seed(9)) * 4.0).clamp(-30, 30) * step
    b[0] = 30 * step
    expected, errors = _full_search(w, b, grid, q_max, True)
    assert rule.least_error_drive(w, b, grid, q_max, fractional=True, **_one_row()) == expected
    assert float(errors.min()) <= 1e-9 * float(errors.median())  # that code's grid holds every bias (float rounding aside)


def _record(monkeypatch, bound):
    sizes = []
    real_round = torch.round

    def round_(x, *a, **k):
        sizes.append(x.numel())
        if x.numel() > bound:
            raise AssertionError(f"{x.numel()} (code, bias) pairs at once against the bound {bound}")
        return real_round(x, *a, **k)

    monkeypatch.setattr(torch, "round", round_)
    return sizes


@pytest.mark.parametrize("bits", [8, 12, 16])
def test_the_transient_is_bounded_independent_of_codes_times_biases(bits, monkeypatch):
    rule = _rule()
    w, b = _perceptron(4096, 32, 3)
    grid = _grid(bits, 0.0)
    bound = getattr(rule, "_BLOCK_PAIRS", 0)
    sizes = _record(monkeypatch, bound)
    rule.least_error_drive(w, b, grid, (1 << (min(bits, 12) - 1)) - 1, fractional=True, **_one_row())
    assert sizes and max(sizes) <= bound


def test_at_16_bits_codes_that_cannot_be_least_are_dropped_early(monkeypatch):
    """A 4,096-output perceptron at 16 bits: the pairs read are a small share of the full search's."""
    rule = _rule()
    w, b = _perceptron(4096, 32, 3)
    grid, q_max = _grid(16, 0.0), 127
    ratio = float(b.abs().max()) / float(w.abs().max())
    full = sum(1 for k in range(1, grid.code_max + 1) if grid.value(k) >= ratio) * 4096
    sizes = _record(monkeypatch, getattr(rule, "_BLOCK_PAIRS", 1 << 18))
    rule.least_error_drive(w, b, grid, q_max, fractional=True, **_one_row())
    assert sum(sizes) <= full / 4, (sum(sizes), full)


def test_no_code_reaches_the_ratio_takes_the_rows_that_hold_it():
    """[m-2, amended pin] Where no code reaches ``max|b| / max|w|`` the drive is no longer the top code value (one row,
    the weight grid widened): it is the least-error code of the least row count that holds the ratio."""
    rule = _rule()
    w, b = _perceptron(64, 16, 4, bias_sd=10.0)
    grid = _grid(8, 0.0)
    ratio = float(b.double().abs().max()) / float(w.abs().max())
    assert ratio > grid.top
    drive, rows = rule.least_error_drive_and_rows(w, b, grid, 127, fractional=True, **_split())
    assert rows == rule.bias_rows(ratio, grid.top) >= 2 and rows * drive >= ratio
    assert rule.least_error_drive(w, b, grid, 127, fractional=True, **_split()) == drive


def test_the_drive_never_writes_its_inputs():
    rule = _rule()
    w, b = _perceptron(256, 16, 6)
    w64, b64 = w.double(), b.double()
    before = (w64.clone(), b64.clone())
    rule.least_error_drive(w64, b64, _grid(12, 0.0), 127, fractional=True, **_one_row())
    assert torch.equal(before[0], w64) and torch.equal(before[1], b64)


def test_a_tie_goes_to_the_least_code():
    """Codes whose row grids hold every bias at the same error tie; the least of them is the drive, as in the full
    search (its first minimum)."""
    rule = _rule()
    w = torch.ones(4, 4)
    b = torch.tensor([0.5, 0.0, 0.0, 0.0])
    grid = rule.DriveGrid(step=0.25, offset=0.0, code_max=255)
    expected, errors = _full_search(w, b, grid, 127, True)
    assert rule.least_error_drive(w, b, grid, 127, fractional=True, **_one_row()) == expected
    assert int((errors == errors.min()).sum()) >= 1


# --- [U-BRANGE2 stage 1c] the rounding edges of the candidate codes (the delta verify's M-10 and the sweep's C-ids) ---


def test_a_code_whose_value_is_the_ratio_stays_a_candidate():
    """[U-BRANGE2 stage 1c, the delta verify's M-10] The first candidate code is the floor of ``(ratio + offset) /
    step``: where that quotient rounds one unit in the last place above a code whose value equals the ratio, the code
    stays a candidate. Grid step 0.30917334227783244, offset 0, 127 codes; code 14's value 4.328426791889655 divided
    by the step is 14.000000000000002 (the verify's search, seed 7). One bias at that value and max|w| 1: code 14
    holds the bias at its row's top code with error 0, the full search's choice. Red on X24 (the first code rounded
    up: the drive 39.265, code 127)."""
    rule = _rule()
    grid = rule.DriveGrid(step=0.30917334227783244, offset=0.0, code_max=127)
    value = grid.value(14)
    assert (value + grid.offset) / grid.step > 14, "fixture: the quotient rounds above the code"
    weight = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    bias = torch.tensor([value], dtype=torch.float64)
    expected, _errors = _full_search(weight, bias, grid, 127, True)
    assert expected == value, "fixture: the full search takes the code at the ratio"
    assert rule.least_error_drive_and_rows(weight, bias, grid, 127, fractional=True, **_one_row()) == (value, 1)


@pytest.mark.parametrize("offset", [0.0, -2.0])
def test_biases_below_the_first_code_value_take_code_one(offset):
    """[U-BRANGE2 stage 1c, sweeps C-R1 and C-R2] The candidates start at code 1: where ``max|b| / max|w|`` lies
    below the first code value, every code holds the ratio and the finest row grid, code 1's, is the full search's
    choice. A declared entry or lane grid may carry any finite offset (the map's schema), a negative one too: at
    offset -2.0 code 1's value is 2.5, and the first code read with the offset's sign reversed would start at 4."""
    rule = _rule()
    grid = rule.DriveGrid(step=0.5, offset=offset, code_max=127)
    w, _ = _perceptron(64, 16, 11)
    b = torch.randn(64, generator=torch.Generator().manual_seed(12)) * 0.004
    assert float(b.abs().max()) / float(w.abs().max()) < grid.value(1), "fixture: the ratio lies below code 1"
    expected, _errors = _full_search(w, b, grid, 127, True)
    assert expected == grid.value(1), "fixture: the full search takes code 1"
    assert rule.least_error_drive(w, b, grid, 127, fractional=True, **_one_row()) == expected


@pytest.mark.parametrize("step, code", [(0.4763186124829258, 93), (1.1580502325653723, 125),
                                        (0.30917334227783244, 14)])
def test_a_code_value_is_held_where_its_quotient_falls_an_ulp_off_the_code(step, code):
    """[U-BRANGE2 stage 1c, sweeps C-R5 and C-R6] ``DriveGrid.holds`` reads a drive's code by rounding ``(drive +
    offset) / step`` to the nearest code: a code value whose quotient falls one unit in the last place below its code
    (93 at step 0.4763...: 92.99999999999999) or above it (125 at step 1.1580...: 125.00000000000001; 14 at the
    verify's step) is a code value of the grid, as the stamp writes it (this step's search,
    probe/holds_edge_search.py, seed 3). A floor or a ceiling would refuse the solvers' own drive."""
    rule = _rule()
    grid = rule.DriveGrid(step=step, offset=0.0, code_max=127)
    drive = grid.value(code)
    assert (drive + grid.offset) / grid.step != code, "fixture: the quotient is off its code"
    assert grid.holds(drive)
    assert not grid.holds(math.nextafter(drive, math.inf)) and not grid.holds(math.nextafter(drive, 0.0))


def test_a_code_value_at_or_below_zero_is_no_drive_the_grid_holds():
    """[U-BRANGE2 stage 1c, sweep C-R7] A drive is a positive code value (the reader's DAC writes it as the always-on
    axon's input): on a grid whose offset exceeds its step, code 1's value is negative and code 2's is 0, and the grid
    holds neither, as the candidates keep only positive values; code 3's, 0.25, it holds."""
    rule = _rule()
    grid = rule.DriveGrid(step=0.25, offset=0.5, code_max=127)
    assert (grid.value(1), grid.value(2), grid.value(3)) == (-0.25, 0.0, 0.25)
    assert not grid.holds(grid.value(1)) and not grid.holds(grid.value(2)) and grid.holds(grid.value(3))
