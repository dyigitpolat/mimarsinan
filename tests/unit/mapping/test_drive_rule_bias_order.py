"""[U-BRANGE2 stage 1, the review's m-4] The drive search's work is bounded whatever the order of the biases.

The second form summed each code's squared departures in bias order and dropped a code once its running sum exceeded
the least error found: a bound that acts only when the running sums carry the biases' errors. Where the biases that
carry error come last (every bias zero but the last three; biases over eight decades), every code was read on nearly
every bias, the cost law's worst case O(codes x biases): 0.29 s at 32,767 codes and 0.55 s at 65,535 against the
perceptron's projection's 0.02 s (design2_review.md m-4). The generic change reads the biases in one order fixed by
their own magnitudes for the bound (the largest first) and sums each surviving code's error in bias order for the
choice, so the bound acts on the first biases read and the choice stays the full search's bit for bit.
Here the work is the number of (code, bias) pairs the rule rounds, counted on the rule itself.
"""

from __future__ import annotations

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


def _drive(w, b, grid, q_max, fractional=True):
    """The rule's drive at one row (every case here has a code at or above the ratio)."""
    rule = _rule()
    budget = getattr(rule, "RowBudget", None)
    if budget is None:
        return rule.least_error_drive(w, b, grid, q_max, fractional=fractional)
    return rule.least_error_drive(w, b, grid, q_max, fractional=fractional, budget=budget(split=False))


def _grid(bits: int, top: float = 8.0):
    levels = (1 << (bits - 1)) - 1
    return _rule().DriveGrid(step=top / levels, offset=0.0, code_max=levels)


def _full_search(w, b, grid, q_max, fractional):
    """Every code at or above the ratio, each error the sequential sum of its squared departures in bias order."""
    bb = b.reshape(-1).double()
    w_max, b_max = float(w.abs().max()), float(bb.abs().max())
    ratio = b_max / w_max
    first = max(1, int(ratio / grid.step) - 2)
    codes = [k for k in range(first, grid.code_max + 1) if grid.value(k) > 0.0 and grid.value(k) >= ratio]
    drives = torch.tensor([grid.value(k) for k in codes], dtype=torch.float64)
    peak = torch.clamp(torch.maximum(torch.full_like(drives, w_max), b_max / drives), min=1e-12)
    scale = q_max / peak if fractional else torch.clamp(torch.floor(q_max / peak), min=1.0)
    errors = []
    for i in range(0, len(codes), 128):
        sl = slice(i, i + 128)
        rows = torch.round(bb[None, :] * (scale[sl] / drives[sl])[:, None]).clamp(-(q_max + 1), q_max)
        terms = (bb[None, :] - rows * (drives[sl] / scale[sl])[:, None]) ** 2
        errors.append(torch.cumsum(terms, dim=1)[:, -1])
    errors = torch.cat(errors)
    return grid.value(codes[int(torch.argmin(errors))]), len(codes)


def _biases(form: str, outputs: int = 4096, seed: int = 2):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(outputs, 64, generator=g) * 0.05
    if form == "zero-led":  # every bias zero but the last three
        b = torch.zeros(outputs)
        b[-3:] = torch.tensor([0.013, -0.007, 0.021])
    elif form == "eight decades":
        b = torch.randn(outputs, generator=g).sign() * torch.pow(10.0, -8.0 * torch.rand(outputs, generator=g)) * 0.02
    elif form == "zero-led, then large":  # the largest biases last
        b = torch.randn(outputs, generator=g) * 1e-6
        b[-64:] = torch.randn(64, generator=g) * 0.02
    else:
        b = torch.randn(outputs, generator=g) * 0.02
    return w, b


def _counted_pairs(monkeypatch):
    sizes = []
    real_round = torch.round

    def round_(x, *a, **k):
        sizes.append(x.numel())
        return real_round(x, *a, **k)

    monkeypatch.setattr(torch, "round", round_)
    return sizes


FORMS = ["zero-led", "eight decades", "zero-led, then large", "normal"]


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("bits", [12, 16])
def test_the_drive_is_the_full_search_s_bit_for_bit_whatever_the_bias_order(form, bits):
    """A guard: the second form already holds it; the bounded order must keep it."""
    w, b = _biases(form)
    grid = _grid(bits)
    expected, _codes = _full_search(w, b, grid, 127, True)
    assert _drive(w, b, grid, 127) == expected


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("bits", [12, 16])
def test_the_work_is_a_small_share_of_codes_times_biases_whatever_the_bias_order(form, bits, monkeypatch):
    """At most an eighth of the full search's (code, bias) pairs, whatever the order the biases come in."""
    w, b = _biases(form)
    grid = _grid(bits)
    _expected, codes = _full_search(w, b, grid, 127, True)
    sizes = _counted_pairs(monkeypatch)
    _drive(w, b, grid, 127)
    full = codes * b.numel()
    assert sum(sizes) <= full / 8, f"{form}: {sum(sizes)} (code, bias) pairs read of the full search's {full}"


def test_a_permutation_of_the_biases_reads_about_the_same_work(monkeypatch):
    """The bound reads the biases in an order of their own magnitudes, so the work does not follow the order the
    perceptron's outputs come in: a zero-led order and its reverse read within a factor of two."""
    w, b = _biases("zero-led")
    grid = _grid(16)
    sizes = _counted_pairs(monkeypatch)
    _drive(w, b, grid, 127)
    led = sum(sizes)
    sizes.clear()
    _drive(w, b.flip(0), grid, 127)
    trailing = sum(sizes)
    assert led <= 2 * trailing and trailing <= 2 * led, (led, trailing)


def test_the_choice_s_errors_are_summed_in_the_biases_own_order(monkeypatch):
    """The bound reads the biases largest first; the errors the choice compares are summed in the biases' own order,
    the full search's (mutant M08 summed them in the bound's order: a near tie could then resolve otherwise). Read on
    the rule's own summation: some call reads the biases in their own order, and every such call carries no bound."""
    rule = _rule()
    sums = getattr(rule, "_sums", None)
    if sums is None:
        pytest.fail("the drive rule holds no summation of a code's error to read")
    seen = []

    def recording(b, drives, w_max, b_max, q_max, fractional, stop=None):
        seen.append((b.clone(), stop))
        return sums(b, drives, w_max, b_max, q_max, fractional, stop=stop)

    monkeypatch.setattr(rule, "_sums", recording)
    w, b = _biases("zero-led, then large")
    _drive(w, b, _grid(12), 127)
    own = [stop for read, stop in seen if torch.equal(read, b.reshape(-1).double())]
    assert own, "no summation read the biases in their own order"
    assert all(stop is None for stop in own)


# --- [U-BRANGE2 stage 1b, the verify's m-3] the bound's reorder margin ------------------------------------------------
# The bound sums each code's errors largest bias first and the choice sums them in bias order; two float64 sums of the
# same non-negative terms in two orders differ by at most 2 n eps of their value, and the margin covers that. With the
# margin at 0 (the verify's VM13) every clause above passed. The perceptron below was found by this step's search
# (probe/margin_search.py, 29,110 small perceptrons read): codes 6 and 7 tie exactly in bias order, while code 6's
# largest-first sum lies one unit in the last place above code 7's, so at margin 0 the bound keeps code 7 alone.

_TIE_BIASES = ("-0x1.bffffffc00000p+3", "-0x1.bffffff000000p+2", "0x1.2ffffffc00000p+5")


def _sequential(terms, order):
    total = 0.0
    for i in order:
        total += terms[i]
    return total


def _code_sums(b, w_max, q_max, drive):
    """One code's error summed in bias order and largest bias first, by sequential float64 sums written here (the
    module's term: ``(b - round(b s / d) d / s)^2`` at ``s = q_max / max(max|w|, max|b| / d)``)."""
    b_max = max(abs(x) for x in b)
    s = q_max / max(w_max, b_max / drive)
    terms = [(x - max(-(q_max + 1), min(q_max, round(x * (s / drive)))) * (drive / s)) ** 2 for x in b]
    largest_first = sorted(range(len(b)), key=lambda i: -abs(b[i]))
    return _sequential(terms, range(len(b))), _sequential(terms, largest_first)


def test_a_code_tied_in_bias_order_and_one_rounding_above_largest_first_is_the_choice():
    rule = _rule()
    b = [float.fromhex(h) for h in _TIE_BIASES]
    w, q_max = torch.tensor([7.0]), 7
    grid = rule.DriveGrid(step=1.0, offset=0.0, code_max=7)
    six, seven = _code_sums(b, 7.0, q_max, 6.0), _code_sums(b, 7.0, q_max, 7.0)
    assert six[0] == seven[0], "fixture: codes 6 and 7 tie in bias order"
    assert six[1] > seven[1], "fixture: code 6's largest-first sum lies above code 7's"
    bias = torch.tensor(b, dtype=torch.float64)
    full, _codes = _full_search(w, bias, grid, q_max, True)
    assert full == 6.0  # the full search: the tie goes to the least code
    assert _drive(w, bias, grid, q_max) == full
