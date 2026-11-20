"""[U-BRANGE2, m-2] The drive site's rule takes the row count: where no code value of the reader's grid reaches
``max|b| / max|w|``, the bias rides the least number of always-on rows that holds it on the weights' own grid.

With one row, a drive below the ratio widens the shared weight grid (every weight of the perceptron coarsens: the
review's m-2, MobileNetV2 12 of 36 perceptrons, ResNet-18 arm N 15 of 21, widened up to 3.5 times). With ``k`` rows each
carrying ``b / k`` at drive ``d`` the row reads ``k d``: the least ``k`` with ``k`` times the top code value at or above
the ratio leaves the weight grid the weights' own, and among the codes with ``k d`` at or above the ratio the drive is
the one whose row grid gives the least squared error of the bias (the drive rule itself on the effective drive ``k d``).
Each extra row is one always-on axon on every core of the perceptron: the row cost the stamps count.
"""

from __future__ import annotations

import math

import pytest
import torch

try:
    import mimarsinan.mapping.support.drive_rule as drive_rule
except ImportError:  # R5 and revision 1: the drive is the top code value, no drive rule module
    drive_rule = None


def _rows_rule():
    if drive_rule is None:
        pytest.fail("no drive rule module on this head (the drive is its reader grid's top code value)")
    missing = [n for n in ("bias_rows", "least_error_drive_and_rows") if not hasattr(drive_rule, n)]
    if missing:
        pytest.fail(f"the drive rule takes no row count: {missing} missing (a drive below the ratio widens the weights)")
    return drive_rule


def _split():
    """[stage 1] The rows the bias needs (split rows, no cores declared), on a head whose rule reads a row budget."""
    budget = getattr(_rows_rule(), "RowBudget", None)
    return {} if budget is None else {"budget": budget(split=True)}


def _grid(bits: int, offset: float = 0.0, top: float = 1.0):
    levels = (1 << (bits - 1)) - 1
    return drive_rule.DriveGrid(step=(top + offset) / levels, offset=offset, code_max=levels)


def _perceptron(outputs: int, fan_in: int, seed: int, ratio: float):
    """Weights and biases whose ``max|b| / max|w|`` is ``ratio`` exactly up to float rounding."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(outputs, fan_in, generator=g) * 0.05
    b = torch.randn(outputs, generator=g)
    b = b / b.abs().max() * float(w.abs().max()) * ratio
    return w, b


def _error_at(b, drive: float, w_max: float, b_max: float, q_max: int, fractional: bool) -> float:
    """The bias's squared error on the row grid of effective drive ``drive``, summed in bias order (the rule's form)."""
    bb = b.reshape(-1).double()
    peak = max(w_max, b_max / drive, 1e-12)
    scale = q_max / peak if fractional else max(math.floor(q_max / peak), 1.0)
    rows = torch.round(bb * (scale / drive)).clamp(-(q_max + 1), q_max)
    return float(torch.cumsum((bb - rows * (drive / scale)) ** 2, dim=0)[-1])


RATIOS = (0.5, 1.7, 2.0, 3.49, 7.25)


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("bits", (4, 8))
def test_the_rows_are_the_least_count_whose_top_code_holds_the_ratio(ratio, bits):
    rule, grid = _rows_rule(), _grid(bits)
    k = rule.bias_rows(ratio, grid.top)
    assert k >= 1 and k * grid.top >= ratio
    assert k == 1 or (k - 1) * grid.top < ratio


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("bits, wbits", ((4, 4), (8, 8), (8, 4)))
@pytest.mark.parametrize("fractional", (True, False))
def test_the_drive_is_the_least_error_code_of_the_rows_effective_grid(ratio, bits, wbits, fractional):
    rule, grid, q_max = _rows_rule(), _grid(bits), (1 << (wbits - 1)) - 1
    w, b = _perceptron(256, 32, 7, ratio)
    w_max, b_max = float(w.abs().max()), float(b.double().abs().max())
    exact = b_max / w_max
    drive, rows = rule.least_error_drive_and_rows(w, b, grid, q_max, fractional=fractional, **_split())
    assert rows == rule.bias_rows(exact, grid.top)
    candidates = [grid.value(c) for c in range(1, grid.code_max + 1)
                  if grid.value(c) > 0.0 and rows * grid.value(c) >= exact]
    errors = [_error_at(b, rows * d, w_max, b_max, q_max, fractional) for d in candidates]
    least = min(errors)
    assert drive == candidates[errors.index(least)]
    # the weights keep their own grid: the effective drive holds the ratio, so max|b| / (k d) <= max|w|
    assert b_max / (rows * drive) <= w_max * (1 + 1e-12)


def test_one_row_where_a_code_reaches_the_ratio_and_the_drive_is_the_drive_rule_s():
    rule, grid, q_max = _rows_rule(), _grid(8), 127
    w, b = _perceptron(512, 64, 3, 0.4)
    drive, rows = rule.least_error_drive_and_rows(w, b, grid, q_max, fractional=True, **_split())
    assert rows == 1
    assert drive == rule.least_error_drive(w, b, grid, q_max, fractional=True, **_split())


def test_rows_never_widen_the_weight_grid_where_one_row_would():
    """The defect read directly: at one row the least drive below the ratio widens ``max|w|`` to ``max|b| / d``."""
    rule, grid = _rows_rule(), _grid(4)
    w, b = _perceptron(64, 16, 5, 3.49)
    w_max, b_max = float(w.abs().max()), float(b.double().abs().max())
    assert b_max / grid.top > w_max  # one row at the top code widens the grid
    drive, rows = rule.least_error_drive_and_rows(w, b, grid, 7, fractional=True, **_split())
    assert rows >= 2 and max(w_max, b_max / (rows * drive)) == pytest.approx(w_max, rel=1e-12)
