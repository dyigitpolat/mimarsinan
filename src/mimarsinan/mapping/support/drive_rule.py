"""[U-BRANGE2] The always-on row's drive: the code of its reader's input grid at which the row departs least from the bias.

A parameter-encoded bias rides an always-on axon row of the shared weight grid (``transformations.perceptron.
bias_drive``). At the row's drive ``d`` (a code value of the grid its reader reads) the row's weight is ``b / d``, the
shared grid sizes ``p = max(max|w|, max|b| / d)`` and scales by ``s = q_max / p`` (or its integer floor), the row's
code is ``round(b s / d)`` and the projected bias is that code times ``d / s``: the bias lies on a grid of step
``d / s``. So:

* a drive below ``d* = max|b| / max|w|`` widens the weight grid (``p > max|w|``): every weight coarsens;
* a drive at or above ``d*`` leaves the weight grid the weights' own, and the bias grid's step grows with the drive.

A drive tied to the reader's input full scale (the top code value) therefore moved every reader's bias grid with the
range of the grid it reads, a cost no range rule saw (W38 MobileNetV2 under U-BRANGE's F-4: bias grid steps up to 1.62
times, Weight Quantization 165 to 140 of 256). Here the drive is a range site of its own, under the rule every range
site takes: among the code values at or above ``d*`` (the top code value where none is), the one whose row grid gives
the least squared error of the bias itself, ties to the least code.

The rows (the review's m-2 and m-5). Where no code reaches ``d*`` on one row, ``k`` rows each carrying ``b / k`` read
``k d`` and keep the weights' own grid; each extra row is one always-on axon on every core of the perceptron. The rows
are the projection's: one where the document deploys one row (no bias-row splitting declared: the top code value then
widens the weight grid least), else the least ``k`` whose top code value holds ``d*``, at least a declared row count,
and at most the declared cores' budget (the largest core's axons less one data axon); a perceptron that needs more is
refused by name (``DriveRowBudgetRefusal``).

Exact (the proven property): the drive is the full search's bit for bit, where the full search sums every code's
squared departures in bias order and takes the least, ties to the least code. The search bounds its work whatever the
order of the biases (m-4): each code's running sum is read in one order fixed by the biases' own magnitudes, the
largest first, and a code is dropped once that sum exceeds the least error found times ``1 + margin`` (every term is
a square, and two float64 sums of the same non-negative terms in two orders differ by at most ``2 n eps`` of their
value: the margin covers it); the codes that survive are summed again in bias order, and that sum decides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from mimarsinan.common.refusal import ToolchainRefusal

__all__ = ["DriveGrid", "DriveRowBudgetRefusal", "RowBudget", "bias_rows", "least_error_drive",
           "least_error_drive_and_rows", "rows_for"]

#: (code, bias) pairs read together: the transient's bound, whatever the codes and the biases.
_BLOCK_PAIRS = 1 << 18
#: A code's running sum starts on this many biases, the chunk doubling while the code can still be the least.
_CHUNK = 64
#: Codes read in the first block (the least codes, from which the least error is known first) and in each later one.
_FIRST_CODES = 64
_CODES = 1024
_EPS = 2.0 ** -53


class DriveRowBudgetRefusal(ToolchainRefusal, ValueError):
    """A bias needs more always-on rows than the row budget declares (the deployed row count or the cores' axons)."""


@dataclass(frozen=True)
class DriveGrid:
    """The input grid a row's drive is a code value of: values ``k step - offset`` for the codes ``1 <= k <=
    code_max`` whose value is positive (the reader's DAC writes them)."""

    step: float
    offset: float
    code_max: int

    def value(self, code: int) -> float:
        return code * self.step - self.offset

    @property
    def top(self) -> float:
        """The largest code value (the grid's full scale, the drive of the former rule)."""
        return self.value(self.code_max)

    def holds(self, drive: float) -> bool:
        """Whether ``drive`` is a positive code value of this grid, bit for bit the float the stamp writes."""
        if not (self.step > 0.0 and drive > 0.0):
            return False
        code = int(round((drive + self.offset) / self.step))
        return 1 <= code <= self.code_max and self.value(code) == drive


@dataclass(frozen=True)
class RowBudget:
    """The rows the deployment gives a bias: ``split`` False deploys one row (no bias-row splitting declared);
    ``split`` True deploys the rows the bias needs, at least ``floor`` (a declared row count) and at most ``axons - 1``
    (the largest declared core's axons, one kept for data; ``None`` declares no core)."""

    split: bool = False
    floor: int = 1
    axons: Optional[int] = None

    @property
    def limit(self) -> Optional[int]:
        return None if self.axons is None else max(0, int(self.axons) - 1)


def bias_rows(ratio: float, top: float) -> int:
    """[U-BRANGE2 m-2] The always-on rows a bias needs on the weights' own grid: the least ``k`` with ``k`` rows at the
    top code value holding ``max|b| / max|w|`` (1 where the top does)."""
    if not (ratio > 0.0 and top > 0.0) or top >= ratio:
        return 1
    k = max(2, int(math.ceil(ratio / top)))
    while k * top < ratio:
        k += 1
    return k


def rows_for(ratio: float, top: float, budget: RowBudget) -> int:
    """The rows the drive site reads under ``budget`` (module docstring), refused by name above the cores' budget."""
    if not budget.split:
        return 1
    rows = max(int(budget.floor), bias_rows(ratio, top))
    limit = budget.limit
    if budget.axons is not None and limit is not None and rows > limit:
        raise DriveRowBudgetRefusal(
            f"the bias needs {rows} always-on rows (max|b| / max|w| = {ratio:.6g} over the reader's top code value "
            f"{top:.6g}, declared at least {int(budget.floor)}), above the row budget {limit} of the declared cores "
            f"({int(budget.axons)} axons, one kept for data)")
    return rows


def _codes(grid: DriveGrid, ratio: float):
    """The candidate codes, ascending, and their values (``grid.value``'s float, ``k step - offset`` in float64): every
    code whose value is positive and at or above ``ratio``."""
    first = max(1, int(math.floor((ratio + grid.offset) / grid.step)))
    k = torch.arange(min(first, grid.code_max + 1), grid.code_max + 1, dtype=torch.int64)
    values = k.to(torch.float64) * grid.step - grid.offset
    keep = (values > 0.0) & (values >= ratio)
    return k[keep], values[keep]


def _row_errors(b: torch.Tensor, drives: torch.Tensor, w_max: float, b_max: float, q_max: int, fractional: bool,
                running: torch.Tensor) -> torch.Tensor:
    """The running sums carried through one chunk of biases ``b`` for ``drives``: each term the squared departure of a
    bias from its row's code times ``d / s``, added in the chunk's order after the running sum."""
    peak = torch.clamp(torch.maximum(torch.full_like(drives, w_max), b_max / drives), min=1e-12)
    scale = q_max / peak if fractional else torch.clamp(torch.floor(q_max / peak), min=1.0)
    rows = torch.round(b[None, :] * (scale / drives)[:, None]).clamp(-(q_max + 1), q_max)
    terms = (b[None, :] - rows * (drives / scale)[:, None]) ** 2
    return torch.cumsum(torch.cat((running[:, None], terms), dim=1), dim=1)[:, -1]


def _sums(b: torch.Tensor, drives: torch.Tensor, w_max: float, b_max: float, q_max: int, fractional: bool,
          stop: Optional[float] = None):
    """Each drive's error summed over ``b`` in its order, in chunks (64 first, doubling, at most ``_BLOCK_PAIRS``
    pairs); with ``stop``, a drive whose running sum exceeds it is dropped. Returns the surviving indices and sums."""
    live = torch.arange(len(drives))
    running = torch.zeros(len(drives), dtype=torch.float64)
    first, chunk = 0, _CHUNK
    while first < b.numel() and live.numel():
        take = max(1, min(chunk, _BLOCK_PAIRS // live.numel()))
        running = _row_errors(b[first:first + take], drives[live], w_max, b_max, q_max, fractional, running)
        first, chunk = first + take, 2 * chunk
        if stop is not None and math.isfinite(stop):
            keep = running <= stop
            if not bool(keep.all()):
                live, running = live[keep], running[keep]
    return live, running


def least_error_drive(weight: torch.Tensor, bias: torch.Tensor, grid: DriveGrid, q_max: int, *, fractional: bool,
                      budget: RowBudget) -> float:
    """The row's drive: :func:`least_error_drive_and_rows`'s drive."""
    return least_error_drive_and_rows(weight, bias, grid, q_max, fractional=fractional, budget=budget)[0]


def least_error_drive_and_rows(weight: torch.Tensor, bias: torch.Tensor, grid: DriveGrid, q_max: int, *,
                               fractional: bool, budget: RowBudget):
    """The drive site's rule with its row count (module docstring): ``(drive, rows)``; ``(top, 1)`` where the row
    carries no value; with one row and no code at or above ``max|b| / max|w|``, the top code value."""
    b = bias.detach().reshape(-1).to(torch.float64).cpu()
    w_max = float(weight.detach().abs().max()) if weight.numel() else 0.0
    b_max = float(b.abs().max()) if b.numel() else 0.0
    if not (b_max > 0.0 and w_max > 0.0) or grid.top <= 0.0:
        return grid.top, 1
    ratio = b_max / w_max
    rows = rows_for(ratio, grid.top, budget)
    codes, values = _codes(grid, ratio / rows)
    codes, values = codes[rows * values >= ratio], values[rows * values >= ratio]
    if codes.numel() == 0:
        return grid.top, rows
    values = values * rows if rows > 1 else values  # the rows' effective drive: k d (one row: d itself, bit for bit)
    screen = b[torch.argsort(b.abs(), descending=True, stable=True)]  # the bound's order: the largest biases first
    margin = max(1e-12, 8.0 * b.numel() * _EPS)
    best_error, best_code = math.inf, None
    start, size = 0, _FIRST_CODES
    while start < codes.numel():
        live, drives = codes[start:start + size], values[start:start + size]
        start, size = start + size, _CODES
        kept, screened = _sums(screen, drives, w_max, b_max, q_max, fractional,
                               stop=best_error * (1.0 + margin) if math.isfinite(best_error) else None)
        if not kept.numel():
            continue
        near = screened <= (1.0 + margin) * min(best_error, float(screened.min()) * (1.0 + margin))
        exact_at = kept[near]
        _all, exact = _sums(b, drives[exact_at], w_max, b_max, q_max, fractional)  # in bias order: the full search's
        at = int(torch.argmin(exact))  # the first least: ties to the least code
        if float(exact[at]) < best_error:  # a later code loses a tie to an earlier one
            best_error, best_code = float(exact[at]), int(live[exact_at[at]])
    return grid.value(best_code if best_code is not None else grid.code_max), rows
