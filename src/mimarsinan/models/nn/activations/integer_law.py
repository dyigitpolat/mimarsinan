"""The integer readout law (E282.L): one rounding rule, torch and int64 forms of one law."""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal

#: fp64 holds every integer of magnitude below this bound exactly; above it the fp64 executor is not exact.
FP64_EXACT_BOUND = 1 << 53


class IntegerLawRangeError(ToolchainRefusal, ValueError):
    """A law whose |z| bound reaches 2^53, where fp64 execution of the integer law stops being exact."""


class OffGridInputError(ToolchainRefusal, RuntimeError):
    """An input of a law-carrying core that does not lie on the core's declared input grid."""


def grid_codes(values: torch.Tensor, step: torch.Tensor, *, where: str) -> torch.Tensor:
    """fp64 integer codes of ``values`` on the grid of ``step`` (an infinite step reads code 0), each
    value asserted on its grid within the rounding of ``values``' own dtype."""
    quotient = values.to(torch.float64) / step.to(torch.float64)
    codes = torch.round(quotient)
    tolerance = 8.0 * torch.finfo(values.dtype).eps * (codes.abs() + 1.0)
    if bool(((quotient - codes).abs() > tolerance).any()):
        worst = float((quotient - codes).abs().max())
        raise OffGridInputError(
            f"{where}: an input lies off its declared grid (worst distance {worst:.3e} codes); the integer "
            f"law needs every data axon on one grid"
        )
    return codes


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Nearest integer, ties away from zero (E102); bit-equal to ``torch.round`` off ties."""
    whole = torch.trunc(x)
    tie = (x - whole).abs() == 0.5
    return torch.where(tie, whole + torch.sign(x), torch.round(x))


def rnd_shift(z: torch.Tensor, shift) -> torch.Tensor:
    """``rnd_s(z) = (z + 2^(s-1) - [z < 0]) >> s`` on integer-valued fp64 ``z`` (|z| < 2^53).

    Scaling by ``2^-s`` is exact, so this is ``round_half_away(z / 2^s)``: the one rounding rule.
    """
    exponent = torch.as_tensor(shift, device=z.device).to(torch.int64)
    return round_half_away(torch.ldexp(z, -exponent))


def rnd_shift_int64(z: np.ndarray, shift) -> np.ndarray:
    """The hardware shift form on int64: ``(z + 2^(s-1) - [z < 0]) >> s``, ``s = 0`` the identity."""
    z = np.asarray(z, dtype=np.int64)
    s = np.broadcast_to(np.asarray(shift, dtype=np.int64), z.shape)
    half = np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)
    negative = np.where(s > 0, (z < 0).astype(np.int64), 0)
    return np.right_shift(z + half - negative, s)


def adc(acc: torch.Tensor, adc_shift, adc_levels) -> torch.Tensor:
    """f_adc on the integer accumulator: ``2^t clamp(rnd_t(acc), -L, L)``; the identity at t = 0, L >= |acc|."""
    levels = torch.as_tensor(adc_levels, dtype=acc.dtype, device=acc.device)
    exponent = torch.as_tensor(adc_shift, device=acc.device).to(torch.int64)
    codes = torch.clamp(rnd_shift(acc, exponent), -levels, levels)
    return torch.ldexp(codes, exponent)


def adc_int64(acc: np.ndarray, adc_shift, adc_levels) -> np.ndarray:
    """f_adc, int64 form."""
    shift = np.asarray(adc_shift, dtype=np.int64)
    levels = np.asarray(adc_levels, dtype=np.int64)
    codes = np.clip(rnd_shift_int64(acc, shift), -levels, levels)
    return np.left_shift(codes, np.broadcast_to(shift, codes.shape))


def apply_law(
    acc: torch.Tensor, *, m0: Any, bias: Any, shift: Any, lo: Any, hi: Any, adc_shift: Any = 0,
    adc_levels: Any = None, operands: Iterable[Tuple[torch.Tensor, Any]] = (),
) -> torch.Tensor:
    """``y = clamp(rnd_s(M0 f_adc(acc) + sum_k M_k q_k + B), lo, hi)`` on integer-valued fp64 tensors."""
    def t(v):
        return torch.as_tensor(v, dtype=acc.dtype, device=acc.device)

    read = acc if adc_levels is None else adc(acc, adc_shift, adc_levels)
    z = read * t(m0) + t(bias)
    for q, multiplier in operands:
        z = z + q * t(multiplier)
    return torch.clamp(rnd_shift(z, shift), t(lo), t(hi))


def apply_law_int64(
    acc: np.ndarray, *, m0: Any, bias: Any, shift: Any, lo: Any, hi: Any, adc_shift: Any = 0,
    adc_levels: Any = None, operands: Iterable[Tuple[np.ndarray, Any]] = (),
) -> np.ndarray:
    """The same law in exact int64 arithmetic (the golden's form)."""
    def i(v):
        return np.asarray(v, dtype=np.int64)

    acc = i(acc)
    read = acc if adc_levels is None else adc_int64(acc, adc_shift, adc_levels)
    z = read * i(m0) + i(bias)
    for q, multiplier in operands:
        z = z + i(q) * i(multiplier)
    return np.clip(rnd_shift_int64(z, shift), i(lo), i(hi))


def instance_reduce(acc: torch.Tensor, m0, *, mode: str, dim: int) -> torch.Tensor:
    """Reduce instance accumulators before one law: sum, or max (min where M0 < 0, a monotone law)."""
    if mode == "sum":
        return acc.sum(dim)
    if mode != "max":
        raise ValueError(f"instance_reduce: mode must be 'sum' or 'max', got {mode!r}")
    positive = torch.as_tensor(m0, device=acc.device) > 0
    return torch.where(positive, acc.amax(dim), acc.amin(dim))


def affine_after_clamp_bounds(u: torch.Tensor, c: torch.Tensor, lo: float, hi: float):
    """Bounds folding ``u clamp(z, lo, hi) + c`` into ``clamp(u z + c, lo', hi')`` (BN after ReLU or ReLU6).

    ``u > 0`` keeps the order of the bounds and ``u < 0`` swaps them; the weights take ``u``.
    """
    first = torch.where(torch.isinf(torch.as_tensor(lo)), -torch.sign(u) * float("inf"), u * lo) + c
    second = torch.where(torch.isinf(torch.as_tensor(hi)), torch.sign(u) * float("inf"), u * hi) + c
    return torch.where(u > 0, first, second), torch.where(u > 0, second, first)


def grid_levels(bits: int) -> int:
    """Positive codes of the symmetric signed grid of ``bits``: ``2^(b-1) - 1``."""
    return (1 << (int(bits) - 1)) - 1


def saturation_reach(m0: int, bias: int, shift: int, hi: int) -> int:
    """The largest |acc| whose law output ``clamp(rnd_s(M0 acc + B), -hi, hi)`` (M0 > 0) is not yet saturated:
    every larger acc reads ``hi`` and every smaller ``-acc`` reads ``-hi``. Exact integer thresholds of
    the half-away shift: ``rnd_s(z) >= hi`` iff ``z >= hi 2^s - h`` and ``rnd_s(z) <= -hi`` iff
    ``z <= -hi 2^s + h``, with ``h = 2^(s-1)`` (0 at s = 0)."""
    half = (1 << (shift - 1)) if shift > 0 else 0
    acc_hi = -((bias - ((hi << shift) - half)) // m0)  # the smallest acc that reads hi
    acc_lo = (-(hi << shift) + half - bias) // m0      # the largest acc that reads -hi
    return max(abs(acc_hi), abs(acc_lo))


def accumulator_bounds(weights, input_levels: int) -> List[int]:
    """Per column, the exact bound of ``|sum_i W_ij q_i|`` over codes ``|q_i| <= input_levels``:
    ``sum_i |W_ij| input_levels`` on the integer weight codes of the data rows (q_min = -2^(b-1) included)."""
    magnitudes = np.abs(np.asarray(weights, dtype=np.float64))
    if magnitudes.ndim != 2 or not np.array_equal(magnitudes, np.round(magnitudes)):
        raise ValueError("accumulator_bounds: weights must be a matrix of integer codes")
    return [int(total) * int(input_levels) for total in magnitudes.sum(axis=0).tolist()]


def accumulator_width(acc_bound: int) -> int:
    """A: the signed bit width holding every accumulator of magnitude at most ``acc_bound``."""
    return int(acc_bound).bit_length() + 1


def law_magnitude_bound(acc_bound, m0, bias, adc_shift) -> int:
    """``max_j |M0_j| (acc_bound_j + 2^t_j) + |B_j|`` in exact integers over every column (each argument a
    scalar or one value per column): a bound of |z|."""
    columns = np.broadcast(*(np.asarray(v).reshape(-1) for v in (acc_bound, m0, bias, adc_shift)))
    return max(
        (abs(int(m)) * (int(a) + (1 << int(t))) + abs(int(b)) for a, m, b, t in columns),
        default=0,
    )
