"""The TTFS float membrane, a fixed function of a neuron's own nonzero terms, whatever packing holds them.

A neuron off the integer chip's lattice (``membrane_lattice``: a raw input of 8-bit codes ``k/255`` in float64, a
float chip) has a float membrane. A dot product in the hard core's own axon order would round by its layout: the
per-axon split's W+/W- rows, a co-resident soft core at an axon offset and a fused core all reorder or zero-pad the
sum, and near an exact rational staircase tie one rounding lands a level away from the identity program.

Here every input and weight is split into windows of ``WINDOW_BITS`` bits on one global power-of-two grid (window
``j`` holds the bits in ``[2^(16 j), 2^(16 (j + 1)))``; each value's own windows, never a scale of its row). The
product sum of an input window and a weight window is a multiple of ``2^(16 (j + l))`` below
``2^(16 (j + l) + 53)`` for at most ``MAX_TERMS`` terms, so float64 holds it exactly in any order and with any
zeros. Those exact window sums are added in one global window order. A neuron's membrane is so a fixed function of
its own ``(input, weight)`` terms and bias, bit for bit the same in every packing of one IR. It is the exact sum
where float64 holds every partial sum (float32 codes on integer weights), else within a few roundings of it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal

#: Bits per window of the global grid; two windows' product and a sum of ``MAX_TERMS`` of them fit 53 bits.
WINDOW_BITS = 16
MAX_TERMS = 2 ** (53 - 2 * WINDOW_BITS)
#: The lowest window whose upper edge float64 holds (2^-1072); it also takes every bit below it.
_LOWEST_WINDOW = -68
#: A window product is a multiple of 2^(16 (j + l)): float64 holds it from the subnormal unit 2^-1074 up, and a sum
#: of ``MAX_TERMS`` of them below 2^1024.
_PRODUCT_WINDOWS = (-67, 60)


class FloatMembraneRangeError(ToolchainRefusal, ValueError):
    """A float membrane past the exact window sums: more than ``MAX_TERMS`` terms, or an input and weight whose window
    product falls below float64's subnormal unit or past its range."""


def _windows(a: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """``[(j, a_j)]`` top window first: ``a_j`` holds every entry's bits in window ``j`` (the lowest window also
    every bit below it), so the ``a_j`` sum to ``a`` exactly; windows that hold no bit are left out."""
    nonzero = a[a != 0]
    if nonzero.size == 0:
        return []
    _, exponent = np.frexp(nonzero)  # |value| < 2^e, its lowest bit at or above 2^(e - 53)
    top = (int(exponent.max()) - 1) // WINDOW_BITS
    bottom = max((int(exponent.min()) - 53) // WINDOW_BITS, _LOWEST_WINDOW)
    out: List[Tuple[int, np.ndarray]] = []
    upper = a
    for j in range(top, bottom, -1):
        lower = np.fmod(a, 2.0 ** (WINDOW_BITS * j))  # exact: the bits of a below window j
        window = upper - lower  # exact: the bits of a in window j
        if window.any():
            out.append((j, window))
        upper = lower
    if upper.any():
        out.append((bottom, upper))
    return out


def _exact_window_membrane(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    v = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
    w_windows = _windows(w)
    for j, x_j in _windows(x):
        for l, w_l in w_windows:
            if not _PRODUCT_WINDOWS[0] <= j + l <= _PRODUCT_WINDOWS[1]:
                if bool(((x_j != 0).astype(np.float64) @ (w_l != 0).T.astype(np.float64)).any()):
                    raise FloatMembraneRangeError(
                        f"an input window 2^{WINDOW_BITS * j} and a weight window 2^{WINDOW_BITS * l} meet on one "
                        f"neuron: their product leaves float64's exact range")
                continue
            v = v + x_j @ w_l.T  # the product sum is exact; the window sums add in one global order
    return v


def float_membrane(x: np.ndarray, w: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """The ``(B, N)`` float membrane of a core's neurons, ``w`` ``(N, A)``, on its input signal ``x`` ``(B, A)``,
    with the hardware bias (one more term on an always-on input, as an always-on row carries it)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if bias is not None:
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        w = np.concatenate([w, np.asarray(bias, dtype=np.float64).reshape(-1, 1)], axis=1)
    if x.shape[1] > MAX_TERMS:
        raise FloatMembraneRangeError(
            f"{x.shape[1]} terms per neuron: a window product sum is exact for at most {MAX_TERMS}")
    finite_x, finite_w = np.isfinite(x), np.isfinite(w)
    if bool(finite_x.all()) and bool(finite_w.all()):
        return _exact_window_membrane(x, w)
    # A non-finite term propagates as in any dot product (NaN times 0 is NaN): such a sample or neuron keeps it.
    v = _exact_window_membrane(np.where(finite_x, x, 0.0), np.where(finite_w, w, 0.0))
    finite_rows = np.asarray(np.all(finite_x, axis=1), dtype=bool)
    finite_neurons = np.asarray(np.all(finite_w, axis=1), dtype=bool)
    touched = ~finite_rows[:, None] | ~finite_neurons[None, :]
    return np.where(touched, x @ w.T, v)
