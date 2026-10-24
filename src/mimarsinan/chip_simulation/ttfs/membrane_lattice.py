"""The integer chip's TTFS membrane lattice, decided per neuron from that neuron's own inputs.

A neuron whose weights, bias and threshold are integral and whose every nonzero-weight input lies on the ``1/(2S)``
grid (a staircase level ``j/S``, a midpoint ``(2k+1)/(2S)``, the always-on row) has its exact membrane on that grid:
``V = m / (2S)`` with ``m`` an integer, whatever order the dot product sums in. The executor computes ``m`` from the
integer inputs and weights (exact in float64 below 2^53, so in any summation order) and decides the staircase from
``m`` and theta exactly (``ttfs_quantized_staircase_lattice_np``). Every other neuron (one reading an input off the
grid, such as a raw input of 8-bit codes ``k/255`` in float64, or a float chip) takes the float membrane of
``float_membrane``, a fixed function of its own terms, so its staircase too is the same in every packing.

The decision reads only the neuron's own weights and inputs, so every packing of one IR decides alike: a soft core
co-resident at an axon offset, the per-axon split's W+/W- rows and a fused core all reorder the float sum, and none
moves an exact membrane. A decision per segment or per hard core would not: a segment whose entry is off the grid
ran every core on float sums, and a hard core mixes the inputs of its co-resident soft cores.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Detects float dust on the ``1/(2S)`` input grid (a level ``j/S`` or a midpoint rounded through float32); not a
#: budget: an input farther off the grid is continuous and keeps the float path.
GRID_DUST = 1e-4
#: float64 holds every integer below 2^53, so an integer dot product inside it sums exactly in any order.
EXACT_INTEGER_RANGE = float(2 ** 53)


def lattice_membrane(seg, ci: int, x: np.ndarray, s: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(on_lattice, m)`` for core ``ci`` on its input signal ``x`` ``(B, A)`` at window ``s``: a ``(B, N)`` mask of
    the neurons whose exact membrane is ``m / (2s)``, and the integer numerators ``m`` ``(B, N)``; None when no
    neuron of the core is on the lattice."""
    integral = seg.integral_neurons(ci)
    if not bool(integral.any()):
        return None
    w = np.asarray(seg.core_params[ci], dtype=np.float64)
    q = np.asarray(x, dtype=np.float64) * (2 * s)
    off_grid = np.abs(q - np.rint(q)) > GRID_DUST
    if bool(off_grid.any()):
        reads_off_grid = (off_grid.astype(np.float64) @ (w != 0).T.astype(np.float64)) > 0
        on_lattice = integral[None, :] & ~reads_off_grid
    else:
        on_lattice = np.broadcast_to(integral[None, :], (q.shape[0], w.shape[0]))
    if not bool(on_lattice.any()):
        return None
    q_int, w_int = np.rint(q), np.rint(w)
    bias = seg.hw_biases[ci]
    b_int = None if bias is None else (2 * s) * np.rint(np.asarray(bias, dtype=np.float64))
    # Exact only inside float64's integer range (H8's accumulator bound): a neuron whose partial sums may pass 2^53
    # keeps the float path, where H8 names the departure.
    bound = np.abs(q_int) @ np.abs(w_int).T + (0.0 if b_int is None else np.abs(b_int))
    on_lattice = on_lattice & (bound < EXACT_INTEGER_RANGE)
    if not bool(on_lattice.any()):
        return None
    m = q_int @ w_int.T
    if b_int is not None:
        m = m + b_int
    return on_lattice, m
