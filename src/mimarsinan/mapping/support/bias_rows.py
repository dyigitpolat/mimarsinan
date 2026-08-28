"""Bias-row splitting: the k always-on core-matrix rows a param-encoded bias rides."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

BIAS_ROW_SPLITTING_OFF = "off"
BIAS_ROW_SPLITTING_AUTO = "auto"
BIAS_ROW_SPLITTING_FIXED = "fixed"
BIAS_ROW_SPLITTING_MODES = (
    BIAS_ROW_SPLITTING_OFF,
    BIAS_ROW_SPLITTING_AUTO,
    BIAS_ROW_SPLITTING_FIXED,
)

MODE_KEY = "bias_row_splitting"
ROWS_KEY = "bias_rows_per_perceptron"

# The integer-ratio snap is exact by construction; this admits only float noise.
_RATIO_TOL = 1e-6


@dataclass(frozen=True)
class BiasRowSplitting:
    """The resolved splitting decision: mode, the fixed override, and whether
    the platform actually splits (a core with an on-chip bias lane never does)."""

    mode: str
    fixed_rows: int
    active: bool

    @property
    def rows_override(self) -> int | None:
        """The declared k, or None when the bound computes it per layer."""
        if self.active and self.mode == BIAS_ROW_SPLITTING_FIXED:
            return int(self.fixed_rows)
        return None


def bias_row_demand(b_max: Any, weight_scale: Any, q_max: float) -> Any:
    """Registers of bias demand, less the lattice tolerance — THE k formula's
    body, generic over python floats and torch tensors so the QAT projection
    evaluates it without a host sync.
    """
    return b_max * weight_scale / float(q_max) - _RATIO_TOL


def bias_row_bound(b_max: float, weight_scale: float, q_max: float) -> int:
    """THE definitive computed bound: ``k = ceil(max_j |b_j| * s_w / q_max)``.

    ``s_w`` is the WEIGHT-ONLY grid scale and ``q_max`` the weight register's
    positive range, so k is the fewest always-on rows whose ±q_max integer
    weights can carry the largest bias in the layer.
    """
    if q_max <= 0.0:
        raise ValueError(f"q_max must be positive, got {q_max}")
    if weight_scale <= 0.0:
        raise ValueError(f"weight_scale must be positive, got {weight_scale}")
    return max(1, int(math.ceil(bias_row_demand(float(b_max), float(weight_scale), q_max))))


def _as_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def bias_rows_from_scales(
    bias_scale: Any, parameter_scale: Any, *, name: Any = None
) -> int:
    """Recover k at the mapping SSOT from the grids the WQ install stamped.

    The two-scale projection snaps ``bias_scale = parameter_scale / k`` with
    integer k, so the installed scales carry the row count with no side channel;
    a shared grid recovers k=1 and maps byte-identically to the legacy path.
    """
    if bias_scale is None:
        return 1
    bs = _as_float(bias_scale)
    if bs <= 0.0:
        return 1
    ratio = _as_float(parameter_scale) / bs
    rows = round(ratio)
    if rows < 1 or abs(ratio - rows) > _RATIO_TOL:
        raise ValueError(
            f"bias-row splitting: {name or '<unnamed>'} carries "
            f"parameter_scale/bias_scale = {ratio!r}, which is not a positive "
            f"integer row count; the WQ two-scale projection must snap the "
            f"bias grid to an integer multiple of the weight grid."
        )
    return int(rows)


def param_encoded_bias_rows(
    bias_scale: Any, parameter_scale: Any, *, hardware_bias: bool, name: Any = None
) -> int:
    """Always-on rows one perceptron's bias occupies on THIS platform.

    No capability flag: the installed grids ARE the authority. A bias grid
    coarser than the weight grid can only come from a weight-only projection,
    which a param-encoded platform reaches only through bias-row splitting —
    so the scales answer the question, every mapper construction site agrees
    without plumbing, and a shared grid answers 1 byte-identically. The
    integer-ratio invariant is the backstop and it refuses loud.
    """
    if hardware_bias:
        return 1
    return bias_rows_from_scales(bias_scale, parameter_scale, name=name)


def split_bias_row_values(biases: np.ndarray, rows: int) -> np.ndarray:
    """The ``(rows, out_features)`` always-on block carrying ``biases``.

    Equal shares: with the two-scale install ``b_j = k * bias_int_j / s_w``, so
    every row emits exactly ``bias_int_j`` and the k integer weights sum to
    ``round(b_j * s_w)`` — the bias is bit-exact on the deployed grid.
    """
    if rows < 1:
        raise ValueError(f"bias rows must be >= 1, got {rows}")
    flat = np.asarray(biases, dtype=float).flatten()
    return np.tile(flat / float(rows), (int(rows), 1))


def core_matrix_with_bias_rows(
    weights_transposed: np.ndarray, biases: np.ndarray, rows: int
) -> np.ndarray:
    """An ``(in_features + rows, out_features)`` core matrix: the transposed
    weights above the always-on block. THE param-encoded assembly, shared by
    every emission site so a bank and an owned core cannot lay out differently.
    """
    w_t = np.asarray(weights_transposed, dtype=float)
    bias_block = split_bias_row_values(biases, rows)
    matrix = np.empty((w_t.shape[0] + bias_block.shape[0], w_t.shape[1]), dtype=float)
    matrix[: w_t.shape[0], :] = w_t
    matrix[w_t.shape[0] :, :] = bias_block
    return matrix
