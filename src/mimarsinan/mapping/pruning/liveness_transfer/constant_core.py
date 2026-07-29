"""NeuralCore constant transfer: the exact crossbar fold, in both directions.

FORWARD (rows -> bias). An axon row ``r`` whose line is ``CONST(c)`` always
contributes ``c * W[r, :]`` to the pre-activation, so that contribution can be
moved onto the core's CONSTANT CARRIER and row ``r`` eliminated EXACTLY:

    (x @ W + b) / theta      with x[r] == c for all t
  = (x' @ W' + b') / theta   where row r is gone and c * W[r, :] sits on the carrier.

The carrier is whatever already encodes a constant on that core — an always-on
axon row (``IRSource(-3)``, the crossbar encoding of a bias: ``chip_export``
folds exactly such a row into ``hardware_bias``) or an owned ``hardware_bias``
vector. No carrier is ever INVENTED, so the fold never assumes a hardware
capability the instance does not already use; a core without one only folds
``CONST(0)`` rows, which need no carrier at all and are the pre-existing
elimination.

Bank-backed cores fold ``CONST(0)`` only. Their axon rows are SHARED physical
structure whose removal is decided by the W3c intersection rule (dead in every
sharing instance), and the compaction path projects the BANK-level row mask
onto every instance — so a per-instance bias fold of a row that survives
physically would double-count. Their columns still descend normally, so a
bank-backed core can still be a CONST producer for its consumers.

UPWARD (rows -> column). Once every row contributing to a neuron column has
been ELIMINATED or FOLDED, the only thing left driving that column is the
carrier, so its output is exactly ``(W[carrier, j] + b_j) / theta`` — a
constant. Phrasing the rule on the eliminated/folded state rather than on the
raw line values is deliberate: it makes "column becomes constant" cost exactly
one propagation wave after "its rows die", which is the hop accounting the
depth replay and the ``masked <= closure <= cascade`` ordering are built on.
BIAS_ONLY cores are precisely the case where the carrier is the only
contributor left, so a bias-only core collapsing into its consumers is not a
special rule — it is this rule followed by ordinary orphan elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Dict, Sequence, Set

import numpy as np

from mimarsinan.mapping.ir import IRSource, NeuralCore

__all__ = [
    "CARRIER_BIAS",
    "CARRIER_ROW",
    "ConstantCarrier",
    "CoreConstantFacts",
    "derive_core_constants",
    "resolve_constant_carrier",
]

CARRIER_ROW = "row"
CARRIER_BIAS = "bias"


@dataclass(frozen=True)
class ConstantCarrier:
    """A core's constant carrier: what the column rule reads, and may it be written?

    ``writable`` is False for bank-backed cores. Their carrier (the shared
    bank's always-on row, or a bias VIEW into the bank) is shared physical
    structure, and their axon rows are removed by the BANK-level mask, so a
    per-instance write would either corrupt a sharer or double-count a row
    that survives physically. Reading it is always safe, which is what lets a
    bank-backed core still be a CONST producer for its consumers.
    """

    kind: str
    row: int = -1          # -1 when the carrier is the bias vector
    writable: bool = True


@dataclass(frozen=True)
class CoreConstantFacts:
    """One core's constant facts for the current lattice state."""

    dead_rows: Set[int]
    fold_rows: Dict[int, float]
    column_values: Dict[int, float]


def resolve_constant_carrier(
    node: NeuralCore,
    *,
    pruned_rows: AbstractSet[int],
    exempt_rows: AbstractSet[int],
) -> ConstantCarrier | None:
    """The core's existing constant carrier, or None when it has none.

    An always-on axon row wins over ``hardware_bias`` because it needs no
    representation change at all. Writability follows matrix ownership (see
    :class:`ConstantCarrier`).
    """
    writable = node.core_matrix is not None
    for i, src in enumerate(node.input_sources.flatten()):
        if (
            isinstance(src, IRSource)
            and src.is_always_on()
            and i not in pruned_rows
            and i not in exempt_rows
        ):
            return ConstantCarrier(kind=CARRIER_ROW, row=i, writable=writable)
    bias = getattr(node, "hardware_bias", None)
    if bias is not None and np.asarray(bias).size:
        return ConstantCarrier(kind=CARRIER_BIAS, writable=writable)
    return None


def derive_core_constants(
    *,
    matrix: np.ndarray,
    bias: np.ndarray | None,
    threshold: float,
    row_values: Sequence[float | None],
    pruned_rows: AbstractSet[int],
    pruned_cols: AbstractSet[int],
    exempt_rows: AbstractSet[int],
    carrier: ConstantCarrier | None,
    admits_nonzero: bool,
) -> CoreConstantFacts:
    """Derive dead rows, foldable rows, and CONST columns for one core.

    ``matrix`` / ``bias`` must be the EFFECTIVE (post-fold) structures: the
    column rule reads the carrier only, so a fold that has already migrated
    onto it must be visible here or the constant would be wrong.
    """
    n_axons, n_neurons = matrix.shape
    n_rows = min(n_axons, len(row_values))
    const_mask = np.zeros(n_axons, dtype=bool)
    const_val = np.zeros(n_axons, dtype=np.float64)
    for i in range(n_rows):
        v = row_values[i]
        if v is not None:
            const_mask[i] = True
            const_val[i] = float(v)

    live = np.ones(n_axons, dtype=bool)
    if pruned_rows:
        live[np.fromiter(
            (i for i in pruned_rows if 0 <= i < n_axons), dtype=np.int64
        )] = False

    foldable = (
        admits_nonzero and carrier is not None and carrier.writable
    )
    dead_rows: Set[int] = set()
    fold_rows: Dict[int, float] = {}
    carrier_row = int(carrier.row) if (carrier and carrier.kind == CARRIER_ROW) else -1
    for i in np.flatnonzero(live & const_mask):
        i = int(i)
        if i in exempt_rows or i == carrier_row:
            continue
        if const_val[i] == 0.0:
            dead_rows.add(i)
        elif foldable:
            fold_rows[i] = float(const_val[i])

    # A column is CONST once every contributing row is RESOLVED: eliminated
    # (contributes exactly 0), the carrier, or a CONST row that can never be
    # eliminated (a bank-backed core's rows: shared physical structure). A
    # CONST row this sweep is about to kill or fold deliberately still BLOCKS,
    # so "column becomes constant" always costs one wave after "its rows die"
    # — that is the hop accounting the depth replay is built on. Only EXACT
    # zeros count as "no contribution": a near-zero weight still lands in the
    # sum, and bit-exactness is the whole point of the fold.
    stuck_const = live & const_mask & (const_val != 0.0) & (not foldable)
    if carrier_row >= 0:
        stuck_const[carrier_row] = False
    unresolved = live & ~stuck_const
    if carrier_row >= 0:
        unresolved[carrier_row] = False
    if unresolved.any():
        blocked = (matrix[unresolved, :] != 0.0).any(axis=0)
    else:
        blocked = np.zeros(n_neurons, dtype=bool)

    contributors = np.where(stuck_const, const_val, 0.0)
    if carrier_row >= 0:
        contributors[carrier_row] = 1.0
    theta = float(threshold)
    if theta == 0.0:
        return CoreConstantFacts(dead_rows, fold_rows, {})
    pre = _pre_activation(contributors, matrix, bias, theta, np.float64)
    # DTYPE INDEPENDENCE, the same gate the ComputeOp probe applies: the
    # deployed program runs fp32 and the certificate an fp64 twin, so only a
    # constant both dtypes agree on can be folded bit-exactly. (On the
    # dyadic-grid instances the certificate requires, they always agree.)
    pre32 = _pre_activation(contributors, matrix, bias, theta, np.float32)
    stable = pre32.astype(np.float64) == pre

    column_values = {
        int(j): float(pre[j])
        for j in np.flatnonzero(~blocked & stable)
        if int(j) not in pruned_cols
    }
    return CoreConstantFacts(
        dead_rows=dead_rows, fold_rows=fold_rows, column_values=column_values
    )


def _pre_activation(contributors, matrix, bias, theta: float, dtype) -> np.ndarray:
    """``(sum_r c_r * W[r, :] + b) / theta`` evaluated entirely in ``dtype``."""
    pre = contributors.astype(dtype) @ np.asarray(matrix, dtype=dtype)
    if bias is not None and np.asarray(bias).size == matrix.shape[1]:
        pre = pre + np.asarray(bias, dtype=dtype).reshape(matrix.shape[1])
    if theta != 1.0:
        pre = pre / dtype(theta)
    return pre
