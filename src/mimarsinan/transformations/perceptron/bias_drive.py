"""The drive of a perceptron's always-on bias row: the input value its axon carries on the chip.

A parameter-encoded bias rides an always-on axon row of the shared weight grid, so the row's weight is the bias over
the row's input value. On a normalized currency (the spiking wires, whose rates span [0, 1]) that input is the
currency's full scale, 1. On the integer value path the network keeps its own units (activation and input scales
1.0) and every core reads its input on a DAC grid; there the row carries the largest code value of the grid its reader
reads, its full scale ``x_fs`` (``mapping.support.bias_drive``, stamped at Weight Quantization entry), so the shared
grid sizes ``max(max|w|, max|b| / x_fs)``, the row's code is ``round(b s / x_fs)`` and the projected bias is that code
times ``x_fs / s``. The value 1.0 is the default: a perceptron without a stamp keeps its projection byte for byte.
"""

from __future__ import annotations

from typing import Any, TypeVar

_Bias = TypeVar("_Bias")

BIAS_ROW_DRIVE = "bias_row_drive"
"""The perceptron (and IR core) attribute holding the always-on row's drive, in the network's value units."""


def bias_row_drive(holder) -> float:
    """The always-on row's drive of a perceptron or an IR core: its stamp, else 1.0 (the normalized full scale)."""
    return float(getattr(holder, "__dict__", {}).get(BIAS_ROW_DRIVE, 1.0))


def set_bias_row_drive(holder, drive: float) -> None:
    """Stamp ``drive`` on a perceptron or an IR core (a plain attribute: it travels in deep copies and pickles and
    leaves the state_dict unchanged); a drive of 1.0 removes the stamp."""
    drive = float(drive)
    if not drive > 0.0:
        raise ValueError(f"an always-on row's drive must be positive, got {drive!r}")
    if drive == 1.0:
        holder.__dict__.pop(BIAS_ROW_DRIVE, None)
    else:
        holder.__dict__[BIAS_ROW_DRIVE] = drive


def bias_row_values(perceptron, effective_bias: _Bias) -> _Bias:
    """The always-on row's weight for ``effective_bias``: the bias over the row's drive (the bias itself at 1.0, or
    without a bias)."""
    drive = bias_row_drive(perceptron)
    if drive == 1.0 or effective_bias is None:
        return effective_bias
    value: Any = effective_bias
    return value / drive
