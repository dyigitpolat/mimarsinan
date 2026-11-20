"""Symmetric integer quantization bounds from bit width, and the width a layer or a core declares (WI-5g)."""

from __future__ import annotations

from typing import Any

import numpy as np

#: The weight width a perceptron's core type declares (``layer_core_types``, WI-5g), stamped on the perceptron.
LAYER_WEIGHT_BITS = "declared_weight_bits"
#: The weight width of the core type an IR core, weight bank, soft core or hard core binds to (WI-5g).
CORE_WEIGHT_BITS = "weight_bits"


def quantization_bounds(bits: int) -> tuple[int, int]:
    """Return ``(q_min, q_max)`` for ``bits``-bit signed symmetric quantization."""
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    q_max = (2 ** (bits - 1)) - 1
    q_min = -(2 ** (bits - 1))
    return q_min, q_max


def layer_weight_bits(perceptron: Any, default: int) -> int:
    """The weight width of ``perceptron``'s grid: its core type's declared width, else the platform's ``default``."""
    bits = getattr(perceptron, LAYER_WEIGHT_BITS, None)
    return int(default) if bits is None else int(bits)


def core_weight_bits(holder: Any, default: int) -> int:
    """The weight width of an IR core or weight bank: its core type's, else the platform's ``default``."""
    bits = getattr(holder, CORE_WEIGHT_BITS, None)
    return int(default) if bits is None else int(bits)


def _bounds_of(holder, bits: int, q_min: int, q_max: int) -> tuple[int, int]:
    """The grid bounds of an IR core or bank: its core type's width (WI-5g), else the program's ``bits``."""
    width = core_weight_bits(holder, bits)
    return (q_min, q_max) if width == int(bits) else quantization_bounds(width)


def _dtype_of(holder, bits: int, q_dtype):
    """The code dtype of an IR core or bank: the program's ``q_dtype``, or its core type's own width's, which holds
    every code of that width (int32 above 16 bits, so a wide type's codes never wrap)."""
    width = core_weight_bits(holder, bits)
    return q_dtype if width == int(bits) else code_dtype(width)


def code_dtype(bits: int):
    """The integer dtype holding every code of a ``bits``-wide weight grid: int8, int16, and int32 above 16 bits
    (the registry admits widths up to 32), so no code wraps."""
    return np.int8 if bits <= 8 else np.int16 if bits <= 16 else np.int32
