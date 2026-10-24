"""Symmetric integer quantization bounds from bit width, and the width a layer or a core declares (WI-5g)."""

from __future__ import annotations

from typing import Any

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
