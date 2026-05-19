"""Symmetric integer quantization bounds from bit width."""

from __future__ import annotations


def quantization_bounds(bits: int) -> tuple[int, int]:
    """Return ``(q_min, q_max)`` for ``bits``-bit signed symmetric quantization."""
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    q_max = (2 ** (bits - 1)) - 1
    q_min = -(2 ** (bits - 1))
    return q_min, q_max
