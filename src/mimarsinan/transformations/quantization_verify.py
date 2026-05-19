"""Shared integer-quantization checks for perceptron and IR paths."""

from __future__ import annotations

import numpy as np


def assert_integer_scaled_matrix(
    mat,
    scale: float,
    q_min: int,
    q_max: int,
    *,
    name: str = "matrix",
    scale_tol: float = 1e-6,
) -> list[str]:
    """Return a list of failure messages (empty when OK)."""
    failures: list[str] = []
    if scale == 0.0:
        failures.append(f"{name}: parameter_scale is 0")
        return failures
    if abs(scale - 1.0) > scale_tol:
        failures.append(f"{name}: parameter_scale={scale} (expected 1.0 after chip quantization)")

    dtype = getattr(mat, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.integer):
        maxv = int(mat.max()) if mat.size else 0
        minv = int(mat.min()) if mat.size else 0
        if maxv > q_max:
            failures.append(f"{name}: max(W)={maxv} > q_max={q_max}")
        if minv < q_min:
            failures.append(f"{name}: min(W)={minv} < q_min={q_min}")
        return failures

    scaled = np.asarray(mat, dtype=np.float64) * float(scale)
    if not np.allclose(scaled, np.round(scaled)):
        failures.append(f"{name}: W*scale is not integer-quantized")
    rounded = np.round(scaled)
    if rounded.max() > q_max or rounded.min() < q_min:
        failures.append(
            f"{name}: quantized range [{rounded.min()}, {rounded.max()}] "
            f"outside [{q_min}, {q_max}]"
        )
    return failures
