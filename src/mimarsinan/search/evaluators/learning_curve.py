"""Parametric learning-curve models and extrapolation for NAS accuracy evaluation."""

from __future__ import annotations

import math
import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit


def exp3(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a - b * np.exp(-c * t)


def pow3(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a - b * np.power(t + 1.0, -c)


def log2(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log1p(b * t)


def soft_clip_accuracy(raw: float, observed_max: float) -> float:
    """Soft-clip an extrapolated accuracy value."""
    if raw <= 0.0:
        return 0.0
    ceiling = min(1.0, observed_max + 0.15)
    if raw <= observed_max:
        return raw
    if raw >= ceiling:
        return ceiling
    margin = ceiling - observed_max
    if margin <= 0:
        return observed_max
    excess = (raw - observed_max) / margin
    compressed = 1.0 - math.exp(-2.0 * excess)
    return observed_max + margin * compressed


_CURVE_MODELS = [
    ("exp3", exp3, 3, [0.0, 0.0, 1e-6], [1.0, 1.0, 50.0]),
    ("pow3", pow3, 3, [0.0, 0.0, 1e-6], [1.0, 1.0, 10.0]),
    ("log2", log2, 2, [0.0, 1e-6], [1.0, 100.0]),
]


def fit_and_extrapolate(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_target: float,
) -> Tuple[float, str]:
    """Try each curve model, pick the best fit, and extrapolate to t_target."""
    best_residual = float("inf")
    best_pred = float(y_obs[-1])
    best_name = "fallback"
    observed_max = float(max(y_obs))

    for name, func, n_params, lb, ub in _CURVE_MODELS:
        if len(t_obs) < n_params + 1:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    func,
                    t_obs,
                    y_obs,
                    bounds=(lb, ub),
                    maxfev=5000,
                    p0=None,
                )
            y_fit = func(t_obs, *popt)
            residual = float(np.mean((y_fit - y_obs) ** 2))

            if residual < best_residual:
                best_residual = residual
                pred = float(func(np.array([t_target]), *popt)[0])
                pred = soft_clip_accuracy(pred, observed_max)
                best_pred = pred
                best_name = name
        except (RuntimeError, ValueError, TypeError):
            continue

    return best_pred, best_name
