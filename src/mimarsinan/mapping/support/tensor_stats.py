"""Tensor statistics that survive deployment-scale inputs."""

from __future__ import annotations

import math

import torch

# torch.quantile rejects inputs above 2^24 elements ("input tensor is too
# large"). A ViT patch-embed input at batch 512 is 77M elements, so any
# calibration that quantiles a live activation must guard the cap.
QUANTILE_ELEMENT_LIMIT = 1 << 24
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")


def _order_statistic(flat: torch.Tensor, index: int) -> torch.Tensor:
    """The ``index``-th smallest entry (0-based), by selection: no sort, no size cap."""
    return torch.kthvalue(flat, index + 1).values


def _exact_quantile(flat: torch.Tensor, q: float, interpolation: str) -> torch.Tensor:
    """``torch.quantile``'s definition read from the order statistics: rank ``q (n - 1)``, its two neighbours and
    torch's interpolation between them (``nearest`` rounds the rank half to even, as torch does)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile() q must be in the range [0, 1] but got {q}")
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"quantile() interpolation must be one of {list(_INTERPOLATIONS)}, got {interpolation!r}")
    rank = q * (flat.numel() - 1)
    below, above = math.floor(rank), math.ceil(rank)
    if interpolation == "lower":
        return _order_statistic(flat, below)
    if interpolation == "higher":
        return _order_statistic(flat, above)
    if interpolation == "nearest":
        return _order_statistic(flat, round(rank))
    low = _order_statistic(flat, below)
    high = low if above == below else _order_statistic(flat, above)
    weight = 0.5 if interpolation == "midpoint" else rank - below
    return torch.lerp(low, high, torch.tensor(weight, dtype=flat.dtype, device=flat.device))


def safe_quantile(
    tensor: torch.Tensor, q: float, limit: int = QUANTILE_ELEMENT_LIMIT,
    interpolation: str = "linear",
) -> torch.Tensor:
    """``torch.quantile`` that tolerates oversized inputs, exact at every size.

    At or below ``limit`` elements it is ``torch.quantile`` itself. Above it (where torch refuses the input) the same
    quantile is read from its order statistics by selection (``torch.kthvalue``), with torch's interpolation rule.
    [U-BRANGE F-3] No subsample is read: a strided subsample of a flattened (batch, token, channel) tensor whose
    channel count is even read even channels only (W38 group 6's diagnosis, m-2).
    """
    flat = tensor.detach().reshape(-1)
    if not flat.is_floating_point():
        flat = flat.float()
    if flat.numel() <= limit:
        return torch.quantile(flat, q, interpolation=interpolation)
    return _exact_quantile(flat, float(q), interpolation)
