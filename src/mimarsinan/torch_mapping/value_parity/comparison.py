"""The probe's value metric: max|Δ| / max(1, max|ref|) in fp64."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ValueDelta:
    max_abs_error: float
    reference_max: float
    relative_error: float


def _agreeing_non_finite(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Positions where both are NaN, or both the same infinity: equal values, so no error."""
    same_infinity = torch.isinf(got) & torch.isinf(want) & (got == want)
    return (torch.isnan(got) & torch.isnan(want)) | same_infinity


def relative_error(got: torch.Tensor, want: torch.Tensor) -> ValueDelta:
    """E270's metric on equal shapes (a shape difference is refused, never broadcast).

    Matching NaN or infinity agrees; a non-finite value on one side only is an infinite error.
    """
    if tuple(got.shape) != tuple(want.shape):
        raise ValueError(
            f"output shape {tuple(got.shape)} differs from the reference's "
            f"{tuple(want.shape)}"
        )
    if want.numel() == 0:
        return ValueDelta(0.0, 0.0, 0.0)
    got, reference = got.detach().double(), want.detach().double()
    agree = _agreeing_non_finite(got, reference)
    diff = torch.where(agree, torch.zeros_like(reference), (got - reference).abs())
    diff = torch.where(torch.isnan(diff), torch.full_like(diff, math.inf), diff)
    finite = reference[torch.isfinite(reference)]
    reference_max = float(finite.abs().max()) if finite.numel() else 0.0
    max_abs = float(diff.max())
    return ValueDelta(max_abs, reference_max, max_abs / max(1.0, reference_max))


def degenerate_kind(want: torch.Tensor) -> Optional[str]:
    """How a reference with no finite non-zero entry carries no information (its finite values decide); else None."""
    finite = want[torch.isfinite(want)]
    if finite.numel() and bool(finite.ne(0).any()):
        return None
    if finite.numel() == want.numel():
        return "zero"
    return "non-finite" if finite.numel() == 0 else "zero or non-finite"


def constant_across_samples(want: torch.Tensor, bound: float = 0.0) -> bool:
    """Every sample's reference output is the first one's within ``bound`` under the probe's metric (NaN equal to
    NaN): the inputs carry no information. ``bound=0`` is exact equality."""
    if want.dim() == 0 or want.shape[0] < 2:
        return False
    return relative_error(want[:1].expand_as(want), want).relative_error <= bound


def uninformative_kind(want: torch.Tensor, bound: float = 0.0) -> Optional[str]:
    """How a reference carries no information on its inputs: no finite non-zero entry, or constant; else None."""
    return degenerate_kind(want) or ("constant" if constant_across_samples(want, bound) else None)
