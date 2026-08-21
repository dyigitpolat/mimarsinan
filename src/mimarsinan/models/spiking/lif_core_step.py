"""Single LIF cycle: matmul contribution + threshold fire-and-reset."""

from __future__ import annotations

import torch

from mimarsinan.models.nn.lif_kernels import (
    lif_fire_and_reset,
    snap_membrane_to_lattice,
)


def lif_core_advance(
    memb: torch.Tensor,
    contribution: torch.Tensor,
    threshold: torch.Tensor,
    *,
    thresholding_mode: str,
    firing_mode: str,
    output_dtype: torch.dtype | None = None,
    lattice_scale: float | None = None,
    membrane_bounds: tuple[float, float] | None = None,
) -> torch.Tensor:
    """The elementwise LIF cycle: integrate a precomputed contribution, fire,
    reset. Shape-agnostic — the physics shared by every charge layout.
    ``lattice_scale`` projects the membrane onto the exact chip lattice
    before the compare (integer-chip cells; ties must fire on both twins).
    ``membrane_bounds`` is the declared register interval a saturating
    unsigned accumulator clamps to on every update; ``None`` (the default
    point) leaves today's unbounded arithmetic bit-identical."""
    memb += contribution
    if membrane_bounds is not None:
        memb.clamp_(membrane_bounds[0], membrane_bounds[1])
    if lattice_scale is not None:
        snap_membrane_to_lattice(memb, lattice_scale)
    return lif_fire_and_reset(
        memb,
        threshold,
        thresholding_mode=thresholding_mode,
        firing_mode=firing_mode,
        output_dtype=output_dtype,
    )


def lif_core_contribute_and_fire(
    memb: torch.Tensor,
    weight: torch.Tensor,
    inp: torch.Tensor,
    threshold: torch.Tensor,
    *,
    hw_bias: torch.Tensor | None,
    thresholding_mode: str,
    firing_mode: str,
    output_dtype: torch.dtype | None = None,
    lattice_scale: float | None = None,
    membrane_bounds: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Integrate inputs into ``memb``, return spike tensor for this cycle."""
    contribution = torch.matmul(weight, inp.T).T
    if hw_bias is not None:
        contribution = contribution + hw_bias
    return lif_core_advance(
        memb,
        contribution,
        threshold,
        thresholding_mode=thresholding_mode,
        firing_mode=firing_mode,
        output_dtype=output_dtype,
        lattice_scale=lattice_scale,
        membrane_bounds=membrane_bounds,
    )
