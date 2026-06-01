"""Single TTFS-cycle step: matmul contribution + fire-once-latch.

Counterpart to ``lif_core_step.lif_core_contribute_and_fire`` for the genuine
binary-spike, cycle-based TTFS mode. Inputs are latched binary spikes; the
membrane integrates them each cycle (bias is a per-cycle current, matching LIF),
and the neuron fires at most once, latching its output.
"""

from __future__ import annotations

import torch

from mimarsinan.models.nn.ttfs_cycle_kernels import ttfs_cycle_fire_and_latch


def ttfs_cycle_contribute_and_fire(
    memb: torch.Tensor,
    weight: torch.Tensor,
    inp: torch.Tensor,
    threshold: torch.Tensor,
    has_fired: torch.Tensor,
    *,
    hw_bias: torch.Tensor | None,
    thresholding_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Integrate ``inp`` into ``memb``, then fire-once-latch; return latched spikes."""
    contribution = torch.matmul(weight, inp.T).T
    if hw_bias is not None:
        contribution = contribution + hw_bias
    memb += contribution
    return ttfs_cycle_fire_and_latch(
        memb,
        threshold,
        has_fired,
        thresholding_mode=thresholding_mode,
        output_dtype=output_dtype,
    )
