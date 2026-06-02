"""Single greedy-cascaded TTFS step: ramp integration + single-spike fire.

Counterpart to ``lif_core_step.lif_core_contribute_and_fire`` for the cascaded
(pipelined) TTFS schedule. Inputs are **single** spikes (one per axon, at its
arrival cycle). Each step accumulates the weighted arrivals into a persistent
``ramp_current`` (the consumer-side latch) and adds that ramp — plus the bias
ramp — to the membrane, so ``membrane(t) = Σ_j w_j·(t − t_j) + b·t``. The neuron
fires **at most once**, emitting a single spike on its firing cycle.
"""

from __future__ import annotations

import torch

from mimarsinan.models.nn.ttfs_cycle_kernels import ttfs_cycle_fire_once


def ttfs_cycle_contribute_and_fire(
    memb: torch.Tensor,
    ramp_current: torch.Tensor,
    weight: torch.Tensor,
    inp: torch.Tensor,
    threshold: torch.Tensor,
    has_fired: torch.Tensor,
    *,
    hw_bias: torch.Tensor | None,
    thresholding_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Accumulate single-spike arrivals into the ramp, integrate, fire once.

    ``ramp_current`` (same shape as ``memb``) is the persistent per-neuron sum of
    the weights of all axons that have spiked so far — updated in place. Returns
    the single-spike output (high only on the firing cycle).
    """
    arrivals = torch.matmul(weight, inp.T).T
    ramp_current += arrivals
    memb += ramp_current
    if hw_bias is not None:
        memb += hw_bias
    return ttfs_cycle_fire_once(
        memb, threshold, has_fired,
        thresholding_mode=thresholding_mode, output_dtype=output_dtype,
    )
