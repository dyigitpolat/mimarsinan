"""Greedy cascaded TTFS fire-once step (single-spike, hardware-faithful).

A genuine TTFS neuron fires **exactly once** — the information is in *when* it
fires, not in a held output. So this kernel emits a spike only on the firing
cycle (the ``has_fired`` 0→1 transition) and is silent forever after; only that
single spike travels on the wire.

The integration that the single spike feeds is a **ramp**, reconstructed at the
*consumer* side (see ``ttfs_cycle_step`` / the per-cycle executor): once a single
spike arrives on an axon at ``t_j``, that axon contributes its weight every
subsequent cycle, so ``membrane(t) = Σ_j w_j·(t − t_j)``. Mirrored by nevresim's
single-spike ``SpikingCompute`` neuron and the SANA-FE single-spike soma +
ramping dendrite.
"""

from __future__ import annotations

import torch

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


def ttfs_cycle_fire_once(
    memb: torch.Tensor,
    threshold: torch.Tensor,
    has_fired: torch.Tensor,
    *,
    thresholding_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Emit a single spike on the cycle ``memb`` first crosses ``threshold``.

    ``has_fired`` (bool, same shape as ``memb``) is the persistent per-neuron
    latch updated in place. The returned tensor is the **single-spike** output:
    ``1`` only on the firing cycle, ``0`` before and after. No membrane reset —
    later input is greedily ignored.
    """
    crossed = _THRESHOLD_OPS[thresholding_mode](threshold, memb)
    newly_fired = crossed & (~has_fired)
    has_fired |= crossed
    if output_dtype is not None:
        return newly_fired.to(output_dtype)
    return newly_fired.float()
