"""Single source of truth for spike-train construction (PyTorch + hybrid)."""

from __future__ import annotations

import logging

import torch

from mimarsinan.chip_simulation import spike_modes
from mimarsinan.models.nn.activations import LIFActivation

logger = logging.getLogger("mimarsinan.spiking.spike_trains")


def uniform_spike_train(rate: torch.Tensor, T: int) -> torch.Tensor:
    """Encode rates in [0, 1] to a uniform-spaced spike train of shape ``(T, ...)``.

    For raw pipeline input only (``run_cycle_accurate`` outer loop). Delegates
    per-cycle semantics to :func:`spike_modes.to_uniform_spikes`.
    """
    T = int(T)
    rate_c = rate.clamp(0.0, 1.0)
    trains = [
        spike_modes.to_uniform_spikes(rate_c, cycle, simulation_length=T)
        for cycle in range(T)
    ]
    return torch.stack(trains, dim=0)


def lif_spike_train(
    pre_activation: torch.Tensor,
    lif: LIFActivation,
    T: int | None = None,
) -> torch.Tensor:
    """Build ``(T, B, ...)`` spike train via ``T`` single-step IF integrations.

    Matches cycle-accurate semantics (membrane evolves across cycles with one
    reset at the start). Used by :meth:`LIFActivation.forward_spiking` when
    ``use_cycle_accurate_trains`` is set.
    """
    from spikingjelly.activation_based import functional

    T = int(T if T is not None else lif.T)
    was_ca = lif._cycle_accurate_mode
    lif.set_cycle_accurate(True)
    functional.reset_net(lif.if_node)
    try:
        scale = lif.activation_scale
        if isinstance(scale, torch.Tensor):
            safe_scale = scale.to(
                device=pre_activation.device, dtype=pre_activation.dtype,
            ).clamp(min=1e-12)
        else:
            safe_scale = max(float(scale), 1e-12)

        # Signed integration (no relu): membrane may go negative, matching the chip.
        x_norm = pre_activation / safe_scale
        spikes = [lif.if_node(x_norm) for _ in range(T)]
        return torch.stack(spikes, dim=0)
    finally:
        lif.set_cycle_accurate(was_ca)


def materialized_spike_train(rate: torch.Tensor, T: int) -> torch.Tensor:
    """Build a full ``(T, ...)`` train upfront for ``spike_mode='SpikeTrain'``.

    When segment inputs are still expressed as rates, materialization uses the
    same Uniform spacing as the per-cycle encoder so replay matches HCM.
    """
    return uniform_spike_train(rate, T)


def rates_to_spike_train(
    rates: torch.Tensor,
    T: int,
    *,
    spike_mode: str,
    log_fallback: bool = True,
) -> torch.Tensor:
    """Legacy fallback: expand clamped rates to a spike train per cycle.

    Not used on the encoding-ComputeOp path when cycle-accurate LIF is enabled.
    """
    if spike_mode == "SpikeTrain":
        if log_fallback:
            logger.debug(
                "rates_to_spike_train SpikeTrain materialization (T=%d, shape=%s)",
                T, tuple(rates.shape),
            )
        return materialized_spike_train(rates.clamp(0.0, 1.0), int(T))

    if log_fallback:
        logger.debug(
            "rates_to_spike_train fallback (spike_mode=%r, T=%d, shape=%s)",
            spike_mode, T, tuple(rates.shape),
        )
    T = int(T)
    rate_c = rates.clamp(0.0, 1.0)
    trains = [
        spike_modes.to_spikes(
            rate_c, cycle, simulation_length=T, spike_mode=spike_mode,
        )
        for cycle in range(T)
    ]
    return torch.stack(trains, dim=0)
