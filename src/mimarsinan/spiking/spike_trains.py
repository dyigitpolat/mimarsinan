"""Single source of truth for spike-train construction (PyTorch + hybrid)."""

from __future__ import annotations

import logging

import torch

from mimarsinan.chip_simulation import spike_modes
from mimarsinan.models.nn.activations import LIFActivation

logger = logging.getLogger("mimarsinan.spiking.spike_trains")


def uniform_comb_operands(
    rate: torch.Tensor, T: int, *, phase_dither: bool = False,
    channels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(clamped rate, comb rotation or None)``: the operands every cycle of a uniform encode reads (U-SC3: the one
    construction ``uniform_spike_train`` and the per-cycle ``cycle_trains.EncodedTrain`` share)."""
    rate_c = rate.clamp(0.0, 1.0)
    offsets = None
    if phase_dither:
        offsets = (
            spike_modes.uniform_phase_offsets(rate_c.shape[-1], int(T), device=rate_c.device)
            if channels is None else
            spike_modes.channel_phase_offsets(channels, int(T), device=rate_c.device)
        )
    return rate_c, offsets


def uniform_comb_cycle(rate_c: torch.Tensor, cycle: int, T: int, offsets: torch.Tensor | None) -> torch.Tensor:
    """Cycle ``cycle`` of the uniform encode of ``uniform_comb_operands``' clamped rate: a 0/1 mask, elementwise."""
    return spike_modes.to_uniform_spikes(rate_c, cycle, int(T), phase_offsets=offsets)


def uniform_spike_train(
    rate: torch.Tensor, T: int, *, phase_dither: bool = False,
    channels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode rates in [0, 1] to a uniform-spaced spike train of shape ``(T, ...)``.

    ``phase_dither`` rotates each channel's comb by a deterministic offset mod T
    (count-exact) to decorrelate cross-channel arrival [calculus sec.15.11],
    keyed on ``channels`` (one channel id per last-dim position: a chip stage's
    channels at unbounded geometry, E1; or one per element of a sample, broadcast
    over the batch: the NF twin's keys, ``phase_channels.element_channels``), else
    on the last-dim index."""
    T = int(T)
    rate_c, offsets = uniform_comb_operands(rate, T, phase_dither=phase_dither, channels=channels)
    trains = [uniform_comb_cycle(rate_c, cycle, T, offsets) for cycle in range(T)]
    return torch.stack(trains, dim=0)


def lif_spike_train(
    pre_activation: torch.Tensor,
    lif: LIFActivation,
    T: int | None = None,
) -> torch.Tensor:
    """Build ``(T, B, ...)`` spike train via ``T`` single-step IF integrations.

    Cycle-accurate: the membrane evolves across cycles with one reset at the start.
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


def materialized_spike_train(
    rate: torch.Tensor, T: int, *, phase_dither: bool = False,
    channels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a full ``(T, ...)`` train upfront for ``spike_mode='SpikeTrain'``.

    Uses the same Uniform spacing as the per-cycle encoder so replay matches HCM.
    """
    return uniform_spike_train(rate, T, phase_dither=phase_dither, channels=channels)


def rates_to_spike_train(
    rates: torch.Tensor,
    T: int,
    *,
    spike_mode: str,
    log_fallback: bool = True,
    phase_dither: bool = False,
    channels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Legacy fallback: expand clamped rates to a spike train per cycle (a dithered
    Uniform or SpikeTrain encode keyed on ``channels``, ``uniform_spike_train``)."""
    if spike_mode == "SpikeTrain":
        if log_fallback:
            logger.debug(
                "rates_to_spike_train SpikeTrain materialization (T=%d, shape=%s)",
                T, tuple(rates.shape),
            )
        return materialized_spike_train(
            rates.clamp(0.0, 1.0), int(T), phase_dither=phase_dither, channels=channels,
        )
    if spike_mode == "Uniform" and phase_dither:
        return uniform_spike_train(rates, int(T), phase_dither=True, channels=channels)

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
