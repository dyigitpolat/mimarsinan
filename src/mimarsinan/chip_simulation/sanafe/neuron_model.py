"""SANA-FE soma attribute mapping for ``SubtractiveLIFReset`` parity."""

from __future__ import annotations

from typing import Optional

from mimarsinan.chip_simulation.firing_strategy import FiringStrategyFactory


def soma_hw_name_for_spiking_mode(spiking_mode: str, schedule=None) -> str:
    """SANA-FE soma plugin name for ``(spiking_mode, schedule)`` (via the policy)."""
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

    return policy_for_spiking_mode(spiking_mode, schedule).soma_hw_name()


def lif_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
    active_start: Optional[int] = None,
    active_length: Optional[int] = None,
    firing_mode: str = "Default",
) -> dict:
    """``model_attributes`` reproducing ``SubtractiveLIFReset`` (no leak, subtractive reset).

    ``active_start``/``active_length`` gate the soma to ``[core.latency, T+core.latency)``
    (the window HCM records into); ``None`` keeps it active the whole simulation.
    """
    reset_mode = FiringStrategyFactory.from_config(
        {
            "firing_mode": firing_mode,
            "thresholding_mode": "<=",
            "spiking_mode": "lif",
        }
    ).sanafe_reset_mode()
    attrs: dict = {
        "threshold": float(threshold),
        "reset_mode": reset_mode,
    }
    if hardware_bias is not None:
        attrs["bias"] = float(hardware_bias)
    if active_start is not None:
        attrs["active_start"] = int(active_start)
    if active_length is not None:
        attrs["active_length"] = int(active_length)
    return attrs


def _ttfs_base_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
    active_start: Optional[int] = None,
    active_length: Optional[int] = None,
    preset_membrane: Optional[float] = None,
) -> dict:
    attrs: dict = {"threshold": float(threshold)}
    if hardware_bias is not None:
        attrs["bias"] = float(hardware_bias)
    if active_start is not None:
        attrs["active_start"] = int(active_start)
    if active_length is not None:
        attrs["active_length"] = int(active_length)
    if preset_membrane is not None:
        attrs["preset_membrane"] = float(preset_membrane)
    return attrs


def ttfs_continuous_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
    active_start: Optional[int] = None,
    active_length: Optional[int] = None,
    preset_membrane: Optional[float] = None,
) -> dict:
    return _ttfs_base_model_attributes(
        threshold=threshold,
        hardware_bias=hardware_bias,
        active_start=active_start,
        active_length=active_length,
        preset_membrane=preset_membrane,
    )


def ttfs_quantized_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
    active_start: Optional[int] = None,
    active_length: Optional[int] = None,
    preset_membrane: Optional[float] = None,
) -> dict:
    return _ttfs_base_model_attributes(
        threshold=threshold,
        hardware_bias=hardware_bias,
        active_start=active_start,
        active_length=active_length,
        preset_membrane=preset_membrane,
    )


def ttfs_cycle_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
    active_start: Optional[int] = None,
    active_length: Optional[int] = None,
) -> dict:
    """Genuine single-spike soma attrs — no ``preset_membrane`` (reconstructs V)."""
    return _ttfs_base_model_attributes(
        threshold=threshold,
        hardware_bias=hardware_bias,
        active_start=active_start,
        active_length=active_length,
    )


def ttfs_cascade_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
    active_start: Optional[int] = None,
    active_length: Optional[int] = None,
) -> dict:
    """Cascaded fire-once-latch soma attrs — no ``preset_membrane`` (genuine
    integration of latched spikes; ungated when ``active_length`` is None)."""
    return _ttfs_base_model_attributes(
        threshold=threshold,
        hardware_bias=hardware_bias,
        active_start=active_start,
        active_length=active_length,
    )


def input_neuron_attributes(spike_times: Optional[list[int]] = None) -> dict:
    """``model_attributes`` for a neuron backed by the ``inputs[N]`` soma.

    SANA-FE 2.1.1's ``input`` soma reads ``model_attributes["spikes"]``
    as a list of 1-indexed timesteps at which the neuron fires.
    """
    return {"spikes": list(spike_times) if spike_times is not None else []}
