"""SANA-FE soma attribute mapping for ``SubtractiveLIFReset`` parity.

SANA-FE 2.1.1's ``leaky_integrate_fire`` soma model exposes the knobs we
need to reproduce mimarsinan's ``SubtractiveLIFReset`` semantics:

* ``leak_decay: 1.0``          — no decay (the soma keeps voltage
                                  across timesteps; equivalent to dv=0).
* ``reset_mode: "soft"``       — subtractive reset by the threshold.
* ``reset: 0.0``               — reset target voltage (unused for soft
                                  reset but required to be present).
* ``threshold: <float>``       — firing threshold.
* ``bias: <float>``            — optional per-neuron bias added each
                                  timestep.

Active-window gating is handled host-side at the input layer (only
inject spikes during the active window); with ``leak_decay=1.0`` the
soma voltage is preserved through zero-input cycles, so soma- and
input-side gating are equivalent.

Why no ``needs_plugin()``: SANA-FE 2.1.1's built-in soma covers
everything Strategy A targeted in the plan.  The custom-plugin escape
hatch is documented in the sub-package ARCHITECTURE.md as the path to
take if a future SANA-FE release drops one of these knobs.
"""

from __future__ import annotations

from typing import Optional

from mimarsinan.chip_simulation.firing_strategy import FiringStrategyFactory

from .presets import SOMA_LIF_NAME, SOMA_TTFS_CONTINUOUS_NAME, SOMA_TTFS_QUANTIZED_NAME


def soma_hw_name_for_spiking_mode(spiking_mode: str) -> str:
    if spiking_mode == "ttfs":
        return SOMA_TTFS_CONTINUOUS_NAME
    if spiking_mode == "ttfs_quantized":
        return SOMA_TTFS_QUANTIZED_NAME
    return SOMA_LIF_NAME


def lif_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
    active_start: Optional[int] = None,
    active_length: Optional[int] = None,
    firing_mode: str = "Default",
) -> dict:
    """Build the ``model_attributes`` dict for a regular LIF neuron.

    The returned dict reproduces ``SubtractiveLIFReset`` exactly: no
    leak, subtractive reset, configurable strict-` <` firing comparator
    (SANA-FE's ``leaky_integrate_fire`` uses strict ` <` by default).

    ``active_start`` / ``active_length`` are optional per-core gating
    cycles (0-based).  When the runner pads simulation length to
    ``T + max_latency`` to flush multi-depth cascades, each core's
    soma must only integrate during ``[core.latency, T + core.latency)``
    — exactly the window HCM's ``_run_neural_segment_rate`` accumulates
    into ``record_in_t`` / ``record_out_t``.  Leaving them unset
    (``None``) keeps the neuron active for the entire simulation,
    preserving the pre-window default for tests built before the gate
    existed.
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


def input_neuron_attributes(spike_times: Optional[list[int]] = None) -> dict:
    """``model_attributes`` for a neuron backed by the ``inputs[N]`` soma.

    SANA-FE 2.1.1's ``input`` soma reads ``model_attributes["spikes"]``
    as a list of 1-indexed timesteps at which the neuron fires.
    """
    return {"spikes": list(spike_times) if spike_times is not None else []}
