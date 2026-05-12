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


def lif_model_attributes(
    *,
    threshold: float,
    hardware_bias: Optional[float] = None,
) -> dict:
    """Build the ``model_attributes`` dict for a regular LIF neuron.

    The returned dict reproduces ``SubtractiveLIFReset`` exactly: no
    leak, subtractive reset, configurable strict-` <` firing comparator
    (SANA-FE's ``leaky_integrate_fire`` uses strict ` <` by default).
    """
    attrs: dict = {
        "threshold": float(threshold),
        "leak_decay": 1.0,
        "reset_mode": "soft",
        "reset": 0.0,
        # SANA-FE's leaky_integrate_fire enforces "must update every
        # timestep" — without this flag, a cycle without input throws
        # at runtime.  We always tick our neurons every cycle (matching
        # the HCM contract), so this is the safe default.
        "force_update": True,
    }
    if hardware_bias is not None:
        attrs["bias"] = float(hardware_bias)
    return attrs


def input_neuron_attributes(spike_times: Optional[list[int]] = None) -> dict:
    """``model_attributes`` for a neuron backed by the ``inputs[N]`` soma.

    SANA-FE 2.1.1's ``input`` soma reads ``model_attributes["spikes"]``
    as a list of 1-indexed timesteps at which the neuron fires.
    """
    return {"spikes": list(spike_times) if spike_times is not None else []}
