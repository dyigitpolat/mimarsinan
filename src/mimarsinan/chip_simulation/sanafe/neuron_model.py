"""SANA-FE soma attribute mapping for ``SubtractiveLIFReset`` parity.

Two strategies, in order of preference:

* **A — built-in soma + per-neuron attributes** (default).  We pass
  ``leak_decay=0``, ``input_decay=0``, ``reset_mode="subtract"``,
  ``threshold_mode="strict"|"inclusive"`` to SANA-FE's bundled
  ``loihi_lif_soma``.  Active-window gating is achieved by injecting
  input spikes only during the active window at the host side (with
  ``leak_decay=0`` the soma voltage is preserved through zero-input
  cycles, so the soma cannot tell the difference).

* **B — custom plugin** (loaded on demand).  When the built-in soma
  lacks a required knob (discovered by the single-core parity test),
  ``plugins/mimarsinan_subtractive_lif.cpp`` is built and pointed to
  via ``load_soma_plugin``.  ``needs_plugin()`` probes the installed
  SANA-FE and returns ``True`` iff Strategy B must be used.

The runner is plugin-agnostic: it asks this module for soma attributes
and a plugin path.
"""

from __future__ import annotations

import os
from typing import Optional


THRESHOLDING_MODE_TO_SANAFE = {
    "<":  "strict",
    "<=": "inclusive",
}


def soma_attributes(
    *,
    threshold: float,
    thresholding_mode: str,
    hardware_bias: Optional[float],
    reset_interval: int,
    reset_offset: int = 0,
    active_start: int = 0,
    active_length: Optional[int] = None,
) -> dict:
    """Build the per-neuron ``model_attributes`` dict for SANA-FE's soma.

    Mirrors the ``SubtractiveLIFReset`` contract from
    ``mimarsinan.chip_simulation.subtractive_lif``: no leak, no synaptic
    persistence, subtractive reset, configurable strict/inclusive
    threshold.
    """
    if thresholding_mode not in THRESHOLDING_MODE_TO_SANAFE:
        raise ValueError(
            f"unsupported thresholding_mode {thresholding_mode!r}; "
            f"expected one of {sorted(THRESHOLDING_MODE_TO_SANAFE.keys())}"
        )
    attrs: dict = {
        "threshold": float(threshold),
        "leak_decay": 0,
        "input_decay": 0,
        "reset_mode": "subtract",
        "reset_voltage": 0.0,
        "threshold_mode": THRESHOLDING_MODE_TO_SANAFE[thresholding_mode],
        "reset_interval": int(reset_interval),
        "reset_offset": int(reset_offset),
        "active_start": int(active_start),
    }
    if active_length is not None:
        attrs["active_length"] = int(active_length)
    if hardware_bias is not None:
        attrs["bias"] = float(hardware_bias)
    return attrs


def needs_plugin() -> bool:  # pragma: no cover — answered by integration test
    """Probe SANA-FE for the knobs we need; ``True`` ⇒ load Strategy B.

    Currently returns ``False``: the single-core parity test will flip
    this to ``True`` (with the relevant probe logic) the first time
    Strategy A is found insufficient.
    """
    return False


def resolve_plugin_path() -> Optional[str]:
    """Return the absolute path to the compiled plugin, or ``None``."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(
        here, "..", "..", "..", "..", "sana_fe", "plugins", "build",
        "libmimarsinan_subtractive_lif.so",
    ))
    return candidate if os.path.isfile(candidate) else None
