"""Bias-delivery mode shared by the spiking activation nodes.

A deployment delivers a neuron's bias either via an on-chip per-neuron register
(``"on_chip"``, mode A) or as the weight on an always-on axon (``"param_encoded"``,
mode B). The two are **dynamically equivalent** in the differentiable forward: both
contribute a cumulative ``bias·(t_local+1)`` to the membrane (the always-on TTFS bias
spike fires once at the core's local window start and is ramp-integrated; the on-chip
register adds the bias every active cycle). So the activation nodes store the mode for
config fidelity / simulator dispatch but do **not** branch their dynamics on it.
"""

from __future__ import annotations

VALID_BIAS_MODES = ("on_chip", "param_encoded")


def validate_bias_mode(bias_mode: str) -> str:
    if bias_mode not in VALID_BIAS_MODES:
        raise ValueError(
            f"bias_mode must be one of {VALID_BIAS_MODES!r}; got {bias_mode!r}"
        )
    return bias_mode


def bias_mode_from_hardware_bias(hardware_bias: bool) -> str:
    """Map the mapping-time ``hardware_bias`` flag to the node's ``bias_mode``."""
    return "on_chip" if hardware_bias else "param_encoded"
