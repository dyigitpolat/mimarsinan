"""Bias-delivery mode (``on_chip`` / ``param_encoded``) shared by the spiking activation nodes."""

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
