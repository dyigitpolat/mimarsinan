"""Shared spiking-simulator configuration constants and validation."""

from __future__ import annotations

from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC

#: [U-DAR] The spiking cores' dtype: the record's (``mapping/ir/deployment_dtype.py``), float64 on the integer membrane
#: lattice under weight quantization.
COMPUTE_DTYPE = SPIKING_ARITHMETIC.core_dtype

FIRING_MODES = frozenset({"Default", "Novena", "TTFS"})
SPIKE_MODES = frozenset({
    "Stochastic",
    "Deterministic",
    "FrontLoaded",
    "Uniform",
    "SpikeTrain",
    "TTFS",
})
THRESHOLDING_MODES = frozenset({"<", "<="})
TTFS_SPIKING_MODES = frozenset({"ttfs", "ttfs_quantized"})
TTFS_FIRING_MODES = frozenset({"TTFS"})


def validate_spiking_init(
    *,
    firing_mode: str,
    spike_mode: str,
    thresholding_mode: str,
) -> None:
    if firing_mode not in FIRING_MODES:
        raise ValueError(f"Invalid firing_mode: {firing_mode!r}")
    if spike_mode not in SPIKE_MODES:
        raise ValueError(f"Invalid spike_mode: {spike_mode!r}")
    if thresholding_mode not in THRESHOLDING_MODES:
        raise ValueError(f"Invalid thresholding_mode: {thresholding_mode!r}")
