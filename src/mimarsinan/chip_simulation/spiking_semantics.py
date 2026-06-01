"""Spiking-mode taxonomy and per-backend capability matrix.

Three orthogonal classifications, queried by intent rather than by hardcoded
mode tuples scattered across the codebase:

- ``ANALYTICAL_TTFS_MODES`` — closed-form, real-valued TTFS forward
  (``ttfs`` continuous, ``ttfs_quantized``). Cores exchange real activations.
- ``CYCLE_BASED_MODES`` — genuine per-cycle integrate-and-fire over binary
  spikes (``lif``, ``ttfs_cycle_based``). Sensitive to spike arrival timing.
- ``TTFS_FAMILY_MODES`` — every mode that uses TTFS firing/encoding config
  (``firing_mode='TTFS'``, latched spike encoding); the union of the analytical
  TTFS modes and ``ttfs_cycle_based``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

ANALYTICAL_TTFS_MODES: FrozenSet[str] = frozenset({"ttfs", "ttfs_quantized"})
# Backward-compat alias: ``TTFS_MODES`` historically meant the analytical family.
TTFS_MODES: FrozenSet[str] = ANALYTICAL_TTFS_MODES
TTFS_FAMILY_MODES: FrozenSet[str] = frozenset(
    {"ttfs", "ttfs_quantized", "ttfs_cycle_based"}
)
CYCLE_BASED_MODES: FrozenSet[str] = frozenset({"lif", "ttfs_cycle_based"})
LIF_MODES: FrozenSet[str] = frozenset({"lif", "rate"})
ALL_SPIKING_MODES: FrozenSet[str] = frozenset(
    {"lif", "rate", "ttfs", "ttfs_quantized", "ttfs_cycle_based"}
)

_ACT_QUANT_MODES: FrozenSet[str] = frozenset({"ttfs_quantized", "ttfs_cycle_based"})


def _norm(spiking_mode: str) -> str:
    return str(spiking_mode or "lif")


def requires_ttfs_firing(spiking_mode: str) -> bool:
    """Mode needs ``firing_mode='TTFS'`` / ``spike_generation_mode='TTFS'``."""
    return _norm(spiking_mode) in TTFS_FAMILY_MODES


def is_analytical_ttfs(spiking_mode: str) -> bool:
    """Mode uses the closed-form real-valued TTFS forward (not cycle-based)."""
    return _norm(spiking_mode) in ANALYTICAL_TTFS_MODES


def is_ttfs_cycle_based(spiking_mode: str) -> bool:
    """Mode is the genuine binary-spike, cycle-based TTFS mode."""
    return _norm(spiking_mode) == "ttfs_cycle_based"


def is_cycle_based(spiking_mode: str) -> bool:
    """Mode performs genuine per-cycle binary integrate-and-fire (LIF or TTFS-cycle)."""
    return _norm(spiking_mode) in CYCLE_BASED_MODES


def forces_activation_quantization(spiking_mode: str) -> bool:
    """Mode requires the activation-quantization chain (S-level activations)."""
    return _norm(spiking_mode) in _ACT_QUANT_MODES


@dataclass(frozen=True)
class BackendSpikingCapabilities:
    lif: bool
    ttfs: bool
    ttfs_quantized: bool
    ttfs_cycle_based: bool


_BACKEND_CAPS: dict[str, BackendSpikingCapabilities] = {
    "hcm": BackendSpikingCapabilities(True, True, True, True),
    "nevresim": BackendSpikingCapabilities(True, True, True, True),
    "unified": BackendSpikingCapabilities(True, True, True, True),
    "hybrid": BackendSpikingCapabilities(True, True, True, True),
    "sanafe": BackendSpikingCapabilities(True, True, True, True),
    "lava": BackendSpikingCapabilities(True, False, False, False),
    "loihi": BackendSpikingCapabilities(True, False, False, False),
    "training": BackendSpikingCapabilities(True, False, False, False),
}


def backend_capabilities(backend: str) -> BackendSpikingCapabilities:
    key = str(backend or "").lower()
    return _BACKEND_CAPS.get(key, BackendSpikingCapabilities(True, False, False, False))


def supports_spiking_mode(backend: str, spiking_mode: str) -> bool:
    spiking = _norm(spiking_mode)
    caps = backend_capabilities(backend)
    if spiking == "ttfs":
        return caps.ttfs
    if spiking == "ttfs_quantized":
        return caps.ttfs_quantized
    if spiking == "ttfs_cycle_based":
        return caps.ttfs_cycle_based
    return caps.lif


def require_spiking_mode_supported(
    spiking_mode: str,
    *,
    backend: str,
    context: str,
) -> None:
    if not supports_spiking_mode(backend, spiking_mode):
        raise ValueError(
            f"{context}: backend {backend!r} does not support "
            f"spiking_mode={spiking_mode!r}"
        )
