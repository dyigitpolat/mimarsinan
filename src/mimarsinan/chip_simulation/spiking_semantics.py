"""Spiking-mode taxonomy and per-backend capability matrix, queried by intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

ANALYTICAL_TTFS_MODES: FrozenSet[str] = frozenset({"ttfs", "ttfs_quantized"})
TTFS_MODES: FrozenSet[str] = ANALYTICAL_TTFS_MODES
TTFS_FAMILY_MODES: FrozenSet[str] = frozenset(
    {"ttfs", "ttfs_quantized", "ttfs_cycle_based"}
)
CYCLE_BASED_MODES: FrozenSet[str] = frozenset({"lif", "ttfs_cycle_based"})
LIF_MODES: FrozenSet[str] = frozenset({"lif"})
ALL_SPIKING_MODES: FrozenSet[str] = frozenset(
    {"lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"}
)

_REMOVED_SPIKING_MODES: dict = {"rate": "lif"}

_ACT_QUANT_MODES: FrozenSet[str] = frozenset({"ttfs_quantized", "ttfs_cycle_based"})


def _norm(spiking_mode: str) -> str:
    return str(spiking_mode or "lif")


def require_known_spiking_mode(spiking_mode: str) -> str:
    """Return the normalized mode, or raise ``ValueError`` if it is not deployable."""
    mode = _norm(spiking_mode)
    if mode in ALL_SPIKING_MODES:
        return mode
    if mode in _REMOVED_SPIKING_MODES:
        raise ValueError(
            f"spiking_mode {mode!r} was removed from the deploy taxonomy; "
            f"use {_REMOVED_SPIKING_MODES[mode]!r} instead."
        )
    raise ValueError(
        f"unknown spiking_mode {mode!r}; valid modes: {sorted(ALL_SPIKING_MODES)}."
    )


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


def is_lif(spiking_mode: str) -> bool:
    """Mode is the plain rate-coded LIF deployment family."""
    return _norm(spiking_mode) in LIF_MODES


def forces_activation_quantization(spiking_mode: str) -> bool:
    """Mode requires the activation-quantization chain (S-level activations)."""
    return _norm(spiking_mode) in _ACT_QUANT_MODES


def is_bit_parity_lossless_conversion(spiking_mode: str) -> bool:
    """Mode whose trained forward is per-neuron bit-equivalent to its deployed executor.

    Exactly the analytic BIT_PARITY family (5u): analytical ``ttfs``. The
    near-lossless modes (lif, ttfs_quantized, synchronized) keep a real endpoint
    gap and join only after that gap is verified sub-SE; cascaded is lossy.
    """
    return is_analytical_ttfs(spiking_mode) and not forces_activation_quantization(
        spiking_mode
    )


DEFAULT_TTFS_CYCLE_SCHEDULE = "cascaded"
TTFS_CYCLE_SCHEDULES: FrozenSet[str] = frozenset({"cascaded", "synchronized"})


def ttfs_cycle_schedule(schedule) -> str:
    """Normalize the ttfs_cycle schedule value, defaulting to ``cascaded``."""
    s = str(schedule or DEFAULT_TTFS_CYCLE_SCHEDULE)
    return s if s in TTFS_CYCLE_SCHEDULES else DEFAULT_TTFS_CYCLE_SCHEDULE


def is_explicit_ttfs_cycle_schedule(schedule) -> bool:
    """``schedule`` explicitly names a known ttfs_cycle schedule (no default applied)."""
    return schedule in TTFS_CYCLE_SCHEDULES


def is_cascaded_ttfs(spiking_mode: str, schedule) -> bool:
    """ttfs_cycle_based running the greedy pipelined (cascaded) schedule."""
    return is_ttfs_cycle_based(spiking_mode) and ttfs_cycle_schedule(schedule) == "cascaded"


def is_synchronized_ttfs(spiking_mode: str, schedule) -> bool:
    """ttfs_cycle_based running the exact sequential (synchronized) schedule."""
    return is_ttfs_cycle_based(spiking_mode) and ttfs_cycle_schedule(schedule) == "synchronized"


def uses_ttfs_floor_ceil_convention(spiking_mode: str, schedule=None) -> bool:
    """Modes that train the floor-staircase + half-step-bias convention yet deploy the ceil TTFS kernel.

    ``ttfs_quantized`` and the synchronized floor-collapse; both take the half-step bias
    compensation and are excluded from the bit-exact per-neuron parity gate.
    """
    return _norm(spiking_mode) == "ttfs_quantized" or is_synchronized_ttfs(
        spiking_mode, schedule
    )


DEFAULT_FIRING_MODE = "Default"
NOVENA_FIRING_MODE = "Novena"


def is_default_firing_mode(firing_mode) -> bool:
    """Firing mode with the Default (subtractive-reset) LIF semantics."""
    return firing_mode == DEFAULT_FIRING_MODE


def is_novena_firing_mode(firing_mode) -> bool:
    """Firing mode with the Novena (zero-reset) LIF semantics."""
    return firing_mode == NOVENA_FIRING_MODE


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
