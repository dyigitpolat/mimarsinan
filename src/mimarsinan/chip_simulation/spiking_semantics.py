"""Spiking-mode capability matrix for simulator backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

TTFS_MODES: FrozenSet[str] = frozenset({"ttfs", "ttfs_quantized"})
LIF_MODES: FrozenSet[str] = frozenset({"lif", "rate"})


@dataclass(frozen=True)
class BackendSpikingCapabilities:
    lif: bool
    ttfs: bool
    ttfs_quantized: bool


_BACKEND_CAPS: dict[str, BackendSpikingCapabilities] = {
    "hcm": BackendSpikingCapabilities(True, True, True),
    "nevresim": BackendSpikingCapabilities(True, True, True),
    "unified": BackendSpikingCapabilities(True, True, True),
    "hybrid": BackendSpikingCapabilities(True, True, True),
    "sanafe": BackendSpikingCapabilities(True, True, True),
    "lava": BackendSpikingCapabilities(True, False, False),
    "loihi": BackendSpikingCapabilities(True, False, False),
    "training": BackendSpikingCapabilities(True, False, False),
}


def backend_capabilities(backend: str) -> BackendSpikingCapabilities:
    key = str(backend or "").lower()
    return _BACKEND_CAPS.get(key, BackendSpikingCapabilities(True, False, False))


def supports_spiking_mode(backend: str, spiking_mode: str) -> bool:
    spiking = str(spiking_mode or "lif")
    caps = backend_capabilities(backend)
    if spiking in TTFS_MODES:
        if spiking == "ttfs":
            return caps.ttfs
        return caps.ttfs_quantized
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
