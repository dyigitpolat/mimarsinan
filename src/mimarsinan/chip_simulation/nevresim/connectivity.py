"""Nevresim chip connectivity mode (compile-time NTTP vs runtime span load)."""

from __future__ import annotations

from typing import Literal, Mapping

ConnectivityMode = Literal["compile_time", "runtime"]

DEFAULT_NEVRESIM_CONNECTIVITY_MODE: ConnectivityMode = "runtime"

_VALID: frozenset[str] = frozenset({"compile_time", "runtime"})


def _coerce(mode: str) -> ConnectivityMode:
    if mode not in _VALID:
        raise ValueError(f"nevresim connectivity mode must be compile_time or runtime, got {mode!r}")
    return mode  # type: ignore[return-value]


def default_nevresim_connectivity_mode() -> ConnectivityMode:
    """Deployment default for nevresim chip wiring."""
    return DEFAULT_NEVRESIM_CONNECTIVITY_MODE


def resolve_nevresim_connectivity_mode(config: Mapping[str, object] | None = None) -> ConnectivityMode:
    """Resolve from pipeline config, falling back to :data:`DEFAULT_NEVRESIM_CONNECTIVITY_MODE`."""
    if config is None:
        return default_nevresim_connectivity_mode()
    raw = config.get("nevresim_connectivity_mode")
    if raw is None:
        return default_nevresim_connectivity_mode()
    return _coerce(str(raw).strip())
