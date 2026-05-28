"""Tests for nevresim connectivity mode defaults."""

from __future__ import annotations

from mimarsinan.chip_simulation.nevresim.connectivity import (
    DEFAULT_NEVRESIM_CONNECTIVITY_MODE,
    default_nevresim_connectivity_mode,
    resolve_nevresim_connectivity_mode,
)


def test_default_is_runtime() -> None:
    assert DEFAULT_NEVRESIM_CONNECTIVITY_MODE == "runtime"
    assert default_nevresim_connectivity_mode() == "runtime"


def test_resolve_from_config() -> None:
    assert resolve_nevresim_connectivity_mode({}) == "runtime"
    assert resolve_nevresim_connectivity_mode({"nevresim_connectivity_mode": "compile_time"}) == "compile_time"
