"""Resolve platform core lists into IR-mapping parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class PlatformMappingParams:
    hardware_bias: bool
    effective_max_axons: int
    effective_max_neurons: int
    allow_coalescing: bool


def resolve_scalar_mapping_params(
    *,
    max_axons: int,
    max_neurons: int,
    hardware_bias: bool = False,
    allow_coalescing: bool = False,
) -> PlatformMappingParams:
    """Resolve tiling limits when only scalar max_axons/max_neurons are known."""
    effective_max_axons = int(max_axons) if hardware_bias else int(max_axons) - 1
    return PlatformMappingParams(
        hardware_bias=bool(hardware_bias),
        effective_max_axons=effective_max_axons,
        effective_max_neurons=int(max_neurons),
        allow_coalescing=bool(allow_coalescing),
    )


def resolve_platform_mapping_params(
    cores: Sequence[dict[str, Any]],
    *,
    allow_coalescing: bool = False,
) -> PlatformMappingParams:
    if not cores:
        raise ValueError("cores must be a non-empty list")
    hardware_bias = all(bool(ct.get("has_bias", True)) for ct in cores)
    max_axons = max(int(ct["max_axons"]) for ct in cores)
    max_neurons = max(int(ct["max_neurons"]) for ct in cores)
    effective_max_axons = max_axons if hardware_bias else max_axons - 1
    return PlatformMappingParams(
        hardware_bias=hardware_bias,
        effective_max_axons=effective_max_axons,
        effective_max_neurons=max_neurons,
        allow_coalescing=bool(allow_coalescing),
    )
