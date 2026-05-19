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
