"""Resolve platform core lists into IR-mapping parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from mimarsinan.models.nn.activations.bias_mode import bias_mode_from_hardware_bias


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
    hardware_bias = declares_hardware_bias(cores)
    max_axons = max(int(ct["max_axons"]) for ct in cores)
    max_neurons = max(int(ct["max_neurons"]) for ct in cores)
    effective_max_axons = max_axons if hardware_bias else max_axons - 1
    return PlatformMappingParams(
        hardware_bias=hardware_bias,
        effective_max_axons=effective_max_axons,
        effective_max_neurons=max_neurons,
        allow_coalescing=bool(allow_coalescing),
    )


def declares_hardware_bias(cores: Any) -> bool:
    """Whether every declared core type carries an on-chip bias lane.

    TOTAL over every shape a draft can hold: the question is ``has_bias``
    alone, so it never inherits the ``max_axons``/``max_neurons`` precondition
    that resolving mapping params carries. An undeclared grid reads as the
    framework default platform, which has a lane; a declaration no reader can
    parse as a core grid declares no lane, and the document's own shape
    validators own that complaint.
    """
    if not cores:
        return True
    if not isinstance(cores, (list, tuple)):
        return False
    if not all(isinstance(core_type, Mapping) for core_type in cores):
        return False
    return all(bool(core_type.get("has_bias", True)) for core_type in cores)


def bias_mode_for_cores(cores: Optional[Sequence[dict[str, Any]]]) -> str:
    """THE bias-delivery mode of a declared core grid.

    One rule for both readers (the pipeline's ``resolve_bias_mode`` and the
    config derivation): a grid where every core type carries a bias lane
    delivers ``on_chip``, anything else ``param_encoded``. Total, because the
    config-derivation reader asks it of a RAW, un-normalized draft grid.
    """
    return bias_mode_from_hardware_bias(declares_hardware_bias(cores))
