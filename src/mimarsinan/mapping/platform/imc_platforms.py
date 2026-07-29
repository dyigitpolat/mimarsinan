"""Named IMC crossbar geometries: a platform is a core-type list plus capability bits.

The geometry model is already generic — ``platform_constraints["cores"]`` is a
list of heterogeneous core types ``{max_axons, max_neurons, count}``. This module
only makes concrete chips *addressable by name* with traceable provenance.

Registered geometries below are PLACEHOLDERS. Sourcing real per-chip numbers is a
separate literature pass; ``provenance`` must name the citation before any
registered platform is used for a published measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

PLACEHOLDER_PROVENANCE = "PLACEHOLDER — geometry not yet sourced from literature"


@dataclass(frozen=True)
class IMCPlatform:
    """One IMC target: crossbar geometry, weight width, and mapping permissions."""

    name: str
    cores: tuple[Mapping[str, Any], ...]
    weight_bits: int
    provenance: str
    capabilities: Mapping[str, Any] = ()  # type: ignore[assignment]

    def validate(self) -> "IMCPlatform":
        """Every core type must declare a positive geometry and population."""
        if not self.cores:
            raise ValueError(f"IMC platform {self.name!r} declares no core types")
        for ct in self.cores:
            axons, neurons = int(ct["max_axons"]), int(ct["max_neurons"])
            count = int(ct.get("count", 1))
            if axons <= 0 or neurons <= 0 or count <= 0:
                raise ValueError(
                    f"IMC platform {self.name!r} core type {dict(ct)} must have "
                    f"positive max_axons, max_neurons and count")
        if int(self.weight_bits) <= 0:
            raise ValueError(f"IMC platform {self.name!r} needs positive weight_bits")
        return self

    @property
    def total_cores(self) -> int:
        return sum(int(ct.get("count", 1)) for ct in self.cores)

    @property
    def total_cells(self) -> int:
        return sum(
            int(ct["max_axons"]) * int(ct["max_neurons"]) * int(ct.get("count", 1))
            for ct in self.cores
        )

    def to_platform_constraints(self) -> dict[str, Any]:
        """Render the platform_constraints body the mapping pipeline consumes."""
        constraints: dict[str, Any] = {
            "cores": [dict(ct) for ct in self.cores],
            "weight_bits": int(self.weight_bits),
        }
        constraints.update(dict(self.capabilities or {}))
        return constraints


_REGISTRY: dict[str, IMCPlatform] = {}


def register_imc_platform(platform: IMCPlatform) -> IMCPlatform:
    """Register a validated named platform; re-registering a name is an error."""
    if platform.name in _REGISTRY:
        raise KeyError(f"IMC platform {platform.name!r} is already registered")
    _REGISTRY[platform.name] = platform.validate()
    return platform


def get_imc_platform(name: str) -> IMCPlatform:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown IMC platform {name!r}; known platforms: {known}")
    return _REGISTRY[name]


def imc_platform_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _square_grid(name: str, side: int, count: int, weight_bits: int) -> IMCPlatform:
    """A homogeneous ``count`` x (side x side) crossbar array."""
    return IMCPlatform(
        name=name,
        cores=({"max_axons": side, "max_neurons": side, "count": count},),
        weight_bits=weight_bits,
        provenance=PLACEHOLDER_PROVENANCE,
        capabilities={"allow_scheduling": True, "schedule_policy": "bank_clustered"},
    )


def heterogeneous_platform(
    name: str,
    core_types: tuple[tuple[int, int, int], ...],
    weight_bits: int,
    provenance: str = PLACEHOLDER_PROVENANCE,
) -> IMCPlatform:
    """Build a platform from ``(max_axons, max_neurons, count)`` tiles.

    Tile types stay SEPARATE all the way into ``platform_constraints["cores"]``:
    a mixed-tile chip must never be flattened to one max geometry, which would
    invent capacity no physical tile has.
    """
    return IMCPlatform(
        name=name,
        cores=tuple(
            {"max_axons": a, "max_neurons": n, "count": c} for a, n, c in core_types
        ),
        weight_bits=weight_bits,
        provenance=provenance,
        capabilities={"allow_scheduling": True, "schedule_policy": "bank_clustered"},
    )


register_imc_platform(_square_grid("imc_128x128", 128, 256, weight_bits=4))
register_imc_platform(_square_grid("imc_256x256", 256, 128, weight_bits=8))
register_imc_platform(_square_grid("imc_512x512", 512, 64, weight_bits=8))
register_imc_platform(
    heterogeneous_platform(
        "imc_mixed_tile", ((512, 512, 16), (128, 128, 128)), weight_bits=4
    )
)
