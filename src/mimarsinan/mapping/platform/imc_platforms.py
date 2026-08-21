"""Named IMC crossbar geometries: a platform is a core-type list plus capability bits.

The geometry model is already generic — ``platform_constraints["cores"]`` is a
list of heterogeneous core types ``{max_axons, max_neurons, count}``. This module
only makes concrete chips *addressable by name* with traceable provenance.

The 12 literature-sourced platforms registered below are transcribed verbatim
(geometry, weight_bits, provenance quote + location) from the extraction-card
pipeline in the structured-elimination paper workspace:
``papers/structured_elimination_aaai/research_artifacts/13_chip_geometries.json``
(see the sibling ``13_chip_geometries.md`` for the curated eligibility table).
Every registered platform's ``provenance`` must name a citation (or, for
synthetic capability-exercise platforms, say so explicitly) before it can be
retrieved via ``get_imc_platform`` — see the PLACEHOLDER_PROVENANCE ratchet
below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mimarsinan.chip_simulation.soma_axes import (
    MEMBRANE_BITS_KEY,
    PER_AXON_SIGN,
    WEIGHT_SIGN_GRANULARITY_KEY,
)

PLACEHOLDER_PROVENANCE = "PLACEHOLDER — geometry not yet sourced from literature"

#: Claim-eligibility classes a registered platform can carry (charter gate G5,
#: `13_chip_geometries.md` §"S2 / allocation-claim eligibility"):
#:   - headline-eligible: quote-sourced geometry AND quote-sourced core population;
#:     usable for S2/allocation headline claims.
#:   - curve-only: geometry quote-sourced, but the core population is either
#:     derived from a quoted hierarchy, approximate, or the geometry itself is a
#:     PI-conditioned model (e.g. a dense-equivalent of a non-crossbar chip);
#:     usable for population-curve / sensitivity claims, never a threshold headline.
#:   - occupancy-only: geometry quote-sourced but the population is entirely ours
#:     (paper never fixes a chip-level array count); usable for per-core
#:     occupancy/utilization metrics only.
#:   - quarantined: quote-sourced but degenerate (e.g. a single-core demo
#:     description) — excluded from S-metric and multi-core allocation
#:     aggregates; geometry/occupancy only.
#:   - synthetic: not a real chip at all — a capability-exercise fixture for the
#:     mapping/packing machinery (e.g. heterogeneous-tile coverage). Never cited
#:     as hardware evidence.
#:   - unclassified: no claim-eligibility ruling has been made (default for
#:     ad-hoc, non-registered ``IMCPlatform`` instances built directly in tests).
CLAIM_ELIGIBILITY_CLASSES = (
    "headline-eligible",
    "curve-only",
    "occupancy-only",
    "quarantined",
    "synthetic",
    "unclassified",
)


@dataclass(frozen=True)
class IMCPlatform:
    """One IMC target: crossbar geometry, weight width, and mapping permissions."""

    name: str
    cores: tuple[Mapping[str, Any], ...]
    weight_bits: int
    provenance: str
    capabilities: Mapping[str, Any] = ()  # type: ignore[assignment]
    claim_eligibility: str = "unclassified"

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
        if self.claim_eligibility not in CLAIM_ELIGIBILITY_CLASSES:
            raise ValueError(
                f"IMC platform {self.name!r} has unknown claim_eligibility "
                f"{self.claim_eligibility!r}; must be one of {CLAIM_ELIGIBILITY_CLASSES}")
        return self

    def validate_for_run(self) -> "IMCPlatform":
        """Registry-retrieval gate: geometry validity PLUS the provenance ratchet.

        Ad-hoc ``IMCPlatform(...)`` instances built directly (as existing tests
        do, with a throwaway ``provenance="placeholder"`` string) never call this
        — only ``get_imc_platform`` does. That keeps the ratchet scoped to the
        named-registry path without breaking tests that exercise the dataclass
        in isolation.
        """
        self.validate()
        if self.provenance == PLACEHOLDER_PROVENANCE:
            raise ValueError(
                f"IMC platform {self.name!r} carries PLACEHOLDER_PROVENANCE and "
                "cannot be retrieved for a run. Source its geometry from "
                "literature (or give it an explicit synthetic-platform "
                "provenance string) before it is used.")
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
    """Fetch a registered platform, ratcheted: placeholder provenance fails loud."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown IMC platform {name!r}; known platforms: {known}")
    return _REGISTRY[name].validate_for_run()


def imc_platform_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _square_grid(
    name: str, side: int, count: int, weight_bits: int, provenance: str,
    claim_eligibility: str = "synthetic",
) -> IMCPlatform:
    """A homogeneous ``count`` x (side x side) crossbar array."""
    return IMCPlatform(
        name=name,
        cores=({"max_axons": side, "max_neurons": side, "count": count},),
        weight_bits=weight_bits,
        provenance=provenance,
        capabilities={"allow_scheduling": True},
        claim_eligibility=claim_eligibility,
    )


def heterogeneous_platform(
    name: str,
    core_types: tuple[tuple[int, int, int], ...],
    weight_bits: int,
    provenance: str = PLACEHOLDER_PROVENANCE,
    claim_eligibility: str = "unclassified",
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
        capabilities={"allow_scheduling": True},
        claim_eligibility=claim_eligibility,
    )


# ---------------------------------------------------------------------------
# Synthetic capability-exercise platforms — NOT real chips. Named honestly so
# the provenance ratchet stays strict without ever implying literature backing.
# ---------------------------------------------------------------------------

_SYNTHETIC_PROVENANCE = "synthetic capability-exercise platform — not a real chip"

register_imc_platform(
    _square_grid("imc_128x128", 128, 256, weight_bits=4, provenance=_SYNTHETIC_PROVENANCE)
)
register_imc_platform(
    _square_grid("imc_256x256", 256, 128, weight_bits=8, provenance=_SYNTHETIC_PROVENANCE)
)
register_imc_platform(
    _square_grid("imc_512x512", 512, 64, weight_bits=8, provenance=_SYNTHETIC_PROVENANCE)
)
register_imc_platform(
    heterogeneous_platform(
        "imc_mixed_tile", ((512, 512, 16), (128, 128, 128)), weight_bits=4,
        provenance=_SYNTHETIC_PROVENANCE, claim_eligibility="synthetic",
    )
)


# ---------------------------------------------------------------------------
# Deployment targets registered outside the literature-transcription pipeline.
# These carry their OWN provenance quote (they are not rows of
# `13_chip_geometries.json`, whose 12-platform table is untouched) and their own
# eligibility ruling.
# ---------------------------------------------------------------------------

_ODIN_PROVENANCE = (
    'frenkel2019odin: "ODIN is based on a single 256-neuron 64k-synapse crossbar '
    'neurosynaptic core ... each synapse occupies 4 bits" (Trans. BioCAS 13(1), '
    "pp.145-158; upstream doc/README.md Sec.1 and Sec.3.2), transcribed from the "
    "RTL at ChFrenkel/ODIN @ 1781931. The DECLARED max_axons is the LOGICAL twin "
    "of the quoted 256 physical rows: weight_sign_granularity='per_axon' spends "
    "an excitatory/inhibitory row pair per logical slot (SPI_SYN_SIGN, "
    "src/synaptic_core.v:128), so 128 logical x 2 = the 256 quoted rows and "
    "128 x 256 x 2 = the quoted 64k synapses."
)

register_imc_platform(
    IMCPlatform(
        name="odin_stock_core",
        cores=({"max_axons": 128, "max_neurons": 256, "count": 1,
                "has_bias": False},),
        weight_bits=4,
        provenance=_ODIN_PROVENANCE,
        capabilities={
            WEIGHT_SIGN_GRANULARITY_KEY: PER_AXON_SIGN,
            MEMBRANE_BITS_KEY: 8,
            # No inter-core partial-sum transfer and no re-thresholding of a
            # partial sum: a per-event soma law makes both count-changing.
            "allow_coalescing": False,
            "allow_neuron_splitting": False,
            "allow_scheduling": False,
        },
        # Population 1 is the degenerate single-core case the curated table
        # quarantines, and the geometry above is a conditioned LOGICAL model of
        # the quoted crossbar — so this is occupancy/geometry evidence only,
        # never an allocation headline.
        claim_eligibility="quarantined",
    )
)


# The 12 literature-sourced platforms live in their own module (data only,
# registered via the seams above) purely to stay under this file's LOC budget
# — see `imc_platforms_literature.py`. Importing it here is what makes those
# registrations happen: anyone importing `imc_platforms` gets the full roster.
import mimarsinan.mapping.platform.imc_platforms_literature  # noqa: E402,F401  # pyright: ignore[reportUnusedImport] — registers the 12 literature platforms
