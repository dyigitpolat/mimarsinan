"""Dry-run placement feasibility by running the real hybrid hard-core packer on the IR graph (CPU, no GPU/sim)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)

_SEGMENT_RE = re.compile(r"segment '([^']+)'")


@dataclass(frozen=True)
class PackFeasibility:
    """Definitive placement verdict from running the real hybrid packer."""

    feasible: bool
    hard_cores: int | None
    overflowing_segment: str | None
    error: str | None


def dryrun_pack_feasible(
    ir_graph, platform_constraints: Mapping[str, Any]
) -> PackFeasibility:
    """Run the real hybrid hard-core packer on ``ir_graph`` and report feasibility.

    Weight-independent (structural threshold grouping), so an untrained model's
    dry-run is bit-identical to the trained deployment's packing.
    """
    strategy = MappingStrategy.resolve(
        ChipCapabilities.from_platform_constraints(platform_constraints)
    )
    try:
        hybrid_mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph,
            cores_config=platform_constraints["cores"],
            strategy=strategy,
        )
    except RuntimeError as exc:
        message = str(exc)
        match = _SEGMENT_RE.search(message)
        return PackFeasibility(
            feasible=False,
            hard_cores=None,
            overflowing_segment=match.group(1) if match else None,
            error=message,
        )

    hard_cores = sum(
        len(segment.cores) for segment in hybrid_mapping.get_neural_segments()
    )
    return PackFeasibility(
        feasible=True,
        hard_cores=int(hard_cores),
        overflowing_segment=None,
        error=None,
    )
