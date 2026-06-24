"""Static placement-capacity estimate: hard cores an IR graph needs, WITHOUT placement.

The greedy ``placement_engine`` raises ``RuntimeError("No more hard cores
available")`` deep inside the pack when a config does not fit — a late,
non-diagnosable crash (E3: VGG16@224 needs ~hundreds of thousands of cores on a
1000-core budget). :func:`estimate_cores_needed` turns that into an EARLY,
diagnosable capacity verdict: a pure, fast, SOUND lower bound on the hard-core
count, computed straight from the IR graph and the platform core budget.

The bound mirrors the diagonal packer (``canonical._remaining_capacity``): a hard
core stacks softcores along the diagonal, consuming both axons and neurons, so a
neural segment needs at least
``max(ceil(Σ axons / max_axons), ceil(Σ neurons / max_neurons), max per-core
frags·groups)`` cores. Because it is a lower bound, the gate rejects only configs
that PROVABLY cannot fit (no packing strategy could place them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.platform.coalescing import coalescing_fragment_count
from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
)


class CapacityExceededError(RuntimeError):
    """Raised when an IR graph provably cannot fit the declared core budget.

    The diagnosable, EARLY replacement for the greedy packer's late
    ``RuntimeError("No more hard cores available")``: carries the static
    ``cores_needed`` vs ``cores_available`` counts and the first segment whose
    cumulative requirement overflows the budget.
    """

    def __init__(
        self,
        cores_needed: int,
        cores_available: int,
        overflowing_segment: str | None,
    ) -> None:
        self.cores_needed = int(cores_needed)
        self.cores_available = int(cores_available)
        self.overflowing_segment = overflowing_segment
        super().__init__(
            f"Placement capacity exceeded: the IR graph statically needs "
            f">= {self.cores_needed} hard cores but only {self.cores_available} "
            f"are available in the declared core budget. The first segment to "
            f"overflow is {overflowing_segment!r}. This config cannot be placed "
            "by ANY packing strategy on this budget; reject it before mapping "
            "rather than crashing late in the greedy packer with "
            '"No more hard cores available".'
        )


@dataclass(frozen=True)
class CapacityEstimate:
    """Static hard-core capacity verdict for an IR graph on a core budget."""

    cores_needed: int
    cores_available: int
    feasible: bool
    overflowing_segment: str | None
    per_segment: Dict[str, int] = field(default_factory=dict)

    def raise_if_infeasible(self) -> "CapacityEstimate":
        """Raise :class:`CapacityExceededError` when not feasible; else return self."""
        if not self.feasible:
            raise CapacityExceededError(
                self.cores_needed, self.cores_available, self.overflowing_segment
            )
        return self


def _segment_lower_bound(
    segment: NeuralSegment, max_axons: int, max_neurons: int
) -> int:
    """Diagonal-packing lower bound on hard cores for one neural segment.

    A hard core consumes both axons and neurons along the diagonal, so the
    segment needs at least ``ceil(Σ axons / max_axons)`` and
    ``ceil(Σ neurons / max_neurons)`` cores; an oversized softcore additionally
    forces its own ``frags·groups`` cores (coalescing fragments × neuron groups).
    The maximum of the three is a sound lower bound.
    """
    total_axons = 0
    total_neurons = 0
    max_per_core = 0
    for core in segment.nodes:
        in_count = int(core.get_input_count())
        out_count = int(core.get_output_count())
        total_axons += in_count
        total_neurons += out_count
        frags = coalescing_fragment_count(in_count, max_axons)
        groups = math.ceil(out_count / max_neurons) if max_neurons > 0 else 1
        max_per_core = max(max_per_core, frags * groups)
    axon_bound = math.ceil(total_axons / max_axons) if max_axons > 0 else 0
    neuron_bound = math.ceil(total_neurons / max_neurons) if max_neurons > 0 else 0
    return max(axon_bound, neuron_bound, max_per_core)


def _resolve_budget(
    platform_constraints: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Return ``(effective_max_axons, effective_max_neurons, total_core_count)``.

    Uses the same ``resolve_platform_mapping_params`` the mapping step uses so the
    effective axon width (minus a bias row when no ``has_bias``) matches what the
    packer actually offers. The budget is the total ``count`` across core types
    (the single physical pool the packer draws from across all segments).
    """
    cores: Sequence[Mapping[str, Any]] = platform_constraints.get("cores") or []
    if not cores:
        raise ValueError(
            "estimate_cores_needed: platform_constraints has no 'cores' budget."
        )
    allow_coalescing = bool(platform_constraints.get("allow_coalescing", False))
    params = resolve_platform_mapping_params(cores, allow_coalescing=allow_coalescing)
    total_count = sum(int(ct["count"]) for ct in cores)
    return (
        int(params.effective_max_axons),
        int(params.effective_max_neurons),
        int(total_count),
    )


def estimate_cores_needed(
    ir_graph,
    platform_constraints: Mapping[str, Any],
) -> CapacityEstimate:
    """Statically estimate the hard cores ``ir_graph`` needs on the core budget.

    Pure and fast (no placement, no simulation): partitions the IR into neural
    segments (host ComputeOps are barriers), computes each segment's diagonal
    lower bound, sums them (the budget is one pool consumed across segments), and
    reports feasibility plus the first segment whose cumulative requirement
    overflows. Reproduces the E3 split: VGG16@32 fits a 2048-core budget;
    VGG16@224 needs hundreds of thousands on a 1000-core budget.
    """
    max_axons, max_neurons, cores_available = _resolve_budget(platform_constraints)

    per_segment: Dict[str, int] = {}
    cumulative = 0
    overflowing_segment: str | None = None
    for segment in partition_ir_graph(ir_graph):
        if not isinstance(segment, NeuralSegment):
            continue
        bound = _segment_lower_bound(segment, max_axons, max_neurons)
        per_segment[segment.label] = bound
        cumulative += bound
        if overflowing_segment is None and cumulative > cores_available:
            overflowing_segment = segment.label

    feasible = cumulative <= cores_available
    return CapacityEstimate(
        cores_needed=int(cumulative),
        cores_available=int(cores_available),
        feasible=feasible,
        overflowing_segment=overflowing_segment if not feasible else None,
        per_segment=per_segment,
    )
