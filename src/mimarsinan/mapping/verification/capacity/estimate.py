"""Static SOUND lower bound on the hard cores an IR graph needs on a core budget, without placement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.platform.coalescing import coalescing_fragment_count
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
)


class CapacityExceededError(RuntimeError):
    """Raised when an IR graph provably cannot fit the declared core budget."""

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
    """Static hard-core capacity verdict for an IR graph on a core budget.

    ``scheduled`` records the SCHEDULED (PEAK-phase) path vs single-pool SUM;
    ``peak_phase_cores`` is the largest phase's occupancy, ``phase_count`` the
    reprogramming-pass count (1 when not scheduled).
    """

    cores_needed: int
    cores_available: int
    feasible: bool
    overflowing_segment: str | None
    per_segment: Dict[str, int] = field(default_factory=dict)
    scheduled: bool = False
    peak_phase_cores: int | None = None
    phase_count: int = 1

    def __post_init__(self) -> None:
        if self.peak_phase_cores is None:
            object.__setattr__(self, "peak_phase_cores", int(self.cores_needed))

    def raise_if_infeasible(self) -> "CapacityEstimate":
        """Raise :class:`CapacityExceededError` when not feasible; else return self."""
        if not self.feasible:
            raise CapacityExceededError(
                self.cores_needed, self.cores_available, self.overflowing_segment
            )
        return self


@dataclass(frozen=True)
class _SegmentBound:
    """One neural segment's diagonal lower bound plus its atomic-unit cost.

    ``atomic_unit`` (largest softcore's ``frags·groups``) cannot split across
    reprogramming phases, so it is the hard floor a scheduled phase must fit.
    """

    bound: int
    atomic_unit: int


def _segment_lower_bound(
    segment: NeuralSegment, max_axons: int, max_neurons: int
) -> _SegmentBound:
    """Diagonal-packing lower bound on hard cores for one neural segment.

    max(ceil(Σ axons / max_axons), ceil(Σ neurons / max_neurons), max per-core
    frags·groups); the per-core max is also the segment's atomic-unit cost.
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
    return _SegmentBound(
        bound=max(axon_bound, neuron_bound, max_per_core),
        atomic_unit=max_per_core,
    )


def _resolve_budget(
    platform_constraints: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Return ``(effective_max_axons, effective_max_neurons, total_core_count)`` via the mapping step's resolver."""
    cores: Sequence[dict[str, Any]] = platform_constraints.get("cores") or []
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


def _resolve_allow_scheduling(
    platform_constraints: Mapping[str, Any],
    allow_scheduling: bool | None,
) -> bool:
    """SSOT for the scheduling permission: explicit arg wins, else read the chip capabilities."""
    if allow_scheduling is not None:
        return bool(allow_scheduling)
    return ChipCapabilities.from_platform_constraints(platform_constraints).allow_scheduling


def estimate_cores_needed(
    ir_graph,
    platform_constraints: Mapping[str, Any],
    allow_scheduling: bool | None = None,
) -> CapacityEstimate:
    """Statically estimate the hard cores ``ir_graph`` needs on the core budget.

    ``allow_scheduling`` OFF ⇒ single-pool SUM verdict; ON ⇒ PEAK per-segment
    phase must fit and the largest atomic_unit must fit the whole budget.
    """
    max_axons, max_neurons, cores_available = _resolve_budget(platform_constraints)
    scheduled = _resolve_allow_scheduling(platform_constraints, allow_scheduling)

    per_segment: Dict[str, int] = {}
    bounds: list[tuple[str, _SegmentBound]] = []
    for segment in partition_ir_graph(ir_graph):
        if not isinstance(segment, NeuralSegment):
            continue
        sb = _segment_lower_bound(segment, max_axons, max_neurons)
        per_segment[segment.label] = sb.bound
        bounds.append((segment.label, sb))

    summed = sum(sb.bound for _, sb in bounds)
    if not scheduled:
        overflowing_segment = _first_sum_overflow(bounds, cores_available)
        feasible = summed <= cores_available
        return CapacityEstimate(
            cores_needed=int(summed),
            cores_available=int(cores_available),
            feasible=feasible,
            overflowing_segment=overflowing_segment if not feasible else None,
            per_segment=per_segment,
        )

    return _scheduled_estimate(
        bounds, per_segment, summed, cores_available
    )


def _first_sum_overflow(
    bounds: Sequence[tuple[str, "_SegmentBound"]], cores_available: int
) -> str | None:
    """Name the first segment at which the running SUM first exceeds the budget."""
    cumulative = 0
    for label, sb in bounds:
        cumulative += sb.bound
        if cumulative > cores_available:
            return label
    return None


def _scheduled_estimate(
    bounds: Sequence[tuple[str, "_SegmentBound"]],
    per_segment: Dict[str, int],
    summed: int,
    cores_available: int,
) -> CapacityEstimate:
    """PEAK-phase feasibility under the scheduled (fresh-pool-per-phase) model.

    Feasible iff the largest atomic unit fits the whole budget; peak phase is
    ``min(max_segment_bound, budget)``, ``phase_count`` = Σ ceil(bound / budget).
    """
    max_segment_bound = max((sb.bound for _, sb in bounds), default=0)
    max_atomic = max((sb.atomic_unit for _, sb in bounds), default=0)
    feasible = max_atomic <= cores_available

    peak_phase_cores = min(max_segment_bound, cores_available)
    phase_count = sum(
        math.ceil(sb.bound / cores_available) if cores_available > 0 else 0
        for _, sb in bounds
    )

    overflowing_segment: str | None = None
    if not feasible:
        for label, sb in bounds:
            if sb.atomic_unit > cores_available:
                overflowing_segment = label
                break

    return CapacityEstimate(
        cores_needed=int(summed),
        cores_available=int(cores_available),
        feasible=feasible,
        overflowing_segment=overflowing_segment,
        per_segment=per_segment,
        scheduled=True,
        peak_phase_cores=int(peak_phase_cores),
        phase_count=int(phase_count),
    )
