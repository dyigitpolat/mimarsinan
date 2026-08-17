"""One pass planner for every plane: residency-first, capacity fallback.

[U1] The ``schedule_policy`` enum is retired. Weight programming is the
primary scheduling objective (owner directive): whenever the bank-clustered
law composes a residency streaming for a segment — every core reads a shared
bank, one latency level, every chunk packs — that composition IS the
schedule; everything else takes the validated capacity split. The shape-only
answer (search candidates, wizard preview, agent introspection) and the
hard-core builder both consume THIS planner, so the composed program cannot
fork between planes. Pass-count dominance was deliberately rejected: a
fitting flat pack programs one bank copy PER INSTANCE, while the residency
composition programs only the resident set — fewer programmed copies always,
with pass inflation bounded by ``max_schedule_passes`` (the law's floor) and
its sync cost priced where schedules are priced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.support.schedule.bank_clustered_law import (
    BankInstance,
    compose_bank_clustered_passes,
)
from mimarsinan.mapping.support.schedule.schedule_budget import (
    effective_core_budget,
)
from mimarsinan.mapping.support.schedule.schedule_partitioner import (
    estimate_passes_for_layout_validated,
)


def resident_passes(pass_count: int, *, policy_applied: bool) -> Tuple[bool, ...]:
    """Which passes of ONE segment execute on already-programmed weights.

    The residency composition streams one shared bank over its passes:
    pass 0 programs the banks and every later pass places a SUBSET of the same
    (bank, region) geometry — verified core-by-core in ``mark_bank_residency``
    — so it sends no weight payload and pays no per-core programming. Any
    other composition (the capacity split) places different weights each pass
    and reprograms all of them. Core INIT is not credited either way: a pass
    resets its cores' neuron state whether or not the weights stayed.
    """
    return tuple(
        bool(policy_applied) and index > 0 for index in range(int(pass_count))
    )


def compute_schedule_sync_count(per_segment_passes: Dict[int, int]) -> int:
    """Sync barriers from scheduled passes: a segment with N passes needs N-1 barriers."""
    return sum(max(n - 1, 0) for n in per_segment_passes.values())


def _has_intra_segment_dependency(softcores: Sequence[LayoutSoftCoreSpec]) -> bool:
    """True when the segment's cores are not all at one latency level.

    Latency is the longest path of NEURAL nodes feeding a core, so a consumer of
    another core in this segment strictly exceeds it. Equal tags therefore prove
    independence; unequal tags are treated as a dependency (conservative: a
    branchy graph could differ without one, and then the capacity path answers —
    the same answer as before this policy existed).
    """
    tags = {0 if sc.latency_tag is None else int(sc.latency_tag) for sc in softcores}
    return len(tags) > 1


def bank_clustered_layout_passes(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    max_schedule_passes: int,
) -> "Optional[List[List[LayoutSoftCoreSpec]]]":
    """Residency passes over shape-only specs, or ``None`` when inapplicable.

    Applicability with the facts a spec carries: every core must read a shared
    bank (``bank_id``), and the segment must hold no intra-segment dependency.

    CONSERVATISM (stated, because it is a real asymmetry): an instance is sized
    by the spec's own ``input_count``/``output_count``, which a layout walk
    records BEFORE elimination, while the builder sizes it post-compaction
    (``compacted_core_extent``). The shape-only extent is therefore >= the
    deployed one, so this side can only DECLINE a segment the builder would
    have composed — never claim residency the hardware cannot hold. A caller
    holding the real cores can pre-shrink the specs
    (``spec_at_compacted_extent``); the two sides then coincide exactly.
    """
    if not softcores:
        return None
    if any(sc.bank_id is None for sc in softcores):
        return None
    if _has_intra_segment_dependency(softcores):
        return None

    instances = [
        BankInstance(
            bank_id=int(sc.bank_id or 0),
            axons=int(sc.input_count),
            neurons=int(sc.output_count),
        )
        for sc in softcores
    ]
    cores_config = [
        {"max_axons": ct.max_axons, "max_neurons": ct.max_neurons, "count": ct.count}
        for ct in core_types
    ]
    chunks = compose_bank_clustered_passes(
        instances, cores_config, max_schedule_passes=max_schedule_passes,
    )
    if chunks is None:
        return None
    return [[softcores[index] for index in chunk] for chunk in chunks]


@dataclass(frozen=True)
class SegmentPassPlan:
    """One segment's composed passes and how they were composed."""

    pass_lists: Tuple[Tuple[LayoutSoftCoreSpec, ...], ...]
    feasible: bool
    residency_applied: bool

    @property
    def pass_count(self) -> int:
        return len(self.pass_lists)

    @property
    def resident_flags(self) -> Tuple[bool, ...]:
        return resident_passes(
            self.pass_count, policy_applied=self.residency_applied
        )


def plan_segment_passes(
    softcores: Sequence[LayoutSoftCoreSpec],
    budget: int,
    *,
    core_types: Sequence[LayoutHardCoreType],
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    max_schedule_passes: int = 8,
    coalescing_group_ids: "Optional[Sequence[Optional[int]]]" = None,
) -> SegmentPassPlan:
    """One segment's passes: the residency composition whenever the law
    composes one and every chunk packs, else the validated capacity split.

    A composition that cannot pack is not the program deployment runs;
    it is declined rather than reported as an infeasible layout.
    ``coalescing_group_ids`` (aligned with *softcores*) keeps a coalesced
    wide-fan-in's fragments in ONE pass on the capacity path — the caller
    holding real cores passes them; the shape-only planes have none.
    """
    chunks = bank_clustered_layout_passes(
        softcores, core_types, max_schedule_passes=max_schedule_passes,
    )
    if chunks is not None and all(
        pack_layout(
            softcores=chunk,
            core_types=core_types,
            allow_neuron_splitting=allow_splitting,
            allow_coalescing=allow_coalescing,
        ).feasible
        for chunk in chunks
    ):
        return SegmentPassPlan(
            pass_lists=tuple(tuple(chunk) for chunk in chunks),
            feasible=True,
            residency_applied=True,
        )

    if budget <= 0:
        return SegmentPassPlan(
            pass_lists=(tuple(softcores),), feasible=True,
            residency_applied=False,
        )
    n_passes, pass_lists, ok = estimate_passes_for_layout_validated(
        softcores, budget,
        max_hw_axons=max(ct.max_axons for ct in core_types) if core_types else 0,
        max_hw_neurons=max(ct.max_neurons for ct in core_types) if core_types else 0,
        allow_coalescing=allow_coalescing,
        allow_splitting=allow_splitting,
        core_types=core_types,
        coalescing_group_ids=coalescing_group_ids,
    )
    del n_passes
    return SegmentPassPlan(
        pass_lists=tuple(tuple(chunk) for chunk in pass_lists),
        feasible=ok,
        residency_applied=False,
    )


@dataclass(frozen=True)
class ProgramPassPlan:
    """Every segment's plan, in segment order — the program the chip runs."""

    segments: Tuple[Tuple[int, SegmentPassPlan], ...]
    budget: int

    @property
    def feasible(self) -> bool:
        return all(plan.feasible for _, plan in self.segments)

    @property
    def pass_lists(self) -> Tuple[Tuple[LayoutSoftCoreSpec, ...], ...]:
        return tuple(
            chunk for _, plan in self.segments for chunk in plan.pass_lists
        )

    @property
    def resident_flags(self) -> Tuple[bool, ...]:
        return tuple(
            flag for _, plan in self.segments for flag in plan.resident_flags
        )

    @property
    def per_segment_passes(self) -> Dict[int, int]:
        return {
            sid: max(plan.pass_count, 1) for sid, plan in self.segments
        }

    @property
    def total_pass_count(self) -> int:
        return sum(self.per_segment_passes.values())

    @property
    def sync_count(self) -> int:
        return compute_schedule_sync_count(self.per_segment_passes)

    def infeasible_segments(self) -> Tuple[int, ...]:
        return tuple(sid for sid, plan in self.segments if not plan.feasible)


def plan_program_passes(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    max_schedule_passes: int = 8,
) -> ProgramPassPlan:
    """Group by the specs' own ``segment_id`` and plan each segment."""
    core_dicts = [
        {"max_axons": ct.max_axons, "max_neurons": ct.max_neurons, "count": ct.count}
        for ct in core_types
    ]
    budget = effective_core_budget(core_dicts)

    seg_softcores: Dict[int, List[LayoutSoftCoreSpec]] = {}
    for sc in softcores:
        sid = sc.segment_id if sc.segment_id is not None else 0
        seg_softcores.setdefault(sid, []).append(sc)

    return ProgramPassPlan(
        segments=tuple(
            (
                sid,
                plan_segment_passes(
                    seg_softcores[sid], budget,
                    core_types=core_types,
                    allow_coalescing=allow_coalescing,
                    allow_splitting=allow_splitting,
                    max_schedule_passes=max_schedule_passes,
                ),
            )
            for sid in sorted(seg_softcores.keys())
        ),
        budget=budget,
    )
