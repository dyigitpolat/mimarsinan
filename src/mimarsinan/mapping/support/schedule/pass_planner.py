"""One pass planner for every plane: residency-first, capacity fallback.

[U1] The ``schedule_policy`` enum is retired. Weight programming is the primary scheduling objective (owner
directive): whenever the bank-clustered law composes a residency streaming for a segment — every core reads a shared
bank, one latency level, every chunk packs — that composition IS the schedule; everything else takes the validated
capacity split. The shape-only answer (search candidates, wizard preview, agent introspection) and the hard-core
builder both consume THIS planner, so the composed program cannot fork between planes. Pass-count dominance was
deliberately rejected: a fitting flat pack programs one bank copy PER INSTANCE, while the residency composition programs
only the resident set — fewer programmed copies always, with pass inflation bounded by ``max_schedule_passes`` (the
law's floor) and its sync cost priced where schedules are priced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Collection, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.packing.canonical import OPERAND_ROLE
from mimarsinan.mapping.platform.mapping_structure import (
    BatchPipelineError, PositionMultiplexingError, refuse_batch_slots)
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.support.pipelined_multiplexing import (
    WavefrontPlan, pipelined_segment_passes, refuse_layer_copies, segment_producers, slot_turns)
from mimarsinan.mapping.support.position_multiplexing import multiplexed_segment_passes, shares_a_bank
from mimarsinan.mapping.support.schedule.bank_clustered_law import (
    BankInstance,
    compose_bank_clustered_passes,
    placement_regions_hold,
)
from mimarsinan.mapping.support.schedule.residency_composition import (
    ResidencyComposition, ShapeOnlyResidency, compose_shape_only_program)
from mimarsinan.mapping.support.schedule.schedule_budget import (
    effective_core_budget,
)
from mimarsinan.mapping.support.schedule.schedule_partitioner import (
    estimate_passes_for_layout_validated,
)


def resident_passes(pass_count: int, *, policy_applied: bool) -> Tuple[bool, ...]:
    """Which passes of ONE segment execute on already-programmed weights.

    The residency composition streams one shared bank over its passes: pass 0 programs the banks and every later pass
    places a SUBSET of the same (bank, region) geometry — verified core-by-core in ``mark_bank_residency`` — so it sends
    no weight payload and pays no per-core programming. Any other composition (the capacity split) places different
    weights each pass and reprograms all of them. Core INIT is not credited either way: a pass resets its cores' neuron
    state whether or not the weights stayed.
    """
    return tuple(bool(policy_applied) and index > 0 for index in range(int(pass_count)))


def compute_schedule_sync_count(per_segment_passes: Dict[int, int]) -> int:
    """Sync barriers from scheduled passes: a segment with N passes needs N-1 barriers."""
    return sum(max(n - 1, 0) for n in per_segment_passes.values())


def _has_intra_segment_dependency(softcores: Sequence[LayoutSoftCoreSpec]) -> bool:
    """True when the segment's cores are not all at one latency level.

    Latency is the longest path of NEURAL nodes feeding a core, so a consumer of another core in this segment strictly
    exceeds it. Equal tags therefore prove independence; unequal tags are treated as a dependency (conservative: a
    branchy graph could differ without one, and then the capacity path answers — the same answer as before this policy
    existed).
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

    Applicability with the facts a spec carries: every core must read a shared bank (``bank_id``), and the segment must
    hold no intra-segment dependency.

    CONSERVATISM (stated, because it is a real asymmetry): an instance is sized by the spec's own ``input_count``/
    ``output_count``, which a layout walk records BEFORE elimination, while the builder sizes it post-compaction
    (``compacted_core_extent``). The shape-only extent is therefore >= the deployed one, so this side can only DECLINE a
    segment the builder would have composed — never claim residency the hardware cannot hold. A caller holding the real
    cores can pre-shrink the specs (``spec_at_compacted_extent``); the two sides then coincide exactly.
    """
    if not softcores:
        return None
    if any(sc.bank_id is None for sc in softcores):
        return None
    if _has_intra_segment_dependency(softcores):
        return None

    instances = [BankInstance(bank_id=int(sc.bank_id or 0), axons=int(sc.input_count), neurons=int(sc.output_count))
                 for sc in softcores]
    cores_config = [{"max_axons": ct.max_axons, "max_neurons": ct.max_neurons, "count": ct.count} for ct in core_types]
    chunks = compose_bank_clustered_passes(instances, cores_config, max_schedule_passes=max_schedule_passes)
    if chunks is None:
        return None
    return [[softcores[index] for index in chunk] for chunk in chunks]


@dataclass(frozen=True)
class SegmentPassPlan:
    """One segment's composed passes and how they were composed."""

    pass_lists: Tuple[Tuple[LayoutSoftCoreSpec, ...], ...]
    feasible: bool
    residency_applied: bool
    #: [W02] Under position multiplexing, per pass: True where it programs its weights (a run's first pass).
    run_heads: Optional[Tuple[bool, ...]] = None
    #: [U-F4] Under pipelined position multiplexing, the wavefront composition (groups, each spec's pass).
    wavefront: Optional[WavefrontPlan] = None
    #: [thru:w2] Under a declared ``batch_pipeline_slots``: B, and per pass the input slot of each of its specs
    #: (``pass_lists[q][k]`` serves input slot ``input_slots[q][k]``). None undeclared (one input per run).
    slots: Optional[int] = None
    input_slots: Optional[Tuple[Tuple[int, ...], ...]] = None

    @property
    def pass_count(self) -> int:
        return len(self.pass_lists)

    @property
    def resident_flags(self) -> Tuple[bool, ...]:
        if self.run_heads is not None:
            return tuple(not head for head in self.run_heads)
        return resident_passes(self.pass_count, policy_applied=self.residency_applied)


def plan_segment_passes(
    softcores: Sequence[LayoutSoftCoreSpec],
    budget: int,
    *,
    core_types: Sequence[LayoutHardCoreType],
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    max_schedule_passes: int = 8,
    coalescing_group_ids: "Optional[Sequence[Optional[int]]]" = None,
    position_multiplexing: Optional[bool] = None,
    shares: Optional[Sequence[bool]] = None,
    pipelined_multiplexing: Optional[bool] = None,
    producers: Optional[Sequence[frozenset]] = None,
    layer_replication: Optional[bool] = None,
    wavefront_group_cores: Optional[int] = None,
    batch_pipeline_slots: Optional[int] = None,
    carry_input_slots: Optional[int] = None,
) -> SegmentPassPlan:
    """One segment's passes: the residency composition whenever the law
    composes one and every chunk packs, else the validated capacity split.
    [W02] Declared ``position_multiplexing``: one copy per shared bank serves its instances
    in turn (``shares``: which instances read their bank's copy; default all); [U-F4] with ``pipelined_multiplexing``
    its levels share passes as a wavefront (``producers``: per spec, the segment positions it reads). [thru:w1] With
    ``layer_replication`` each wavefront group's free cores hold copies of its banks (``wavefront_group_cores``: the
    cores one copy of a group may take); a declaration it cannot serve refuses by name (``refuse_layer_copies``).
    [thru:w2] With ``batch_pipeline_slots`` B a run serves B inputs: they share each wavefront group's passes, and
    [thru:w5] every pass outside the wavefront serves them in turn under one load (``_slot_turns``), the pass budget
    counted per run; a B the chip cannot serve (``carry_input_slots``) refuses by name (``refuse_batch_slots``).

    A composition that cannot pack is not the program deployment runs;
    it is declined rather than reported as an infeasible layout.
    ``coalescing_group_ids`` (aligned with *softcores*) keeps a coalesced
    wide-fan-in's fragments in ONE pass on the capacity path; [W04] absent, the
    specs' own groups (``coalescing_group_id``), so both planes keep them alike. A segment holding a dynamic
    operand bank's instances takes the capacity split (N2) and refuses a declared ``position_multiplexing``.
    """
    refuse_layer_copies(layer_replication, wavefront_group_cores, pipelined_multiplexing)
    refuse_batch_slots(batch_pipeline_slots, carry_input_slots, pipelined_multiplexing)
    if coalescing_group_ids is None and any(getattr(s, "coalescing_group_id", None) is not None for s in softcores):
        coalescing_group_ids = [getattr(s, "coalescing_group_id", None) for s in softcores]
    # N2: an operand bank's payload is rewritten every pass (design §2.7), so residency never composes over one, and
    # neither does W02's one copy serving positions in turn (a run's later passes are weight-resident): refused.
    operand = any(getattr(s, "core_role", None) == OPERAND_ROLE for s in softcores)
    if position_multiplexing and operand:
        raise PositionMultiplexingError("position_multiplexing serves one copy of a bank over passes, and a dynamic "
                                        "operand bank's payload is rewritten every pass (N2): refused beside one")
    if position_multiplexing and pipelined_multiplexing and shares_a_bank(softcores, shares):
        if producers is None:
            raise PositionMultiplexingError("pipelined_multiplexing orders a segment's instances by the instances "
                                            "they read, and this plan was given no producers")
        wave = pipelined_segment_passes(
            softcores, core_types, producers=producers, budget=budget, allow_coalescing=allow_coalescing,
            allow_splitting=allow_splitting, max_schedule_passes=max_schedule_passes, shares=shares,
            coalescing_group_ids=coalescing_group_ids, layer_replication=layer_replication,
            wavefront_group_cores=wavefront_group_cores, batch_pipeline_slots=batch_pipeline_slots,
            carry_input_slots=carry_input_slots)
        return SegmentPassPlan(wave.pass_lists, wave.feasible, False, run_heads=wave.run_heads, wavefront=wave,
                               slots=wave.slots, input_slots=wave.entry_slots)
    once = (lambda plan: _slot_turns(plan, batch_pipeline_slots, max_schedule_passes))
    if position_multiplexing and shares_a_bank(softcores, shares):
        mux = multiplexed_segment_passes(
            softcores, core_types, budget=budget, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
            max_schedule_passes=max_schedule_passes, shares=shares, coalescing_group_ids=coalescing_group_ids)
        return once(SegmentPassPlan(mux.pass_lists, mux.feasible, False, run_heads=mux.run_heads))
    chunks = None if operand else bank_clustered_layout_passes(
        softcores, core_types, max_schedule_passes=max_schedule_passes,
    )
    if chunks is not None and placement_regions_hold(
        chunks, core_types, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
    ):
        return once(SegmentPassPlan(tuple(tuple(chunk) for chunk in chunks), feasible=True, residency_applied=True))

    if budget <= 0:
        return once(SegmentPassPlan(pass_lists=(tuple(softcores),), feasible=True, residency_applied=False))
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
    return once(SegmentPassPlan(tuple(tuple(chunk) for chunk in pass_lists), feasible=ok, residency_applied=False))


def _slot_turns(plan: SegmentPassPlan, slots: Optional[int], max_schedule_passes: int) -> SegmentPassPlan:
    """[thru:w5] A segment planned outside every wavefront group, under a declared ``batch_pipeline_slots`` B: each of
    its passes holds by the group rule as a group of one pass (``slot_turns``), serving the B inputs in consecutive
    passes, input slot 0 to B-1, slot 0's with the pass's own load (its run head, the capacity split's every pass, the
    residency composition's first) and every other resident on it. The residency composition keeps its own marks (every
    pass after its first places the regions that pass programs). The pass budget is counted per run. Undeclared, the
    plan itself."""
    if slots is None:
        return plan
    n = int(slots)
    if n * len(plan.pass_lists) > int(max_schedule_passes):
        raise BatchPipelineError(f"batch_pipeline_slots {n}: a segment of {len(plan.pass_lists)} passes per input "
                                 f"needs {n * len(plan.pass_lists)} passes per run, above max_schedule_passes "
                                 f"{int(max_schedule_passes)}; declare a larger pass budget")
    turns = slot_turns(tuple(not resident for resident in plan.resident_flags), n)
    return replace(plan, pass_lists=tuple(plan.pass_lists[q] for q, _b, _head in turns),
                   run_heads=None if plan.residency_applied else tuple(head for _q, _b, head in turns), slots=n,
                   input_slots=tuple((b,) * len(plan.pass_lists[q]) for q, b, _head in turns))


@dataclass(frozen=True)
class ProgramPassPlan:
    """Every segment's plan, in segment order — the program the chip runs."""

    segments: Tuple[Tuple[int, SegmentPassPlan], ...]
    budget: int
    residency: Optional[ResidencyComposition] = None  # [E85] set under a declared residency_plan

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
    residency: Optional[ShapeOnlyResidency] = None,
    runs: Optional[Sequence[int]] = None,
    position_multiplexing: Optional[bool] = None,
    pipelined_multiplexing: Optional[bool] = None,
    producers: Optional[Sequence[Collection[int]]] = None,
    layer_replication: Optional[bool] = None,
    wavefront_group_cores: Optional[int] = None,
    batch_pipeline_slots: Optional[int] = None,
    carry_input_slots: Optional[int] = None,
) -> ProgramPassPlan:
    """Group by the specs' own ``segment_id`` and plan each segment. [U1 W9] ``runs`` (aligned with *softcores*: the
    walk's IR-run segment of each spec, ``residency_rules.spec_runs``) groups them by the builder's segments
    (``partition_ir_graph``), each planned as the builder plans it (``plan_segment_passes`` over the segment's specs,
    whose capacity split cuts at their ``segment_id``), so the plane's passes and pass counts are the builder's.
    [U-F4] ``producers``: per spec, the program indices of the specs it reads (each segment reads its own). [thru:k]
    The layer copies' keys reach every segment's plan (``plan_segment_passes``), [thru:w2] as do the batch
    pipeline's."""
    refuse_layer_copies(layer_replication, wavefront_group_cores, pipelined_multiplexing)
    refuse_batch_slots(batch_pipeline_slots, carry_input_slots, pipelined_multiplexing)
    if residency is not None:  # [E85] one residency composition over the whole program, the builder's refusal rules
        comp = compose_shape_only_program(softcores, core_types, residency, allow_coalescing=allow_coalescing,
                                          allow_splitting=allow_splitting, max_schedule_passes=max_schedule_passes)
        return ProgramPassPlan(tuple((s, SegmentPassPlan(tuple(tuple(softcores[i] for i in m) for m in ps), True, False))
                                     for s, ps in sorted(comp.segment_passes().items())), comp.budget, comp)
    core_dicts = [
        {"max_axons": ct.max_axons, "max_neurons": ct.max_neurons, "count": ct.count}
        for ct in core_types
    ]
    budget = effective_core_budget(core_dicts)

    seg_softcores: Dict[int, List[LayoutSoftCoreSpec]] = {}
    seg_members: Dict[int, List[int]] = {}
    for index, sc in enumerate(softcores):
        sid = int(runs[index]) if runs is not None else (sc.segment_id if sc.segment_id is not None else 0)
        seg_softcores.setdefault(sid, []).append(sc)
        seg_members.setdefault(sid, []).append(index)

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
                    position_multiplexing=position_multiplexing, pipelined_multiplexing=pipelined_multiplexing,
                    producers=segment_producers(producers, seg_members[sid]),
                    layer_replication=layer_replication, wavefront_group_cores=wavefront_group_cores,
                    batch_pipeline_slots=batch_pipeline_slots, carry_input_slots=carry_input_slots,
                ),
            )
            for sid in sorted(seg_softcores.keys())
        ),
        budget=budget,
    )
