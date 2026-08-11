"""Shape-only pass planning under a declared ``schedule_policy``.

The layout answer (search candidates, wizard preview, agent introspection) must
compose the passes the hard-core builder will compose — otherwise a scheduled
platform is searched against one program and deployed as another. This module is
the shape-only twin of ``packing/hybrid_build_scheduled.py``'s per-segment
decision: try the declared policy, fall back to the capacity split exactly as
the builder does.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.support.schedule.bank_clustered_law import (
    BankInstance,
    compose_bank_clustered_passes,
)
from mimarsinan.mapping.support.schedule.schedule_partitioner import (
    estimate_passes_for_layout_validated,
)

BANK_CLUSTERED = "bank_clustered"

PassPlan = Tuple[int, List[List[LayoutSoftCoreSpec]], bool, bool]


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
    """Bank-clustered passes over shape-only specs, or ``None`` when inapplicable.

    Mirrors ``try_bank_clustered_passes``' applicability with the facts a spec
    carries: every core must read a shared bank (``bank_id``), and the segment
    must hold no intra-segment dependency.

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


def plan_segment_passes(
    softcores: Sequence[LayoutSoftCoreSpec],
    budget: int,
    *,
    core_types: Sequence[LayoutHardCoreType],
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    schedule_policy: str = "pool",
    max_schedule_passes: int = 8,
) -> PassPlan:
    """One segment's passes under the declared policy.

    Returns ``(num_passes, pass_lists, all_feasible, policy_applied)``.
    ``policy_applied`` is False whenever the answer is the historical capacity
    split, so callers keep their previous behaviour byte-identical on every
    platform the policy does not reach.
    """
    if schedule_policy == BANK_CLUSTERED:
        chunks = bank_clustered_layout_passes(
            softcores, core_types, max_schedule_passes=max_schedule_passes,
        )
        # A composition that cannot pack is not the program deployment runs;
        # decline it rather than report an infeasible layout.
        if chunks is not None and all(
            pack_layout(
                softcores=chunk,
                core_types=core_types,
                allow_neuron_splitting=allow_splitting,
                allow_coalescing=allow_coalescing,
            ).feasible
            for chunk in chunks
        ):
            return len(chunks), chunks, True, True

    if budget <= 0:
        return 1, [list(softcores)], True, False
    n_passes, pass_lists, ok = estimate_passes_for_layout_validated(
        softcores, budget,
        max_hw_axons=max(ct.max_axons for ct in core_types) if core_types else 0,
        max_hw_neurons=max(ct.max_neurons for ct in core_types) if core_types else 0,
        allow_coalescing=allow_coalescing,
        allow_splitting=allow_splitting,
        core_types=core_types,
    )
    return n_passes, pass_lists, ok, False
