"""Per-pass softcore placements + wire census — a candidate's NoC fragments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.noc.wire_census import LayoutWireCensus
from mimarsinan.mapping.support.schedule.schedule_budget import (
    effective_core_budget,
)
from mimarsinan.mapping.support.schedule.schedule_policy import (
    plan_segment_passes,
)


def execution_stage_latencies(
    softcores: Sequence[LayoutSoftCoreSpec],
    pass_placements: Sequence[Sequence[Tuple[int, int]]],
    *,
    retimed: bool,
) -> Tuple[int, ...]:
    """Per-execution-stage max core latency, one entry per stage.

    The deployed program executes one stage per (pass, segment, depth level)
    when per-hop re-timing is armed — each level re-based, so its internal
    max latency is 0 — and otherwise one stage per (pass, segment) spanning
    that segment's whole latency range. Feeding this to
    ``chip_simulation.stage_timesteps.program_latency_steps`` reproduces the
    step count the runner executes (measured: the MLP study run's 3 levels
    at T=4 → 3 x 5 = 15, exactly its sealed ``compute_steps``).
    """
    stages: List[int] = []
    for placements in pass_placements:
        by_segment: Dict[int, List[int]] = {}
        for softcore_index, _hardcore in placements:
            spec = softcores[softcore_index]
            segment = int(spec.segment_id or 0)
            by_segment.setdefault(segment, []).append(int(spec.latency_tag or 0))
        for _segment, latencies in sorted(by_segment.items()):
            levels = sorted(set(latencies))
            if retimed and len(levels) > 1:
                # Each level is its own re-based execution stage.
                stages.extend(0 for _ in levels)
            else:
                base = min(levels) if levels else 0
                stages.append(max(latencies) - base if latencies else 0)
    return tuple(stages)


@dataclass(frozen=True)
class LayoutNocFragments:
    """Shape-only NoC inputs of one candidate.

    ``pass_placements``: per pass, ``(global softcore index, hardcore index)``
    for every placed unit (fragments repeat their origin). ``census``: the
    distinct-wire counts of the walked graph. Together with the candidate's
    resolved floorplan and declared activity these are everything the NoC
    estimator needs.
    """

    pass_placements: Tuple[Tuple[Tuple[int, int], ...], ...]
    #: None when no traffic axis asked for it — the placements (cheap) are
    #: always produced because the LATENCY model needs the pass structure.
    census: Optional[LayoutWireCensus]


def collect_noc_fragments(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    census: Optional[LayoutWireCensus],
    allow_scheduling: bool,
    allow_neuron_splitting: bool,
    allow_coalescing: bool,
    schedule_policy: str,
    max_schedule_passes: int,
) -> LayoutNocFragments:
    """Place every pass of the candidate program and seal the placements.

    Mirrors ``compute_mapping_stats``'s pass planning (same planner, same
    packer) but packs EVERY pass — the NoC estimate needs all placements, not
    just the worst pass's census. Cross-pass producer/consumer pairs are carry
    traffic, not mesh traffic; the estimator excludes them by pass membership.
    """
    softcores = list(softcores)
    global_index = {id(sc): i for i, sc in enumerate(softcores)}

    if allow_scheduling:
        core_dicts = [
            {"max_axons": ct.max_axons, "max_neurons": ct.max_neurons,
             "count": ct.count}
            for ct in core_types
        ]
        budget = effective_core_budget(core_dicts)
        seg_softcores: Dict[int, List[LayoutSoftCoreSpec]] = {}
        for sc in softcores:
            sid = sc.segment_id if sc.segment_id is not None else 0
            seg_softcores.setdefault(sid, []).append(sc)
        pass_lists: List[List[LayoutSoftCoreSpec]] = []
        for sid in sorted(seg_softcores):
            _, seg_pass_lists, seg_ok, _ = plan_segment_passes(
                seg_softcores[sid], budget,
                core_types=core_types,
                allow_coalescing=allow_coalescing,
                allow_splitting=allow_neuron_splitting,
                schedule_policy=schedule_policy,
                max_schedule_passes=max_schedule_passes,
            )
            if not seg_ok:
                raise ValueError(
                    f"NoC fragments need a schedulable candidate; segment "
                    f"{sid} has a softcore no pass can pack"
                )
            pass_lists.extend(seg_pass_lists)
    else:
        pass_lists = [softcores]

    per_pass: List[Tuple[Tuple[int, int], ...]] = []
    for pass_softcores in pass_lists:
        pack = pack_layout(
            softcores=pass_softcores, core_types=core_types,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            collect_placements=True,
        )
        if not pack.feasible or pack.placements is None:
            raise ValueError(
                f"NoC fragments: a planned pass does not pack ({pack.error})"
            )
        per_pass.append(tuple(
            (global_index[id(pass_softcores[local])], hardcore)
            for local, hardcore in pack.placements
        ))
    return LayoutNocFragments(pass_placements=tuple(per_pass), census=census)
