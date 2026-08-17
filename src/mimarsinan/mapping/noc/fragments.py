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
from mimarsinan.mapping.support.schedule.pass_planner import (
    plan_program_passes,
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


def execution_stage_placements(
    softcores: Sequence[LayoutSoftCoreSpec],
    pass_placements: Sequence[Sequence[Tuple[int, int]]],
    *,
    retimed: bool,
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """[R3] Per-EXECUTION-STAGE placements with stage-local hardcore indices.

    The runner maps EACH execution stage's cores from tile 0, core 0 — a
    re-timed program runs one small remapped chip per depth level, never the
    co-resident placement the pass planned. Pricing hops on the pass
    placement modeled 113.6 hops where the executed program measured a true
    0 (every level fit one tile). Grouping mirrors
    :func:`execution_stage_latencies` — one stage per (pass, segment, level)
    when re-timing is armed, else one per (pass, segment) — and hardcore
    indices are re-based per stage in placement order, the runner's own
    packing order.
    """
    stages: List[Tuple[Tuple[int, int], ...]] = []
    for placements in pass_placements:
        by_group: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for softcore_index, hardcore in placements:
            spec = softcores[softcore_index]
            segment = int(spec.segment_id or 0)
            level = int(spec.latency_tag or 0) if retimed else 0
            by_group.setdefault((segment, level), []).append(
                (softcore_index, hardcore)
            )
        for _key, group in sorted(by_group.items()):
            reindex: Dict[int, int] = {}
            local: List[Tuple[int, int]] = []
            for softcore_index, hardcore in group:
                local_core = reindex.setdefault(int(hardcore), len(reindex))
                local.append((softcore_index, local_core))
            stages.append(tuple(local))
    return tuple(stages)


@dataclass(frozen=True)
class PassProgram:
    """[E2] One pass's install cost: per-core committed rectangles + the
    residency law's answer (a resident pass installs nothing)."""

    core_cells: Tuple[int, ...]
    resident: bool

    @property
    def cores(self) -> int:
        return len(self.core_cells)


@dataclass(frozen=True)
class LayoutNocFragments:
    """Shape-only pass structure of one candidate.

    ``pass_placements``: per pass, ``(global softcore index, hardcore index)``
    for every placed unit (fragments repeat their origin). ``census``: the
    distinct-wire counts of the walked graph. ``pass_programs``: per pass, the
    occupied-core payload census and its residency. Together with the
    candidate's resolved floorplan and declared activity these are everything
    the NoC, latency and programming models need.
    """

    pass_placements: Tuple[Tuple[Tuple[int, int], ...], ...]
    #: None when no traffic axis asked for it — the placements (cheap) are
    #: always produced because the LATENCY model needs the pass structure.
    census: Optional[LayoutWireCensus]
    pass_programs: Tuple[PassProgram, ...] = ()
    #: [R3] Per-EXECUTION-STAGE placements (stage-local hardcore indices) —
    #: what the runner actually maps; None until the caller that knows the
    #: firing semantics derives them (``execution_stage_placements``).
    stage_placements: Optional[Tuple[Tuple[Tuple[int, int], ...], ...]] = None


def collect_noc_fragments(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    census: Optional[LayoutWireCensus],
    allow_scheduling: bool,
    allow_neuron_splitting: bool,
    allow_coalescing: bool,
    max_schedule_passes: int,
) -> LayoutNocFragments:
    """Place every pass of the candidate program and seal the placements.

    Consumes the ONE planner ``compute_mapping_stats`` and the builder consume
    [U1] but packs EVERY pass — the NoC estimate needs all placements, not
    just the worst pass's census. Cross-pass producer/consumer pairs are carry
    traffic, not mesh traffic; the estimator excludes them by pass membership.
    """
    softcores = list(softcores)
    global_index = {id(sc): i for i, sc in enumerate(softcores)}

    if allow_scheduling:
        plan = plan_program_passes(
            softcores, core_types,
            allow_coalescing=allow_coalescing,
            allow_splitting=allow_neuron_splitting,
            max_schedule_passes=max_schedule_passes,
        )
        if not plan.feasible:
            raise ValueError(
                f"NoC fragments need a schedulable candidate; segment(s) "
                f"{list(plan.infeasible_segments())} hold a softcore no "
                f"pass can pack"
            )
        pass_lists = [list(chunk) for chunk in plan.pass_lists]
        # Residency is a per-SEGMENT question: only a segment whose passes
        # the residency composition produced keeps weights installed.
        pass_resident = list(plan.resident_flags)
    else:
        pass_lists = [softcores]
        pass_resident = [False]

    per_pass: List[Tuple[Tuple[int, int], ...]] = []
    programs: List[PassProgram] = []
    for pass_softcores, resident in zip(pass_lists, pass_resident):
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
        programs.append(PassProgram(
            core_cells=tuple(
                int(snapshot.used_axons) * int(snapshot.used_neurons)
                for snapshot in (pack.used_core_snapshots or ())
            ),
            resident=bool(resident),
        ))
    return LayoutNocFragments(
        pass_placements=tuple(per_pass), census=census,
        pass_programs=tuple(programs),
    )
