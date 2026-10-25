"""Per-execution-stage facts of a candidate's planned passes: the stage latencies and the stage-local placements the
runner maps, a spec's segment its builder segment (the walk's runs) where stated, else its depth segment."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.noc.wire_census import LayoutWireCensus
from mimarsinan.mapping.packing.pass_timing import project_planned_passes


def execution_stage_latencies(
    softcores: Sequence[LayoutSoftCoreSpec],
    pass_placements: Sequence[Sequence[Tuple[int, int]]],
    *,
    retimed: bool,
    census: Optional[LayoutWireCensus] = None,
    runs: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Per-execution-stage max core latency, one entry per stage.

    The deployed program executes one stage per (pass, segment, depth level) when per-hop re-timing is armed (each
    level re-based, so its internal max latency is 0), and otherwise one stage per (pass, segment) spanning that
    segment's whole latency range. Feeding this to ``chip_simulation.stage_timesteps.program_latency_steps``
    reproduces the step count the runner executes (measured: the MLP study run's 3 levels at T=4 → 3 x 5 = 15,
    exactly its sealed ``compute_steps``). [U1 W10] With the walk's wire census, a pass the fused segment pins spans
    its pinned latencies; with the walk's ``runs`` (``spec_runs``), the builder's segments (its fused segments) group
    the specs, else their depth segments do.
    """
    segment_of = spec_segments(softcores, runs)
    pinned = (project_planned_passes([int(s.output_count) for s in softcores],
                                     [[i for i, _ in p] for p in pass_placements], census.pair_wires,
                                     segment_of).pinned_spans
              if census is not None and not retimed else (None,) * len(pass_placements))
    stages: List[int] = []
    for placements, span in zip(pass_placements, pinned):
        if span is not None:
            stages.append(span)
            continue
        by_segment: Dict[int, List[int]] = {}
        for softcore_index, _hardcore in placements:
            spec = softcores[softcore_index]
            by_segment.setdefault(int(segment_of[softcore_index]), []).append(int(spec.latency_tag or 0))
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
    runs: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """[R3] Per-EXECUTION-STAGE placements with stage-local hardcore indices.

    The runner maps EACH execution stage's cores from tile 0, core 0 — a re-timed program runs one small remapped
    chip per depth level, never the co-resident placement the pass planned. Pricing hops on the pass placement
    modeled 113.6 hops where the executed program measured a true 0 (every level fit one tile). Grouping mirrors
    :func:`execution_stage_latencies` — one stage per (pass, segment, level) when re-timing is armed, else one per
    (pass, segment), a spec's segment its builder segment (``runs``), else its depth segment — and hardcore indices
    are re-based per stage in placement order, the runner's own packing order.
    """
    stages: List[Tuple[Tuple[int, int], ...]] = []
    segment_of = spec_segments(softcores, runs)
    for placements in pass_placements:
        by_group: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for softcore_index, hardcore in placements:
            spec = softcores[softcore_index]
            segment = segment_of[softcore_index]
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


def spec_segments(softcores: Sequence[LayoutSoftCoreSpec], runs: Optional[Sequence[int]]) -> List[int]:
    """The fused segment of each spec: its builder segment (``runs``), else its depth segment (``segment_id``)."""
    return [int(r) for r in runs] if runs is not None else [int(s.segment_id or 0) for s in softcores]
