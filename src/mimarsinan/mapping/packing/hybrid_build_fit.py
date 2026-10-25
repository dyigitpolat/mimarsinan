"""[fix_b] The builder's fit rule: a chip that holds the program's unscheduled pack programs it once.

With scheduling allowed and no residency plan declared, U1 plans the passes, and the planner pins them where they fit
the declared pool side by side (``mark_fit_residency``). U1 alone judged that fit on its own passes, which on larger
programs spread over more cores than the pool: t0_43 at 104 + 104 ran 15 passes over 527 cores, 27,478,784 B per
period, where the toolchain's unscheduled program (the single shared pool, ``hybrid_build_pool._build_single_pool``:
one stage per segment, every segment side by side) packs 205 cores. So where U1's program writes every period and the
single pool packs with every core pinned, the single pool is the program: one pass per segment, nothing written per
period (K27). The shape-only plane states the same rule on its own pack (``residency_rules.unscheduled_pack_holds``).

[fix_c] The rule judges every plan (``residency_rules.fit_rule_applies``): under a declared floor or step index the
composition's passes, not pinned whole, reprogrammed every period on chips that hold the whole program (t0_43 at
104 + 104: the floor's 19 passes, 21,463,040 B per period). Where the declared pool admits the unscheduled pack, that
pack is the program, as under U1, and the mapping carries no composition plan (the step function's rows describe the
unit plan the fit replaces; its passes' marks pin every core). The stateless reload baseline is never judged.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.packing.hybrid_build_scheduled import _is_capacity_failure
from mimarsinan.mapping.packing.pass_timing import PassTimingError
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENT_CORES
from mimarsinan.mapping.support.residency_rules import fit_rule_applies


def fit_program(build_single_pool: Callable[[List[Any], Dict[int, Any]], None], stages: List[Any],
                maps: Dict[int, Any], residency: Any, *, residency_plan: Any, cells: int,
                cores_config: Sequence[dict]) -> Tuple[Dict[int, Any], Any]:
    """The fit rule on a scheduled build: (reindex maps, executed plan) of the program it stands for, replacing
    ``stages`` in place by the single pool where the rule applies and holds (the plan is then None)."""
    if not fit_rule_applies(residency_plan):
        return maps, residency
    held = single_pool_holds(build_single_pool, stages, cells=cells, cores_config=cores_config)
    if held is None:
        return maps, residency
    stages[:] = held[0]
    return held[1], None


def single_pool_holds(build_single_pool: Callable[[List[Any], Dict[int, Any]], None], scheduled: Sequence[Any], *,
                      cells: int, cores_config: Sequence[dict]) -> Optional[Tuple[List[Any], Dict[int, Any]]]:
    """The single pool's stages and reindex maps where the ``scheduled`` stages (U1's, or a declared plan's) write
    every period (not every core pinned) and the single pool packs with every core pinned; otherwise None. A program whose ``cells`` (its specs'
    cells at their compacted extents) exceed the pool's is no fit without a build; the single pool's capacity failure
    and its co-packing timing refusal are no fit; any other error of the build propagates."""
    if _all_pinned(scheduled):
        return None
    capacity = sum(int(c["max_axons"]) * int(c["max_neurons"]) * int(c["count"]) for c in cores_config)
    if not cells or cells > capacity:
        return None
    held: List[Any] = []
    maps: Dict[int, Any] = {}
    try:
        build_single_pool(held, maps)
    except PassTimingError:
        return None
    except RuntimeError as exc:
        if _is_capacity_failure(exc):
            return None
        raise
    return (held, maps) if _all_pinned(held) else None


def _all_pinned(stages: Sequence[Any]) -> bool:
    """Every core of every neural stage pinned (U1's fit pins mark all of a pass's cores or none; a declared plan
    marks each pass, possibly with part of its cores [fix_c])."""
    neural = [s for s in stages if s.kind == "neural"]
    return bool(neural) and all(getattr(s, SCHEDULE_RESIDENT_CORES, None) == frozenset(range(len(
        s.hard_core_mapping.cores))) for s in neural)
