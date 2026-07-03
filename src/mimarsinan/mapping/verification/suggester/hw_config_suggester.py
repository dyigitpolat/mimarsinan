from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.verification.suggester.hw_suggestion_helpers import (
    _count_cores_needed_two_types, _count_per_type_usage, _dimension_bounds,
    _make_two_core_types, _min_hw_coverage,
    _occupancy_ok, _pack_with_two_types,
)
from mimarsinan.mapping.verification.suggester.hw_suggestion_types import HardwareSuggestion

def suggest_hardware_config(
    softcores: Sequence[LayoutSoftCoreSpec],
    *,
    allow_coalescing: bool = False,
    hardware_bias: bool = True,
    axon_granularity: int = 1,
    neuron_granularity: int = 1,
    safety_margin: float = 0.15,
    allow_neuron_splitting: bool = False,
) -> HardwareSuggestion:
    """Suggest two core types and pool counts for the given layout softcores."""
    if not softcores:
        return HardwareSuggestion(
            core_types=[],
            total_cores=0,
            rationale="No softcores — nothing to map.",
        )

    softcores_list = list(softcores)
    max_ax, max_neu, _, _ = _dimension_bounds(softcores_list)
    h, w = _min_hw_coverage(
        softcores_list, allow_coalescing, axon_granularity, neuron_granularity,
        allow_neuron_splitting=allow_neuron_splitting,
    )
    g_ax = max(axon_granularity, 1)
    g_neu = max(neuron_granularity, 1)
    max_iter = 50
    best_hw: tuple[int, int] | None = None
    best_counts: tuple[int, int] | None = None
    best_total: int | None = None

    def _make_types(h_: int, w_: int) -> list[tuple[int, int]]:
        return _make_two_core_types(
            h_, w_, allow_coalescing, allow_neuron_splitting, g_ax, g_neu,
            max_ax=max_ax, max_neu=max_neu,
        )

    def _count(type1_: tuple, type2_: tuple) -> int:
        return _count_cores_needed_two_types(
            softcores_list, type1_, type2_, safety_margin,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )

    def _pack(type1_: tuple, type2_: tuple, total_: int):
        return _pack_with_two_types(
            softcores_list, type1_, type2_, total_,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )

    grow_h = not (allow_neuron_splitting ^ allow_coalescing)
    grow_w = True

    group_sizes: Dict[int, int] = {}
    for _sc in softcores_list:
        tg = int(_sc.threshold_group_id)
        group_sizes[tg] = group_sizes.get(tg, 0) + 1
    largest_group = max(group_sizes.values()) if group_sizes else 0
    n_groups = len(group_sizes)

    skip_occupancy = allow_neuron_splitting or allow_coalescing

    first_feasible_area: int | None = None
    first_feasible: tuple[tuple[int, int], int] | None = None
    _no_progress_iters = 0
    _last_used_count: int | None = None

    def _area_for(h_: int, w_: int, total_: int) -> int:
        types = _make_types(h_, w_)
        c1_, c2_ = (total_ + 1) // 2, total_ // 2
        return types[0][0] * types[0][1] * c1_ + types[1][0] * types[1][1] * c2_

    for _ in range(max_iter):
        types_spec = _make_types(h, w)
        type1, type2 = types_spec[0], types_spec[1]
        total = _count(type1, type2)
        if skip_occupancy:
            c1, c2 = (total + 1) // 2, total // 2
            best_hw = (h, w)
            best_counts = (c1, c2)
            best_total = total
            break
        feasible, cores_used, counts = _pack(type1, type2, total)
        if feasible and first_feasible is None:
            first_feasible = ((h, w), total)
            first_feasible_area = _area_for(h, w, total)
        if not feasible or counts is None:
            if grow_h:
                h = int(math.ceil(h * 1.25)) if h else 1
            w = int(math.ceil(w * 1.25)) if w else 1
            continue
        if _occupancy_ok(counts):
            c1, c2 = (total + 1) // 2, total // 2
            best_hw = (h, w)
            best_counts = (c1, c2)
            best_total = total
            break
        used_count = len(counts) if counts else 0
        if _last_used_count is not None and used_count >= _last_used_count:
            _no_progress_iters += 1
        else:
            _no_progress_iters = 0
        _last_used_count = used_count
        if first_feasible is not None and _no_progress_iters >= 3:
            (h, w), total = first_feasible
            c1, c2 = (total + 1) // 2, total // 2
            best_hw = (h, w)
            best_counts = (c1, c2)
            best_total = total
            break
        if grow_h:
            h = int(math.ceil(h * 1.25)) if h else 1
        w = int(math.ceil(w * 1.25)) if w else 1

    if best_hw is None or best_counts is None or best_total is None:
        h, w = _min_hw_coverage(
            softcores_list, allow_coalescing, axon_granularity, neuron_granularity,
            allow_neuron_splitting=allow_neuron_splitting,
        )
        types_spec = _make_types(h, w)
        type1, type2 = types_spec[0], types_spec[1]
        best_total = _count(type1, type2)
        best_hw = (h, w)
        best_counts = ((best_total + 1) // 2, best_total // 2)

    h, w = best_hw
    c1, c2 = best_counts
    types_spec = _make_types(h, w)
    type1, type2 = types_spec[0], types_spec[1]

    t1_used, t2_used = _count_per_type_usage(
        softcores_list, type1, type2, c1, c2,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_coalescing=allow_coalescing,
    )
    _unused_alloc = 0
    if t1_used == 0:
        _unused_alloc = c1
    elif t2_used == 0:
        _unused_alloc = c2
    if (
        t1_used is not None
        and t2_used is not None
        and (t1_used == 0 or t2_used == 0)
        and (t1_used + t2_used) > 0
        and _unused_alloc >= 8
    ):
        def _scale(used: int) -> int:
            return max(used + 1, int(math.ceil(used * (1.0 + safety_margin))))
        c1 = _scale(t1_used) if t1_used > 0 else 1
        c2 = _scale(t2_used) if t2_used > 0 else 1

    core_types: List[Dict[str, Any]] = [
        {
            "max_axons": type1[0],
            "max_neurons": type1[1],
            "count": c1,
            "has_bias": hardware_bias,
        },
        {
            "max_axons": type2[0],
            "max_neurons": type2[1],
            "count": c2,
            "has_bias": hardware_bias,
        },
    ]
    best_total = c1 + c2

    rationale_parts = [
        f"{len(softcores)} softcores \u2192 two types {type1[0]}\u00d7{type1[1]} and {type2[0]}\u00d7{type2[1]}, "
        f"{best_total} total cores (>50% with \u22654 softcores)"
    ]
    if allow_coalescing:
        rationale_parts.append("coalescing enabled")
    if allow_neuron_splitting:
        rationale_parts.append("neuron splitting enabled")
    if safety_margin > 0:
        rationale_parts.append(f"{int(safety_margin*100)}% safety margin applied")

    return HardwareSuggestion(
        core_types=core_types,
        total_cores=best_total,
        rationale=". ".join(rationale_parts) + ".",
    )


