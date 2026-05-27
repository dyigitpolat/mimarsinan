"""Greedy two-type hardware core configuration suggester for layout softcores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)


@dataclass
class HardwareSuggestion:
    """Suggested ``platform_constraints``-ready core_types list and metadata."""

    core_types: List[Dict[str, Any]]
    total_cores: int
    rationale: str
    num_passes: int = 1
    estimated_latency_multiplier: float = 1.0


def _next_multiple(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple* (>= 1)."""
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def _dimension_bounds(softcores: Sequence[LayoutSoftCoreSpec]) -> tuple[int, int, int, int]:
    """Return (max_ax, max_neu, min_dim, max_dim) for softcores."""
    if not softcores:
        return 0, 0, 0, 0
    max_ax = max(sc.input_count for sc in softcores)
    max_neu = max(sc.output_count for sc in softcores)
    min_dim = max(min(sc.input_count, sc.output_count) for sc in softcores)
    max_dim = max(max(sc.input_count, sc.output_count) for sc in softcores)
    return max_ax, max_neu, min_dim, max_dim


def _median(values: list) -> int:
    """Median of a sorted list (0 for empty)."""
    return values[len(values) // 2] if values else 0


def _floor20(max_val: int) -> int:
    """20% of max_val, rounded up — the minimum split-threshold dimension."""
    return int(math.ceil(0.2 * max_val))


def _make_two_core_types(
    h: int,
    w: int,
    allow_coalescing: bool,
    allow_neuron_splitting: bool,
    axon_granularity: int,
    neuron_granularity: int,
    *,
    max_ax: int = 0,
    max_neu: int = 0,
) -> list[tuple[int, int]]:
    """Return two (max_axons, max_neurons) type specs from seed dimensions and feature flags."""
    g_ax = max(axon_granularity, 1)
    g_neu = max(neuron_granularity, 1)

    if allow_neuron_splitting and allow_coalescing:
        a1 = _next_multiple(w, g_ax)
        n1 = _next_multiple(h, g_neu)
        a2 = _next_multiple(w * 2, g_ax)
        n2 = _next_multiple(h * 2, g_neu)
        return [(a1, n1), (a2, n2)]

    if allow_neuron_splitting:
        n1 = _next_multiple(w, g_neu)
        n2_raw = w * 2 if max_neu == 0 else min(w * 2, max_neu)
        n2 = _next_multiple(n2_raw, g_neu)
        if n2 <= n1:
            n2 = _next_multiple(n1 + 1, g_neu)
        return [(h, n1), (h, n2)]

    if allow_coalescing:
        a1 = _next_multiple(w, g_ax)
        a2 = _next_multiple(max(w * 2, max_ax) if max_ax > 0 else w * 2, g_ax)
        if a2 <= a1:
            a2 = _next_multiple(a1 + 1, g_ax)
        return [(a1, h), (a2, h)]

    return [(h, w), (w, h)]


def _pack_with_two_types(
    softcores: Sequence[LayoutSoftCoreSpec],
    type1: tuple[int, int],
    type2: tuple[int, int],
    total_count: int,
    *,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> tuple[bool, int | None, tuple[int, ...] | None]:
    """Pack with two core types (50/50 split). Return (feasible, cores_used, used_core_softcore_counts)."""
    c1, c2 = (total_count + 1) // 2, total_count // 2
    if c1 <= 0 and c2 <= 0:
        return False, None, None
    hw_types = [
        LayoutHardCoreType(max_axons=type1[0], max_neurons=type1[1], count=max(1, c1)),
        LayoutHardCoreType(max_axons=type2[0], max_neurons=type2[1], count=max(1, c2)),
    ]
    result = pack_layout(
        softcores=list(softcores),
        core_types=hw_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_coalescing=allow_coalescing,
    )
    if not result.feasible:
        return False, None, None
    return True, result.cores_used, result.used_core_softcore_counts


def _min_hw_coverage(
    softcores: Sequence[LayoutSoftCoreSpec],
    allow_coalescing: bool,
    axon_granularity: int,
    neuron_granularity: int,
    *,
    allow_neuron_splitting: bool = False,
) -> tuple[int, int]:
    """Return (h, w) seed dimensions for ``_make_two_core_types``."""
    max_ax, max_neu, min_dim, max_dim = _dimension_bounds(softcores)
    if not softcores:
        return 1, 1
    g = max(axon_granularity, neuron_granularity, 1)

    sorted_ax = sorted(sc.input_count for sc in softcores)
    sorted_neu = sorted(sc.output_count for sc in softcores)

    if allow_neuron_splitting and allow_coalescing:
        h = _next_multiple(max(_median(sorted_neu) // 2, _floor20(max_neu)), g)
        w = _next_multiple(max(_median(sorted_ax) // 2, _floor20(max_ax)), g)
        return max(h, 1), max(w, 1)

    if allow_neuron_splitting:
        h = _next_multiple(max_ax, g)
        w = _next_multiple(max(_median(sorted_neu) // 2, _floor20(max_neu)), g)
        return h, max(w, 1)

    if allow_coalescing:
        h = _next_multiple(max_neu, g)
        w = _next_multiple(max(_median(sorted_ax) // 2, _floor20(max_ax)), g)
        return max(h, 1), max(w, 1)

    h = _next_multiple(max_dim, g)
    w = _next_multiple(min_dim, g)
    return h, w


def _occupancy_ok(counts: tuple[int, ...] | None, min_frac: float = 0.5, min_per_core: int = 4) -> bool:
    """True if more than min_frac of used cores have at least min_per_core softcores."""
    if not counts or len(counts) < 2:
        return True
    n_ok = sum(1 for c in counts if c >= min_per_core)
    return n_ok > len(counts) * min_frac


def _count_per_type_usage(
    softcores: Sequence[LayoutSoftCoreSpec],
    type1: tuple[int, int],
    type2: tuple[int, int],
    c1_hint: int,
    c2_hint: int,
    *,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> tuple[int | None, int | None]:
    """Pack with a large pool; return per-type used core counts, or (None, None)."""
    pool1 = max(c1_hint, len(softcores))
    pool2 = max(c2_hint, len(softcores))
    hw_types = [
        LayoutHardCoreType(max_axons=type1[0], max_neurons=type1[1], count=pool1),
        LayoutHardCoreType(max_axons=type2[0], max_neurons=type2[1], count=pool2),
    ]
    result = pack_layout(
        softcores=list(softcores),
        core_types=hw_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_coalescing=allow_coalescing,
    )
    if not result.feasible or result.used_core_snapshots is None:
        return None, None
    t1_used = 0
    t2_used = 0
    for snap in result.used_core_snapshots:
        if (snap.axons_per_core, snap.neurons_per_core) == (type1[0], type1[1]):
            t1_used += 1
        elif (snap.axons_per_core, snap.neurons_per_core) == (type2[0], type2[1]):
            t2_used += 1
    return t1_used, t2_used


def _count_cores_needed_two_types(
    softcores: Sequence[LayoutSoftCoreSpec],
    type1: tuple[int, int],
    type2: tuple[int, int],
    safety_margin: float,
    *,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> int:
    """Minimum two-type pool size for feasible packing, plus safety margin."""
    n_sc = len(softcores)
    if not n_sc:
        return 0
    upper = max(n_sc, 1)
    for _ in range(30):
        feasible, _, _ = _pack_with_two_types(
            softcores, type1, type2, upper,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )
        if feasible:
            break
        upper *= 2
    else:
        return max(1, int(math.ceil(n_sc * (1.0 + safety_margin))))

    lower = 1
    for _ in range(6):
        if lower >= upper:
            break
        mid = (lower + upper) // 2
        feasible, _, _ = _pack_with_two_types(
            softcores, type1, type2, mid,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )
        if feasible:
            upper = mid
        else:
            lower = mid + 1

    padded = int(math.ceil(upper * (1.0 + safety_margin)))
    return max(padded, upper + 1)


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
    first_feasible: tuple[tuple[int, int], int] | None = None  # ((h, w), total)
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


def suggest_hardware_config_scheduled(
    softcores: Sequence[LayoutSoftCoreSpec],
    *,
    max_passes: int = 8,
    latency_weight: float = 1.0,
    allow_coalescing: bool = False,
    hardware_bias: bool = True,
    axon_granularity: int = 1,
    neuron_granularity: int = 1,
    safety_margin: float = 0.15,
    allow_neuron_splitting: bool = False,
) -> HardwareSuggestion:
    """Suggest hardware config minimizing core area × pass count (scheduled mapping)."""
    from mimarsinan.mapping.schedule_partitioner import estimate_passes_for_layout

    if not softcores:
        return HardwareSuggestion(
            core_types=[], total_cores=0,
            rationale="No softcores — nothing to map.",
        )

    max_passes = max(1, int(max_passes))
    softcores_list = list(softcores)

    common_kwargs = dict(
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
        axon_granularity=axon_granularity,
        neuron_granularity=neuron_granularity,
        safety_margin=safety_margin,
        allow_neuron_splitting=allow_neuron_splitting,
    )

    single = suggest_hardware_config(softcores_list, **common_kwargs)

    def _core_area(suggestion: HardwareSuggestion) -> float:
        return sum(
            ct["max_axons"] * ct["max_neurons"] * ct["count"]
            for ct in suggestion.core_types
        )

    best = single
    best_cost = _core_area(single) * (1.0 ** latency_weight)
    best_passes = 1

    if single.core_types:
        ref_ax = max(ct["max_axons"] for ct in single.core_types)
        ref_neu = max(ct["max_neurons"] for ct in single.core_types)
    else:
        max_ax, max_neu, _, _ = _dimension_bounds(softcores_list)
        ref_ax, ref_neu = max_ax or 256, max_neu or 256

    seen_pass_counts: set[int] = {1}
    budgets_to_try = set()
    for divisor in range(2, max_passes + 1):
        budgets_to_try.add(max(1, single.total_cores // divisor))
    for b in [1, 2, 4, 8, 16, 32]:
        if b < single.total_cores:
            budgets_to_try.add(b)

    for budget in sorted(budgets_to_try, reverse=True):
        est_passes, pass_lists = estimate_passes_for_layout(
            softcores_list, budget,
            max_hw_axons=ref_ax, max_hw_neurons=ref_neu,
            allow_coalescing=allow_coalescing, allow_splitting=allow_neuron_splitting,
        )
        if est_passes <= 1 or est_passes in seen_pass_counts or est_passes > max_passes:
            continue
        seen_pass_counts.add(est_passes)

        largest_pass = max(pass_lists, key=len)
        try:
            suggestion = suggest_hardware_config(largest_pass, **common_kwargs)
        except Exception:
            continue
        if not suggestion.core_types:
            continue

        cost = _core_area(suggestion) * (est_passes ** latency_weight)
        if cost < best_cost:
            best = suggestion
            best_cost = cost
            best_passes = est_passes

    rationale_parts = [best.rationale.rstrip(".")]
    if best_passes > 1:
        rationale_parts.append(f"{best_passes} schedule passes (cores reused)")
        rationale_parts.append(f"latency ~{best_passes}x single-pass")
    best.rationale = ". ".join(rationale_parts) + "."
    best.num_passes = best_passes
    best.estimated_latency_multiplier = float(best_passes)
    return best


def suggest_hardware_config_for_model(
    model_repr,
    *,
    max_axons: int,
    max_neurons: int,
    allow_coalescing: bool = False,
    hardware_bias: bool = True,
    axon_granularity: int = 1,
    neuron_granularity: int = 1,
    safety_margin: float = 0.15,
    allow_neuron_splitting: bool = False,
) -> HardwareSuggestion:
    """Convenience wrapper: run layout mapping then suggest hardware config."""
    from mimarsinan.mapping.verification.mapping_verifier import verify_soft_core_mapping

    result = verify_soft_core_mapping(
        model_repr,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    )

    if not result.feasible:
        return HardwareSuggestion(
            core_types=[],
            total_cores=0,
            rationale=f"Layout mapping failed: {result.error}",
        )

    return suggest_hardware_config(
        result.softcores,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
        axon_granularity=axon_granularity,
        neuron_granularity=neuron_granularity,
        safety_margin=safety_margin,
        allow_neuron_splitting=allow_neuron_splitting,
    )
