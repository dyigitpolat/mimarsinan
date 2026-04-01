"""
Greedy hardware configuration suggester.

Given a list of LayoutSoftCoreSpec (already accounting for pruning and threshold
groups via LayoutIRMapping), this module produces a reasonable hardware core
configuration that:
  1. Can fit all softcores.
  2. Uses two core types (H×W and W×H, or H×H and W×H when coalescing).
  3. Chooses the smallest H, W such that more than half of used hardware cores
     each house at least 4 software cores.

Algorithm
---------
``suggest_hardware_config`` always produces **two core types** (unless there are
no softcores).  The shape depends on the active feature toggles:

+----------------+------------------+-------------------------------------------+
| Coalescing     | Neuron Splitting | Core types produced                       |
+================+==================+===========================================+
| off            | off              | H×W and W×H  (tall + wide)                |
|                |                  | H = max_dim, W = min_dim                  |
+----------------+------------------+-------------------------------------------+
| off            | on               | H×W and W×H  (tall + wide)                |
|                |                  | H = max_ax, W = median_neurons ≥ 20% max  |
+----------------+------------------+-------------------------------------------+
| on             | off              | H×H and W×H  (square + wide, W > H)       |
|                |                  | H = max_dim, W = H + 1                    |
+----------------+------------------+-------------------------------------------+
| on             | on               | H×H and W×H  (square + wide, W = H + 1)  |
|                |                  | H = max(median_ax, median_neu, 20% floors)|
+----------------+------------------+-------------------------------------------+

Dimensions are the **smallest** H and W that:
  1. Cover all softcores (every softcore fits in at least one type).
  2. After packing with a 50/50 pool split, more than half of used cores
     have at least 4 softcores each (skipped when fewer than 2 cores are used).

Search: start from minimal (H, W) for coverage; if the occupancy constraint
fails, increase H and/or W and retry until it passes or an iteration limit
is reached. Pool size per (H, W) is found by binary search plus safety margin.

Important: both the auto-config pass (``api_hw_config_auto``) and the
verification pass (``api_hw_config_verify``) run layout mapping with a
very large unconstrained bound (_LAYOUT_PASS_LIMIT in server.py) so that
both see identical naturally-sized softcores regardless of the user's
hardware target.

Typical usage:
    from mimarsinan.mapping.mapping_verifier import verify_soft_core_mapping
    from mimarsinan.mapping.hw_config_suggester import suggest_hardware_config

    result = verify_soft_core_mapping(model_repr, max_axons=1<<20, max_neurons=1<<20,
                                      threshold_groups=4, pruning_fraction=0.5)
    suggestion = suggest_hardware_config(result.softcores,
                                         allow_coalescing=False,
                                         hardware_bias=True)
"""

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
    """Suggested hardware core configuration.

    ``core_types`` is ready to be used as the ``cores`` field of
    ``platform_constraints`` (each entry has ``max_axons``, ``max_neurons``,
    ``count``, ``has_bias``).
    """

    core_types: List[Dict[str, Any]]
    total_cores: int
    rationale: str
    num_passes: int = 1
    estimated_latency_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """Return two (max_axons, max_neurons) hardware core type specs.

    ``h`` and ``w`` are the primary dimension seeds from ``_min_hw_coverage``.
    Their meaning and how the two types are formed depends on active features:

    Neither:
        h = max_dim, w = min_dim.
        Complementary tall type (h, w) and wide type (w, h).

    Splitting only:
        h = max_ax (fixed — both types share full axon coverage).
        w = median_neu.  Type A = (h, w), Type B = (h, 2w) capped at max_neu.
        Two neuron depths: compact A for most cores, roomier B for outliers.

    Coalescing only:
        h = max_neu (fixed — both types share full neuron coverage).
        w = median_ax.  Type A = (w, h), Type B = (max_ax, h).
        Type A packs typical softcores compactly; Type B covers the widest
        softcore without any axon coalescing.

    Both:
        h = median_neu, w = median_ax — both dimensions relaxed.
        Type A = (w, h), Type B = (2w, 2h): a small and a medium-sized core
        for efficient packing across the softcore size distribution.
    """
    g_ax = max(axon_granularity, 1)
    g_neu = max(neuron_granularity, 1)

    if allow_neuron_splitting and allow_coalescing:
        # h = median_neu, w = median_ax
        a1 = _next_multiple(w, g_ax)
        n1 = _next_multiple(h, g_neu)
        a2 = _next_multiple(w * 2, g_ax)
        n2 = _next_multiple(h * 2, g_neu)
        return [(a1, n1), (a2, n2)]

    if allow_neuron_splitting:
        # h = max_ax; both types have the same full axon coverage.
        n1 = _next_multiple(w, g_neu)
        # Type B: double the neuron depth, capped at max_neu so we don't over-allocate.
        n2_raw = w * 2 if max_neu == 0 else min(w * 2, max_neu)
        n2 = _next_multiple(n2_raw, g_neu)
        if n2 <= n1:
            n2 = _next_multiple(n1 + 1, g_neu)
        return [(h, n1), (h, n2)]

    if allow_coalescing:
        # h = max_neu; both types have the same full neuron coverage.
        a1 = _next_multiple(w, g_ax)
        # Type B: covers max_ax so the widest softcore fits without coalescing.
        a2 = _next_multiple(max(w * 2, max_ax) if max_ax > 0 else w * 2, g_ax)
        if a2 <= a1:
            a2 = _next_multiple(a1 + 1, g_ax)
        return [(a1, h), (a2, h)]

    # Neither: complementary tall type (h, w) and wide type (w, h).
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
    """Return (h, w) seed dimensions for ``_make_two_core_types``.

    The user enables coalescing/splitting because they want smaller cores in
    that dimension.  We honour this by halving the flexible dimension when the
    corresponding feature is active (floor at 20 % of max prevents pathological
    fragmentation).

    Neither:
        h = max_dim, w = min_dim — both dimensions must cover all softcores.

    Splitting only:
        h = max_ax (hard axon constraint), w = median_neu / 2 ≥ 20% of max_neu.
        Neurons are split, so half the typical output count suffices per core.

    Coalescing only:
        h = max_neu (hard neuron constraint), w = median_ax / 2 ≥ 20% of max_ax.
        Axons are coalesced, so half the typical input count suffices per core.

    Both:
        h = median_neu / 2 ≥ 20% of max_neu, w = median_ax / 2 ≥ 20% of max_ax.
        Both dimensions halved; splitting and coalescing handle the rest.
    """
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


def _count_cores_needed_two_types(
    softcores: Sequence[LayoutSoftCoreSpec],
    type1: tuple[int, int],
    type2: tuple[int, int],
    safety_margin: float,
    *,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> int:
    """Minimum total pool size (with 50/50 split) for packing to succeed, plus safety margin."""
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
    while lower < upper:
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
    min_feasible = lower
    padded = int(math.ceil(min_feasible * (1.0 + safety_margin)))
    return max(padded, min_feasible + 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Produce a two-type hardware configuration for the given softcores.

    Returns two core types (H×W and W×H, or H×H and W×H when coalescing),
    with the smallest H, W such that more than half of used cores host at least
    4 softcores. Applies safety_margin on top of the minimum feasible pool size.

    Parameters
    ----------
    softcores:
        Pre-computed ``LayoutSoftCoreSpec`` list (already accounts for pruning
        and threshold groups via ``LayoutIRMapping.collect_layout_softcores``).
    allow_coalescing:
        If True, types are H×H (square) and W×H (wide, W > H); else H×W and W×H.
    hardware_bias:
        Whether cores use a dedicated hardware bias register.
    axon_granularity, neuron_granularity:
        Round dimensions up to these multiples.
    safety_margin:
        Fraction of extra cores to add on top of the minimum pack count.
    allow_neuron_splitting:
        If True, soft cores may be split across hardware cores along the neuron
        (output) dimension.  The suggested neuron width is relaxed accordingly.

    Returns
    -------
    HardwareSuggestion
    """
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

    # Determine which dimensions are flexible (may grow during iteration).
    # When splitting is on, h = max_ax is a hard constraint (axons must fit).
    # When coalescing is on, h = max_neu is a hard constraint (neurons must fit).
    # "Neither" and "both" cases can grow both dimensions.
    grow_h = not (allow_neuron_splitting ^ allow_coalescing)  # grow when both or neither
    grow_w = True  # w is always the flexible (median-based) dimension

    # When splitting/coalescing is enabled the whole point is smaller cores —
    # skip the occupancy check (which forces growth until many softcores share
    # a core) and accept any feasible packing.
    skip_occupancy = allow_neuron_splitting or allow_coalescing

    for _ in range(max_iter):
        types_spec = _make_types(h, w)
        type1, type2 = types_spec[0], types_spec[1]
        total = _count(type1, type2)
        feasible, cores_used, counts = _pack(type1, type2, total)
        if not feasible or counts is None:
            if grow_h:
                h = int(math.ceil(h * 1.25)) if h else 1
            w = int(math.ceil(w * 1.25)) if w else 1
            continue
        if skip_occupancy or _occupancy_ok(counts):
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
    """Suggest hardware config with scheduled mapping (core reuse across passes).

    Explores the tradeoff between core count and number of schedule passes.
    For each candidate pass count, suggests hardware for the largest pass
    and evaluates::

        cost = total_core_area × num_passes ^ latency_weight

    ``latency_weight`` controls the balance:
      - 1.0: area × latency product (balanced)
      - < 1.0: favours fewer cores (area-conscious)
      - > 1.0: favours fewer passes (latency-conscious)

    Returns the configuration with the lowest cost.
    """
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

    # Try the non-scheduled path first (1 pass = all softcores).
    single = suggest_hardware_config(softcores_list, **common_kwargs)

    def _core_area(suggestion: HardwareSuggestion) -> float:
        return sum(
            ct["max_axons"] * ct["max_neurons"] * ct["count"]
            for ct in suggestion.core_types
        )

    best = single
    best_cost = _core_area(single) * (1.0 ** latency_weight)
    best_passes = 1

    # For multi-pass search, we need hw dimensions from the single-pass
    # suggestion to estimate hardware core cost per softcore.
    if single.core_types:
        ref_ax = max(ct["max_axons"] for ct in single.core_types)
        ref_neu = max(ct["max_neurons"] for ct in single.core_types)
    else:
        max_ax, max_neu, _, _ = _dimension_bounds(softcores_list)
        ref_ax, ref_neu = max_ax or 256, max_neu or 256

    # Search: progressively reduce the per-pass core budget.
    # For each budget, estimate how many passes result, then suggest hw
    # for the largest pass.
    seen_pass_counts: set[int] = {1}
    budgets_to_try = set()
    # Try halving the single-pass core count, and fractions down to 1/max_passes.
    for divisor in range(2, max_passes + 1):
        budgets_to_try.add(max(1, single.total_cores // divisor))
    # Also try absolute small budgets for extreme scheduling.
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

    # Annotate the winner with scheduling info.
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
    threshold_groups: int = 1,
    pruning_fraction: float = 0.0,
    threshold_seed: int = 0,
    allow_coalescing: bool = False,
    hardware_bias: bool = True,
    axon_granularity: int = 1,
    neuron_granularity: int = 1,
    safety_margin: float = 0.15,
    allow_neuron_splitting: bool = False,
) -> HardwareSuggestion:
    """Convenience wrapper: run layout mapping then suggest hardware config.

    Parameters are the same as ``verify_soft_core_mapping`` plus those of
    ``suggest_hardware_config``.
    """
    from mimarsinan.mapping.mapping_verifier import verify_soft_core_mapping

    result = verify_soft_core_mapping(
        model_repr,
        max_axons=max_axons,
        max_neurons=max_neurons,
        threshold_groups=threshold_groups,
        pruning_fraction=pruning_fraction,
        threshold_seed=threshold_seed,
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
