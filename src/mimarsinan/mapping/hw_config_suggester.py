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
no softcores):

- **Without coalescing**: types are H×W and W×H (one long, one wide).
- **With coalescing**: types are H×H (square) and W×H with W > H (square + wide).

Dimensions are the **smallest** H and W that:
  1. Cover all softcores (every softcore fits in at least one type).
  2. After packing with a 50/50 pool split, more than half of used cores
     have at least 4 softcores each (skipped when fewer than 2 cores are used).

Search: start from minimal (H, W) for coverage; if the occupancy constraint
fails, increase H and/or W and retry until it passes or an iteration limit
is reached. Pool size per (H, W) is found by binary search plus safety margin.

Important: both the auto-config pass (``api_hw_config_auto``) and the
verification pass (``api_hw_config_verify``) run layout mapping with a
large fixed bound (≥ 4096 axons/neurons) so that both see identical,
naturally-sized softcores.

Typical usage:
    from mimarsinan.mapping.mapping_verifier import verify_soft_core_mapping
    from mimarsinan.mapping.hw_config_suggester import suggest_hardware_config

    result = verify_soft_core_mapping(model_repr, max_axons=4096, max_neurons=4096,
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


def _make_two_core_types(
    h: int,
    w: int,
    allow_coalescing: bool,
    axon_granularity: int,
    neuron_granularity: int,
) -> list[tuple[int, int]]:
    """Return list of (max_axons, max_neurons) for the two core types.

    Without coalescing: (H, W) and (W, H). With coalescing: (H, H) and (W, H), W > H.
    """
    g_ax = max(axon_granularity, 1)
    g_neu = max(neuron_granularity, 1)
    if allow_coalescing:
        if w <= h:
            w = h + 1
        return [(h, h), (w, h)]
    return [(h, w), (w, h)]


def _pack_with_two_types(
    softcores: Sequence[LayoutSoftCoreSpec],
    type1: tuple[int, int],
    type2: tuple[int, int],
    total_count: int,
) -> tuple[bool, int | None, tuple[int, ...] | None]:
    """Pack with two core types (50/50 split). Return (feasible, cores_used, used_core_softcore_counts)."""
    c1, c2 = (total_count + 1) // 2, total_count // 2
    if c1 <= 0 and c2 <= 0:
        return False, None, None
    hw_types = [
        LayoutHardCoreType(max_axons=type1[0], max_neurons=type1[1], count=max(1, c1)),
        LayoutHardCoreType(max_axons=type2[0], max_neurons=type2[1], count=max(1, c2)),
    ]
    result = pack_layout(softcores=list(softcores), core_types=hw_types)
    if not result.feasible:
        return False, None, None
    return True, result.cores_used, result.used_core_softcore_counts


def _min_hw_coverage(
    softcores: Sequence[LayoutSoftCoreSpec],
    allow_coalescing: bool,
    axon_granularity: int,
    neuron_granularity: int,
) -> tuple[int, int]:
    """Return smallest (H, W) that cover all softcores. W > H when coalescing."""
    max_ax, max_neu, min_dim, max_dim = _dimension_bounds(softcores)
    if not softcores:
        return 1, 1
    g = max(axon_granularity, neuron_granularity, 1)
    if allow_coalescing:
        # (H,H) fits (a,b) iff max(a,b) <= H. (W,H) fits (a,b) iff a <= W and b <= H.
        # Ensure square covers all: H >= max_dim. Then W > H for the wide type.
        h = _next_multiple(max_dim, g)
        w = _next_multiple(h + 1, g)
        return h, w
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
) -> int:
    """Minimum total pool size (with 50/50 split) for packing to succeed, plus safety margin."""
    n_sc = len(softcores)
    if not n_sc:
        return 0
    upper = max(n_sc, 1)
    for _ in range(30):
        feasible, _, _ = _pack_with_two_types(softcores, type1, type2, upper)
        if feasible:
            break
        upper *= 2
    else:
        return max(1, int(math.ceil(n_sc * (1.0 + safety_margin))))
    lower = 1
    while lower < upper:
        mid = (lower + upper) // 2
        feasible, _, _ = _pack_with_two_types(softcores, type1, type2, mid)
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
    h, w = _min_hw_coverage(
        softcores_list, allow_coalescing, axon_granularity, neuron_granularity
    )
    g_ax = max(axon_granularity, 1)
    g_neu = max(neuron_granularity, 1)
    max_iter = 50
    best_hw: tuple[int, int] | None = None
    best_counts: tuple[int, int] | None = None
    best_total: int | None = None

    for _ in range(max_iter):
        types_spec = _make_two_core_types(h, w, allow_coalescing, g_ax, g_neu)
        type1, type2 = types_spec[0], types_spec[1]
        total = _count_cores_needed_two_types(softcores_list, type1, type2, safety_margin)
        feasible, cores_used, counts = _pack_with_two_types(
            softcores_list, type1, type2, total
        )
        if not feasible or counts is None:
            h = int(math.ceil(h * 1.25)) if h else 1
            w = int(math.ceil(w * 1.25)) if w else 1
            continue
        if _occupancy_ok(counts):
            c1, c2 = (total + 1) // 2, total // 2
            best_hw = (h, w)
            best_counts = (c1, c2)
            best_total = total
            break
        h = int(math.ceil(h * 1.25)) if h else 1
        w = int(math.ceil(w * 1.25)) if w else 1

    if best_hw is None or best_counts is None or best_total is None:
        h, w = _min_hw_coverage(
            softcores_list, allow_coalescing, axon_granularity, neuron_granularity
        )
        types_spec = _make_two_core_types(h, w, allow_coalescing, g_ax, g_neu)
        type1, type2 = types_spec[0], types_spec[1]
        best_total = _count_cores_needed_two_types(
            softcores_list, type1, type2, safety_margin
        )
        best_hw = (h, w)
        best_counts = ((best_total + 1) // 2, best_total // 2)

    h, w = best_hw
    c1, c2 = best_counts
    types_spec = _make_two_core_types(h, w, allow_coalescing, g_ax, g_neu)
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
        f"{len(softcores)} softcores → two types {type1[0]}×{type1[1]} and {type2[0]}×{type2[1]}, "
        f"{best_total} total cores (>50% with ≥4 softcores)"
    ]
    if allow_coalescing:
        rationale_parts.append("coalescing enabled")
    if safety_margin > 0:
        rationale_parts.append(f"{int(safety_margin*100)}% safety margin applied")

    return HardwareSuggestion(
        core_types=core_types,
        total_cores=best_total,
        rationale=". ".join(rationale_parts) + ".",
    )


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
    )
