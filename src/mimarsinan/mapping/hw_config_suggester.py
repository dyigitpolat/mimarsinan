"""
Greedy hardware configuration suggester.

Given a list of LayoutSoftCoreSpec (already accounting for pruning and threshold
groups via LayoutIRMapping), this module produces a reasonable hardware core
configuration that:
  1. Can fit all softcores.
  2. Minimises total core count and wasted axon/neuron buffer space.
  3. Uses simple heuristics — no optimisation search.

Algorithm
---------
``suggest_hardware_config`` always produces a **single core type** whose
``max_axons`` and ``max_neurons`` equal the maximum across all softcores
(guaranteeing every softcore fits).  The count is found by:

  1. Doubling the pool from ``len(softcores)`` until ``pack_layout`` succeeds.
  2. Binary-searching downward to find the true minimum feasible pool.
  3. Applying a safety margin (default 15 %) on top of that minimum.

This two-phase approach is necessary because the greedy packing algorithm
can produce better results when the available pool is large (more unused
cores to choose from), making ``cores_used`` from a large pool an
**optimistic underestimate** of the cores actually needed at that exact count.

Important: both the auto-config pass (``api_hw_config_auto``) and the
verification pass (``api_hw_config_verify``) run layout mapping with a
large fixed bound (≥ 4096 axons/neurons) so that both see identical,
naturally-sized softcores.  The suggested ``max_axons`` / ``max_neurons``
are then the true maximum softcore dimensions, not an artificially
tiled view.

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
from dataclasses import dataclass, field
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


def _pack_feasible(
    softcores: Sequence[LayoutSoftCoreSpec], max_axons: int, max_neurons: int, count: int
) -> bool:
    hw_types = [LayoutHardCoreType(max_axons=max_axons, max_neurons=max_neurons, count=count)]
    return pack_layout(softcores=list(softcores), core_types=hw_types).feasible


def _count_cores_needed(
    softcores: Sequence[LayoutSoftCoreSpec],
    max_axons: int,
    max_neurons: int,
    safety_margin: float = 0.15,
) -> int:
    """Find the minimum pool where greedy packing succeeds, then apply safety margin.

    We binary-search for the true minimum rather than trusting ``cores_used``
    from a large pool, because the greedy algorithm can pack fewer cores when
    the pool is large (more unused cores to choose from), making the
    ``cores_used`` count optimistic for smaller pools.
    """
    if not softcores:
        return 0

    n_sc = len(softcores)

    # Find an upper bound where packing definitely succeeds (double from n_sc).
    upper = max(n_sc, 1)
    for _ in range(30):
        if _pack_feasible(softcores, max_axons, max_neurons, upper):
            break
        upper *= 2
    else:
        return int(math.ceil(n_sc * (1.0 + safety_margin)))

    # Binary-search for the true minimum feasible pool.
    lower = 1
    while lower < upper:
        mid = (lower + upper) // 2
        if _pack_feasible(softcores, max_axons, max_neurons, mid):
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
    """Produce a simple, reasonable hardware configuration for the given softcores.

    This is *not* an optimiser — it uses straightforward heuristics:

    1. Attempt to detect two distinct size classes (large vs small cores).
    2. For each class, the core dimensions are the *maximum* of that class
       (optionally rounded up to *granularity* multiples for tidy hardware).
    3. Simulate greedy packing to find how many cores are required.
    4. Apply a ``safety_margin`` (default 15%) on top.

    Parameters
    ----------
    softcores:
        Pre-computed ``LayoutSoftCoreSpec`` list (already accounts for pruning
        and threshold groups via ``LayoutIRMapping.collect_layout_softcores``).
    allow_coalescing:
        Passed through to the rationale string; coalescing itself is handled
        upstream by ``IRMapping``.
    hardware_bias:
        Whether cores use a dedicated hardware bias register (vs an always-on
        axon row).  Affects ``has_bias`` in the returned core type dicts.
    axon_granularity:
        Round max_axons up to a multiple of this value (default 1 = no rounding).
    neuron_granularity:
        Round max_neurons up to a multiple of this value (default 1 = no rounding).
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

    # Single core type covering all softcores — guarantees every softcore fits.
    max_ax = _next_multiple(max(sc.input_count for sc in softcores), axon_granularity)
    max_neu = _next_multiple(max(sc.output_count for sc in softcores), neuron_granularity)
    count = _count_cores_needed(list(softcores), max_ax, max_neu, safety_margin)

    core_types: List[Dict[str, Any]] = [{
        "max_axons": max_ax,
        "max_neurons": max_neu,
        "count": count,
        "has_bias": hardware_bias,
    }]

    rationale_parts = [
        f"{len(softcores)} softcores → {max_ax}×{max_neu} core type, {count} instances"
    ]
    if allow_coalescing:
        rationale_parts.append("coalescing enabled")
    if safety_margin > 0:
        rationale_parts.append(f"{int(safety_margin*100)}% safety margin applied")

    return HardwareSuggestion(
        core_types=core_types,
        total_cores=count,
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
