"""
Schedule partitioner for splitting neural segments into multiple passes.

When the hardware has fewer cores than a neural segment requires, this module
partitions the segment's cores into ordered passes that can each fit on
the available hardware.  Passes execute sequentially, reusing the same physical
cores (reprogrammed between passes).

The partitioning respects one hard invariant:

1. **Latency ordering** — cores in pass *k* depend only on cores in passes
   < *k* (or external inputs).  This is guaranteed by assigning latency groups
   in increasing order.

Within a single latency group, cores are independent (by definition of "same
depth in the dependency DAG"), so any partition of the group is valid.

Note: coalescing groups do NOT need to stay together — the state buffer in
``SpikingHybridCoreFlow`` handles inter-pass data flow for partial-sum
fragments.  Each core is an individual scheduling unit.

Hardware core cost estimation
-----------------------------
A single softcore may require multiple hardware cores after coalescing expansion
(wide axons) or neuron splitting (many neurons).  The unified partitioner
accounts for this by computing:

    hw_cost(sc) = ceil(axons / max_hw_axons) * ceil(neurons / max_hw_neurons)

where the ceil factors are 1 when the corresponding feature is disabled.

Budget helpers
--------------
``effective_core_budget`` applies a 20% scarcity discount when heterogeneous
core types are present.  Both the wizard verifier and the hybrid mapping builder
call this to ensure identical budgets.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_latency import IRLatency
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Budget helper — single source of truth for both wizard and runtime
# ---------------------------------------------------------------------------

def effective_core_budget(cores_config: Sequence[dict]) -> int:
    """Compute the effective hardware-core budget for schedule partitioning.

    When multiple core types exist, the budget is reduced by 20% to account
    for per-type scarcity (a core needing e.g. 64 neurons can only use types
    that offer >= 64 neurons, even if smaller types have spare capacity).
    """
    total = sum(int(ct["count"]) for ct in cores_config)
    n_types = len(cores_config)
    return int(total * 0.8) if n_types > 1 else total


# ---------------------------------------------------------------------------
# Hardware core cost helpers
# ---------------------------------------------------------------------------

def _hw_cost(
    item: T,
    max_hw_axons: int,
    max_hw_neurons: int,
    allow_coalescing: bool,
    allow_splitting: bool,
) -> int:
    """Estimate how many hardware cores an item will consume.

    Works with any object that has ``get_input_count()`` and
    ``get_output_count()`` — both ``NeuralCore`` and ``LayoutSoftCoreSpec``
    implement this interface.
    """
    ax = item.get_input_count()
    neu = item.get_output_count()
    ax_f = math.ceil(ax / max_hw_axons) if allow_coalescing and max_hw_axons > 0 else 1
    neu_f = math.ceil(neu / max_hw_neurons) if allow_splitting and max_hw_neurons > 0 else 1
    return max(ax_f * neu_f, 1)


def _hw_cost_layout(
    sc: LayoutSoftCoreSpec,
    max_hw_axons: int,
    max_hw_neurons: int,
    allow_coalescing: bool,
    allow_splitting: bool,
) -> int:
    """Estimate how many hardware cores a LayoutSoftCoreSpec will consume.

    Thin wrapper kept for backward-compatible test imports.
    """
    return _hw_cost(sc, max_hw_axons, max_hw_neurons, allow_coalescing, allow_splitting)


# ---------------------------------------------------------------------------
# Unified generic partitioner
# ---------------------------------------------------------------------------

def _partition_with_latencies(
    items: Sequence[T],
    latency_getter: Callable[[T], int],
    max_cores_per_pass: int,
    cost_fn: Callable[[T], int],
) -> list[list[T]]:
    """Core partitioning algorithm shared by NeuralCore and LayoutSoftCoreSpec paths.

    Groups *items* by latency (via *latency_getter*), then greedily assigns
    latency groups to passes in increasing order.  When a single latency group
    exceeds *max_cores_per_pass*, it is split using first-fit-decreasing
    bin-packing (``_bin_pack_items_costed``).

    Returns a list of passes, each a list of items.
    """
    if not items:
        return []

    if max_cores_per_pass <= 0:
        raise ValueError("max_cores_per_pass must be positive")

    lat_groups: dict[int, list[T]] = defaultdict(list)
    for item in items:
        lat_groups[latency_getter(item)].append(item)

    passes: list[list[T]] = []
    current_pass: list[T] = []
    current_cost = 0

    for lat in sorted(lat_groups.keys()):
        group = lat_groups[lat]
        units = [[g] for g in group]
        group_cost = sum(cost_fn(g) for g in group)

        if group_cost > max_cores_per_pass:
            if current_pass:
                passes.append(current_pass)
                current_pass = []
                current_cost = 0
            sub_passes = _bin_pack_items_costed(units, max_cores_per_pass, cost_fn)
            passes.extend(sub_passes)
        elif current_cost + group_cost > max_cores_per_pass:
            if current_pass:
                passes.append(current_pass)
            current_pass = list(group)
            current_cost = group_cost
        else:
            current_pass.extend(group)
            current_cost += group_cost

    if current_pass:
        passes.append(current_pass)

    return passes


# ---------------------------------------------------------------------------
# NeuralCore partitioner (used at hard-core mapping build time)
# ---------------------------------------------------------------------------

def partition_segment_into_passes(
    cores: list[NeuralCore],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
) -> list[list[NeuralCore]]:
    """Partition a neural segment's cores into sequential passes.

    Each pass's estimated hardware core cost does not exceed
    *max_cores_per_pass*.

    Parameters
    ----------
    cores:
        All NeuralCores belonging to one neural segment.
    max_cores_per_pass:
        Total hardware cores available (sum of all core type counts).
    max_hw_axons, max_hw_neurons:
        Largest hardware core dimensions (used for cost estimation).
        When 0, each softcore is counted as 1 hardware core.
    allow_coalescing, allow_splitting:
        Whether axon coalescing / neuron splitting are enabled.
    """
    if not cores:
        return []

    def _cost(c: NeuralCore) -> int:
        if max_hw_axons > 0 and max_hw_neurons > 0:
            return _hw_cost(c, max_hw_axons, max_hw_neurons, allow_coalescing, allow_splitting)
        return 1

    latency_map = _compute_core_latencies(cores)

    return _partition_with_latencies(
        items=cores,
        latency_getter=lambda c: latency_map[c.id],
        max_cores_per_pass=max_cores_per_pass,
        cost_fn=_cost,
    )


# ---------------------------------------------------------------------------
# Layout-level estimation (operates on LayoutSoftCoreSpec, no actual weights)
# ---------------------------------------------------------------------------

def estimate_passes_for_layout(
    softcores: Sequence[LayoutSoftCoreSpec],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
) -> Tuple[int, List[List[LayoutSoftCoreSpec]]]:
    """Estimate the number of schedule passes from layout soft-core specs.

    Groups by ``segment_id`` then delegates to the **same** partitioning
    algorithm used by :func:`partition_segment_into_passes`, ensuring
    identical pass counts for identical shapes.

    Returns ``(num_passes, pass_lists)`` where *pass_lists* is flattened
    across all segments (segment 0 passes first, then segment 1, etc.).
    """
    if not softcores or max_cores_per_pass <= 0:
        return (0, []) if not softcores else (1, [list(softcores)])

    def _cost(sc: LayoutSoftCoreSpec) -> int:
        if max_hw_axons > 0 and max_hw_neurons > 0:
            return _hw_cost(sc, max_hw_axons, max_hw_neurons, allow_coalescing, allow_splitting)
        return 1

    by_segment: dict[int, list[LayoutSoftCoreSpec]] = defaultdict(list)
    for sc in softcores:
        seg = sc.segment_id if sc.segment_id is not None else 0
        by_segment[seg].append(sc)

    all_passes: list[list[LayoutSoftCoreSpec]] = []

    for seg_id in sorted(by_segment.keys()):
        seg_cores = by_segment[seg_id]
        seg_passes = _partition_with_latencies(
            items=seg_cores,
            latency_getter=lambda sc: int(sc.latency_tag) if sc.latency_tag is not None else 0,
            max_cores_per_pass=max_cores_per_pass,
            cost_fn=_cost,
        )
        all_passes.extend(seg_passes)

    num_passes = len(all_passes)
    return num_passes, all_passes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_core_latencies(cores: list[NeuralCore]) -> dict[int, int]:
    """Compute per-core latency using IRLatency on a minimal sub-graph.

    Uses save/restore to avoid mutating the original cores' latency attributes.
    """
    import numpy as np
    saved = {c.id: c.latency for c in cores}
    graph = IRGraph(
        nodes=list(cores),
        output_sources=np.array([], dtype=object),
        weight_banks={},
    )
    ir_lat = IRLatency(graph)
    ir_lat.calculate()
    result = {c.id: (c.latency if c.latency is not None else 0) for c in cores}
    for c in cores:
        c.latency = saved[c.id]
    return result


def _build_atomic_units(cores: list[NeuralCore]) -> list[list[NeuralCore]]:
    """Group cores into atomic scheduling units.

    Cores sharing the same ``coalescing_group_id`` form one unit (they must
    not be split).  Standalone cores (no coalescing group) are individual
    units.
    """
    coalescing_groups: dict[int, list[NeuralCore]] = defaultdict(list)
    standalone: list[list[NeuralCore]] = []

    for c in cores:
        gid = getattr(c, "coalescing_group_id", None)
        if gid is not None:
            coalescing_groups[gid].append(c)
        else:
            standalone.append([c])

    units = list(coalescing_groups.values()) + standalone
    # Sort largest-first for better bin-packing.
    units.sort(key=lambda u: len(u), reverse=True)
    return units


def _bin_pack_items_costed(
    units: list[list[T]],
    max_per_pass: int,
    cost_fn: Callable[[T], int],
) -> list[list[T]]:
    """First-fit-decreasing bin-pack of atomic units into passes using cost function."""
    passes: list[list[T]] = []
    pass_costs: list[int] = []

    for unit in units:
        size = sum(cost_fn(c) for c in unit)
        if size > max_per_pass:
            passes.append(list(unit))
            pass_costs.append(size)
            continue

        placed = False
        for i, cost in enumerate(pass_costs):
            if cost + size <= max_per_pass:
                passes[i].extend(unit)
                pass_costs[i] += size
                placed = True
                break

        if not placed:
            passes.append(list(unit))
            pass_costs.append(size)

    return passes


# Backward-compatible alias
_bin_pack_units_costed = _bin_pack_items_costed
