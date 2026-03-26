"""
Schedule partitioner for splitting neural segments into multiple passes.

When the hardware has fewer cores than a neural segment requires, this module
partitions the segment's cores into ordered passes that can each fit on
the available hardware.  Passes execute sequentially, reusing the same physical
cores (reprogrammed between passes).

The partitioning respects two hard invariants:

1. **Latency ordering** — cores in pass *k* depend only on cores in passes
   < *k* (or external inputs).  This is guaranteed by assigning latency groups
   in increasing order.
2. **Coalescing integrity** — all hardware cores belonging to a single
   coalescing group (wide axon tiling) must reside in the **same** pass.
   The hardware lacks membrane-potential initialization across passes, so
   partial sums cannot be accumulated temporally.  If no core type has a
   sufficient ``count`` for the widest coalescing group, the configuration
   is infeasible.

Within a single latency group, cores are independent (by definition of "same
depth in the dependency DAG"), so any partition of the group is valid.

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
# Validate-and-split refinement (mirrors build-time _flush_or_split)
# ---------------------------------------------------------------------------

def _validate_and_split_passes(
    passes: list[list[T]],
    pack_validator: Callable[[list[T]], bool],
    fragment_expander: Optional[Callable[[T], Optional[list[T]]]] = None,
    max_per_pass: int = 0,
) -> tuple[list[list[T]], bool]:
    """Validate each pass with real packing; split or fragment on failure.

    Mirrors the build-time ``_flush_or_split`` pattern: if a pass fails
    typed packing, split it in half and retry recursively.

    When a **single-item** pass fails and *fragment_expander* is provided,
    the item is expanded into fragment-sized sub-items (e.g. coalescing /
    splitting fragments).  Fragments are distributed into sub-passes of at
    most *max_per_pass* items and each sub-pass is validated recursively.
    This handles the case where a softcore's fragments exceed the available
    core count — the fragments are simply scheduled across multiple passes.

    If fragment expansion is not possible (expander returns ``None``) or a
    fragment itself cannot pack, the configuration is truly infeasible.

    Returns ``(validated_passes, all_feasible)``.
    """
    validated: list[list[T]] = []
    all_ok = True

    def _process(items: list[T]) -> None:
        nonlocal all_ok
        if not items:
            return
        if pack_validator(items):
            validated.append(items)
            return
        if len(items) > 1:
            mid = len(items) // 2
            _process(items[:mid])
            _process(items[mid:])
            return
        # Single item failed — try fragment expansion.
        if fragment_expander is not None:
            frags = fragment_expander(items[0])
            if frags is not None:
                k = max_per_pass if max_per_pass > 0 else len(frags)
                sub_passes = [frags[i:i + k] for i in range(0, len(frags), k)]
                for sp in sub_passes:
                    _process(sp)
                return
        all_ok = False
        validated.append(items)

    for p in passes:
        _process(p)

    return validated, all_ok


def _make_softcore_fragment_expander(
    hw_types: Sequence,
    allow_coalescing: bool,
    allow_splitting: bool,
) -> Callable[[LayoutSoftCoreSpec], Optional[list[LayoutSoftCoreSpec]]]:
    """Create a splitting-only fragment expander for layout softcores.

    When a softcore is too large for a single-pass packing, the expander
    produces sub-softcores split along the **neuron** dimension only.
    Each fragment retains the full input (axon) width so that coalescing
    groups remain intact within a single pass.

    Returns ``None`` when:
    - No core type can satisfy the coalescing requirement (insufficient
      ``count``).  This marks the configuration as truly infeasible.
    - The softcore already fits in a single pass (no expansion needed).
    """
    if not hw_types:
        return lambda _sc: None

    def _expander(sc: LayoutSoftCoreSpec) -> Optional[list[LayoutSoftCoreSpec]]:
        best_hw = None
        best_n_split = 0

        for hw in hw_types:
            if not allow_splitting and sc.output_count > hw.max_neurons:
                continue

            n_coalesce = (
                math.ceil(sc.input_count / hw.max_axons)
                if allow_coalescing and hw.max_axons > 0
                else 1
            )

            if n_coalesce > hw.count:
                continue

            n_split = (
                math.ceil(sc.output_count / hw.max_neurons)
                if allow_splitting and hw.max_neurons > 0
                else 1
            )

            if n_split > 1 and (best_hw is None or n_split < best_n_split):
                best_n_split = n_split
                best_hw = hw

        if best_hw is None:
            return None

        frag_neu = min(sc.output_count, best_hw.max_neurons)

        return [
            LayoutSoftCoreSpec(
                input_count=sc.input_count,
                output_count=frag_neu,
                segment_id=sc.segment_id,
                latency_tag=sc.latency_tag,
                threshold_group_id=sc.threshold_group_id,
                name=f"{sc.name}_split_frag{i}" if sc.name else None,
            )
            for i in range(best_n_split)
        ]

    return _expander


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
    core_types: Optional[Sequence] = None,
) -> Tuple[int, List[List[LayoutSoftCoreSpec]]]:
    """Estimate the number of schedule passes from layout soft-core specs.

    Groups by ``segment_id`` then delegates to the **same** partitioning
    algorithm used by :func:`partition_segment_into_passes`, ensuring
    identical pass counts for identical shapes.

    When *core_types* (a sequence of ``LayoutHardCoreType``) is provided,
    each proposed pass is validated with ``pack_layout`` against the real
    typed hardware.  Passes that fail are split in half and retried,
    mirroring the build-time ``_flush_or_split`` pattern.  If a single
    softcore cannot pack, the overall result is marked infeasible.

    Returns ``(num_passes, pass_lists)`` where *pass_lists* is flattened
    across all segments (segment 0 passes first, then segment 1, etc.).
    When *core_types* is given and any single-item pass fails, the third
    element of the returned tuple indicates infeasibility — callers should
    use :func:`estimate_passes_for_layout_validated` for this.
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

    # Validate each pass with real typed packing when core_types is given.
    if core_types is not None:
        from mimarsinan.mapping.layout.layout_packer import pack_layout as _pack_layout

        def _validator(pass_scs: list[LayoutSoftCoreSpec]) -> bool:
            try:
                pr = _pack_layout(
                    softcores=pass_scs,
                    core_types=list(core_types),
                    allow_neuron_splitting=allow_splitting,
                    allow_axon_coalescing=allow_coalescing,
                )
                return pr.feasible
            except Exception:
                return False

        _expander = _make_softcore_fragment_expander(
            list(core_types), allow_coalescing, allow_splitting,
        )
        all_passes, _all_ok = _validate_and_split_passes(
            all_passes, _validator,
            fragment_expander=_expander,
            max_per_pass=max_cores_per_pass,
        )

    num_passes = len(all_passes)
    return num_passes, all_passes


def estimate_passes_for_layout_validated(
    softcores: Sequence[LayoutSoftCoreSpec],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    core_types: Sequence,
) -> Tuple[int, List[List[LayoutSoftCoreSpec]], bool]:
    """Like :func:`estimate_passes_for_layout` but returns feasibility.

    Returns ``(num_passes, pass_lists, all_feasible)`` where
    *all_feasible* is ``False`` when at least one single-softcore pass
    could not pack on the given *core_types*.
    """
    if not softcores or max_cores_per_pass <= 0:
        ok = not softcores
        return (0, [], ok) if not softcores else (1, [list(softcores)], ok)

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

    from mimarsinan.mapping.layout.layout_packer import pack_layout as _pack_layout

    def _validator(pass_scs: list[LayoutSoftCoreSpec]) -> bool:
        try:
            pr = _pack_layout(
                softcores=pass_scs,
                core_types=list(core_types),
                allow_neuron_splitting=allow_splitting,
                allow_axon_coalescing=allow_coalescing,
            )
            return pr.feasible
        except Exception:
            return False

    _expander = _make_softcore_fragment_expander(
        list(core_types), allow_coalescing, allow_splitting,
    )
    all_passes, all_ok = _validate_and_split_passes(
        all_passes, _validator,
        fragment_expander=_expander,
        max_per_pass=max_cores_per_pass,
    )
    return len(all_passes), all_passes, all_ok


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
