from __future__ import annotations
import math
from typing import Sequence
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
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

