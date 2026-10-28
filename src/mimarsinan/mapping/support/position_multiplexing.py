"""[W02] Position multiplexing: one copy of each shared weight bank serves its instances (conv positions) in turn.

A declared cost-only mapping decision of the pass planner (``position_multiplexing``). Per segment and per
dependency level, the instances of a bank run one per pass on one copy: a run of passes places one instance of each
bank of the run in every pass, its first pass programs the copies and every later pass executes on them (resident),
so the program commits each bank's cells once per run and pays for it in passes, latency and carry, the existing
terms. Owned cores of a level take the validated capacity split. Undeclared, the planner composes as before (one
copy per instance wherever the chip holds them), byte for byte.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.schedule.schedule_partitioner import estimate_passes_for_layout_validated

__all__ = ["CopyKey", "MultiplexedPasses", "PositionMultiplexingError", "copy_keys", "multiplexed_segment_passes",
           "shares_a_bank"]

#: An instance's copy key: instances of one bank with one extent share one copy.
CopyKey = Tuple[int, int, int]


@dataclass(frozen=True)
class MultiplexedPasses:
    """One segment's passes under position multiplexing; ``run_heads[j]``: pass j programs its weights."""

    pass_lists: Tuple[Tuple[LayoutSoftCoreSpec, ...], ...]
    run_heads: Tuple[bool, ...]
    feasible: bool


def copy_keys(specs: Sequence[LayoutSoftCoreSpec], shares: Optional[Sequence[bool]]) -> List[Optional[CopyKey]]:
    return [
        None if spec.bank_id is None or (shares is not None and not shares[i])
        else (int(spec.bank_id), int(spec.input_count), int(spec.output_count))
        for i, spec in enumerate(specs)
    ]


def shares_a_bank(specs: Sequence[LayoutSoftCoreSpec], shares: Optional[Sequence[bool]] = None) -> bool:
    """Whether the segment holds a copy key with two or more instances (else the decision changes nothing)."""
    counts = Counter(key for key in copy_keys(specs, shares) if key is not None)
    return any(n >= 2 for n in counts.values())


def multiplexed_segment_passes(
    specs: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    budget: int,
    allow_coalescing: bool,
    allow_splitting: bool,
    max_schedule_passes: int,
    shares: Optional[Sequence[bool]] = None,
    coalescing_group_ids: Optional[Sequence[Optional[int]]] = None,
    label: str = "segment",
) -> MultiplexedPasses:
    """Levels in latency order; per level its owned cores' capacity passes, then the runs of its copy keys.

    Copy keys of one instance count run together while one instance of each packs on the declared cores; a run of
    n instances is n passes, each the same multiset of shapes, so every pass packs as the run's first. ``shares``
    (aligned with ``specs``) excludes instances that do not read their bank's copy (the builder's compacted ones).
    """
    keys = copy_keys(specs, shares)
    levels = sorted({int(spec.latency_tag or 0) for spec in specs})
    passes: List[Tuple[LayoutSoftCoreSpec, ...]] = []
    heads: List[bool] = []
    feasible = True
    for level in levels:
        members = [i for i, spec in enumerate(specs) if int(spec.latency_tag or 0) == level]
        owned = [i for i in members if keys[i] is None]
        if owned:
            _, owned_passes, ok = estimate_passes_for_layout_validated(
                [specs[i] for i in owned], budget, core_types=core_types,
                allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                coalescing_group_ids=None if coalescing_group_ids is None else [coalescing_group_ids[i] for i in owned])
            feasible = feasible and ok
            passes.extend(tuple(chunk) for chunk in owned_passes)
            heads.extend(True for _ in owned_passes)
        instances: Dict[CopyKey, List[int]] = {}
        for i in members:
            key = keys[i]
            if key is not None:
                instances.setdefault(key, []).append(i)
        by_count: Dict[int, List[CopyKey]] = {}
        for key, idx in instances.items():
            by_count.setdefault(len(idx), []).append(key)
        for count, copies in by_count.items():
            for run in _runs(copies, instances, specs, core_types, allow_coalescing, allow_splitting, label):
                for k in range(count):
                    passes.append(tuple(specs[instances[key][k]] for key in run))
                    heads.append(k == 0)
    if len(passes) > int(max_schedule_passes):
        raise PositionMultiplexingError(
            f"position_multiplexing: {label} needs {len(passes)} passes (one copy of each shared bank serving its "
            f"instances in turn), above max_schedule_passes {int(max_schedule_passes)}; declare a larger pass budget")
    return MultiplexedPasses(pass_lists=tuple(passes), run_heads=tuple(heads), feasible=feasible)


def _runs(copies, instances, specs, core_types, allow_coalescing, allow_splitting, label) -> List[List[CopyKey]]:
    """Copy keys grouped greedily, in order, while one instance of each packs on the declared cores."""
    def packs(group: List[CopyKey]) -> bool:
        return pack_layout(softcores=[specs[instances[key][0]] for key in group], core_types=core_types,
                           allow_neuron_splitting=allow_splitting, allow_coalescing=allow_coalescing).feasible

    runs: List[List[CopyKey]] = []
    current: List[CopyKey] = []
    for key in copies:
        if packs(current + [key]):
            current.append(key)
            continue
        if not current or not packs([key]):
            spec = specs[instances[key][0]]
            raise PositionMultiplexingError(
                f"position_multiplexing: in {label}, one copy of bank {key[0]} ({spec.input_count} axons, "
                f"{spec.output_count} neurons) does not pack on the declared cores")
        runs.append(current)
        current = [key]
    if current:
        runs.append(current)
    return runs
