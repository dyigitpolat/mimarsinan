"""[wsm V2] Bank-clustered pass composition: banks stay resident, instances stream."""

from __future__ import annotations

from math import ceil
from typing import Sequence


def _fitting_capacity(group: list, cores_config: Sequence[dict]) -> "list[int] | None":
    """Per-core-type usable counts for this bank (None = some instance fits nowhere)."""
    max_axons = max(len(core.input_sources.flatten()) for core in group)
    max_neurons = max(int(core.get_output_count()) for core in group)
    fits = [
        int(ct.get("count", 0))
        if max_axons <= int(ct["max_axons"]) and max_neurons <= int(ct["max_neurons"])
        else 0
        for ct in cores_config
    ]
    return fits if any(fits) else None


def _grant(alloc: dict, taken: list, fits: dict, bank) -> bool:
    """Take one core of a type that fits ``bank``; False when none remains."""
    for type_index, capacity in enumerate(fits[bank]):
        if capacity > taken[type_index] and capacity > 0:
            taken[type_index] += 1
            alloc[bank] += 1
            return True
    return False


def try_bank_clustered_passes(
    *,
    cores: list,
    cores_config: Sequence[dict],
    weight_banks: dict,
    max_schedule_passes: int,
) -> "list[list] | None":
    """Compose passes so every physical core keeps one bank while its
    instance queue drains (the weight-stationary regime).

    Allocation law: ``loads(b) = ceil(n_b / max_passes)`` is the FEASIBILITY
    FLOOR, then allocations EXPAND into remaining pool capacity (type-aware)
    to shrink the makespan — weight reuse must not idle the chip; each
    duplicate costs exactly one extra bank programming, surfaced by the
    W_prog report. Returns per-pass core chunks (round-robin slot emission,
    banks ordered by descending pass count, validated per-ordinal stable so
    a resident physical core never changes its bank), or ``None`` when the
    segment is outside this policy's proven class: any owned core, any
    intra-segment dependency, an instance exceeding every core type,
    infeasible minimal residency, or an unstable composition (e.g. banks
    with UNEQUAL instance counts whose ragged tails desynchronize — no
    real vehicle today; equal-count tile banks stay lockstep by
    construction). Callers fall back to the capacity path.
    """
    del weight_banks  # applicability is structural; sizes live on placements
    if not cores:
        return None
    segment_ids = {core.id for core in cores}
    groups: dict[int, list] = {}
    for core in cores:
        if core.weight_bank_id is None:
            return None
        for source in core.input_sources.flatten():
            if getattr(source, "node_id", None) in segment_ids:
                return None
        groups.setdefault(int(core.weight_bank_id), []).append(core)

    pool_total = sum(int(ct.get("count", 0)) for ct in cores_config)
    if pool_total <= 0 or len(groups) > pool_total:
        return None
    fits: dict[int, list[int]] = {}
    for bank, group in groups.items():
        capacity = _fitting_capacity(group, cores_config)
        if capacity is None:
            return None
        fits[bank] = capacity

    counts = {b: len(g) for b, g in groups.items()}
    budget = max(1, int(max_schedule_passes))
    taken = [0] * len(cores_config)
    alloc = {b: 0 for b in groups}
    # Feasibility floor first, tightest banks (fewest fitting types) first.
    for bank in sorted(groups, key=lambda b: (sum(1 for c in fits[b] if c), b)):
        for _ in range(max(1, ceil(counts[bank] / budget))):
            if not _grant(alloc, taken, fits, bank):
                return None
    # Expansion: duplicate the bottleneck banks onto idle capacity while it
    # shrinks the makespan (granting to non-bottleneck banks buys nothing).
    # Equal-count banks expand in LOCKSTEP rounds — equal allocations keep
    # their ragged tails aligned, which per-ordinal stability requires.
    classes: dict[int, list[int]] = {}
    for bank in groups:
        classes.setdefault(counts[bank], []).append(bank)
    while True:
        count = max(
            classes, key=lambda n: (ceil(n / alloc[classes[n][0]]), n)
        )
        members = classes[count]
        if ceil(count / alloc[members[0]]) <= 1:
            break
        if alloc[members[0]] >= count:
            break
        granted = []
        for bank in members:
            if not _grant(alloc, taken, fits, bank):
                break
            granted.append(bank)
        if len(granted) < len(members):
            for bank in granted:  # roll back the partial round
                alloc[bank] -= 1
            break

    order = sorted(
        groups, key=lambda b: (-ceil(counts[b] / alloc[b]), -counts[b], b)
    )
    passes = max(ceil(counts[b] / alloc[b]) for b in order)
    remaining = {b: list(groups[b]) for b in order}
    chunks: list[list] = []
    for _ in range(passes):
        chunk: list = []
        for slot in range(max(alloc.values())):
            for b in order:
                if slot < alloc[b] and remaining[b]:
                    chunk.append(remaining[b].pop(0))
        if chunk:
            chunks.append(chunk)

    # Per-ordinal stability: a physical core must carry ONE bank across every
    # pass it appears in (chunks must shrink as prefixes of pass 0's layout).
    reference = [int(core.weight_bank_id) for core in chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk) > len(reference):
            return None
        if any(
            int(core.weight_bank_id) != reference[i]
            for i, core in enumerate(chunk)
        ):
            return None
    return chunks


def _stage_geometry(stage) -> list:
    segment = stage.hard_core_mapping
    geometry = []
    for placements in segment.soft_core_placements_per_hard_core:
        geometry.append(frozenset(
            (placement.get("weight_bank_id"), int(placement["axons"]),
             int(placement["neurons"]), int(placement["axon_offset"]),
             int(placement["neuron_offset"]))
            for placement in placements
        ))
    return geometry


def mark_bank_residency(pass_stages: list) -> None:
    """Verify every pass p>0 places, per physical core ordinal, a SUBSET of
    pass 0's (bank, region) placements — regions already programmed, so zero
    new programming (the ragged final pass uses fewer regions). Then mark
    those stages weight-resident. Any divergence is a scheduler bug — fail
    loud, never silently claim residency."""
    if len(pass_stages) <= 1:
        return
    reference = _stage_geometry(pass_stages[0])
    for stage in pass_stages[1:]:
        geometry = _stage_geometry(stage)
        if len(geometry) > len(reference) or any(
            not geometry[i] <= reference[i] for i in range(len(geometry))
        ):
            raise RuntimeError(
                f"bank_clustered residency violated at pass "
                f"{stage.schedule_pass_index} of segment "
                f"{stage.schedule_segment_index}: placement geometry diverged "
                f"from pass 0 — weights would silently reprogram."
            )
        stage.schedule_weights_resident = True
