"""[wsm V2] The bank-clustered pass-composition law, stated once.

Two callers ask the same question at two completenesses: the hard-core builder
composes passes over IR ``NeuralCore``s at deployment time, and the shape-only
layout answer (search candidates, wizard preview, agent introspection) composes
them over ``LayoutSoftCoreSpec``s. They must agree, so the allocation lives HERE
and neither side re-derives it — a second implementation of this law would be a
second schedule, which is exactly the divergence W5.2 closes.

Everything policy-specific about *applicability* (owned cores, intra-segment
dependencies) stays with the caller that can see it; this module only allocates
resident cores to banks and emits the pass chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class BankInstance:
    """One bank-backed instance as the allocator sees it: which bank, what extent."""

    bank_id: int
    axons: int
    neurons: int


def _fitting_capacity(
    group: Sequence[BankInstance], cores_config: Sequence[Mapping[str, Any]]
) -> "List[int] | None":
    """Per-core-type usable counts for this bank (None = some instance fits nowhere)."""
    max_axons = max(inst.axons for inst in group)
    max_neurons = max(inst.neurons for inst in group)
    fits = [
        int(ct.get("count", 0))
        if max_axons <= int(ct["max_axons"]) and max_neurons <= int(ct["max_neurons"])
        else 0
        for ct in cores_config
    ]
    return fits if any(fits) else None


def _grant(
    alloc: Dict[int, int], taken: List[int], fits: Dict[int, List[int]], bank: int
) -> bool:
    """Take one core of a type that fits ``bank``; False when none remains."""
    for type_index, capacity in enumerate(fits[bank]):
        if capacity > taken[type_index] and capacity > 0:
            taken[type_index] += 1
            alloc[bank] += 1
            return True
    return False


def compose_bank_clustered_passes(
    instances: Sequence[BankInstance],
    cores_config: Sequence[Mapping[str, Any]],
    *,
    max_schedule_passes: int,
) -> "List[List[int]] | None":
    """Compose passes so every physical core keeps one bank while its instance
    queue drains (the weight-stationary regime).

    Returns per-pass lists of INSTANCE INDICES into *instances* (round-robin slot
    emission, banks ordered by descending pass count, validated per-ordinal
    stable so a resident physical core never changes its bank), or ``None`` when
    the segment is outside this policy's proven class: an instance exceeding
    every core type, infeasible minimal residency, or an unstable composition
    (e.g. banks with UNEQUAL instance counts whose ragged tails desynchronize).

    Allocation law: ``loads(b) = ceil(n_b / max_passes)`` is the FEASIBILITY
    FLOOR, then allocations EXPAND into remaining pool capacity (type-aware) to
    shrink the makespan — weight reuse must not idle the chip; each duplicate
    costs exactly one extra bank programming, surfaced by the W_prog report.
    """
    if not instances:
        return None

    groups: Dict[int, List[int]] = {}
    for index, instance in enumerate(instances):
        groups.setdefault(int(instance.bank_id), []).append(index)

    pool_total = sum(int(ct.get("count", 0)) for ct in cores_config)
    if pool_total <= 0 or len(groups) > pool_total:
        return None

    fits: Dict[int, List[int]] = {}
    for bank, group in groups.items():
        capacity = _fitting_capacity([instances[i] for i in group], cores_config)
        if capacity is None:
            return None
        fits[bank] = capacity

    counts = {bank: len(group) for bank, group in groups.items()}
    budget = max(1, int(max_schedule_passes))
    taken = [0] * len(cores_config)
    alloc = {bank: 0 for bank in groups}
    # Feasibility floor first, tightest banks (fewest fitting types) first.
    for bank in sorted(groups, key=lambda b: (sum(1 for c in fits[b] if c), b)):
        for _ in range(max(1, ceil(counts[bank] / budget))):
            if not _grant(alloc, taken, fits, bank):
                return None
    # Expansion: duplicate the bottleneck banks onto idle capacity while it
    # shrinks the makespan (granting to non-bottleneck banks buys nothing).
    # Equal-count banks expand in LOCKSTEP rounds — equal allocations keep
    # their ragged tails aligned, which per-ordinal stability requires.
    classes: Dict[int, List[int]] = {}
    for bank in groups:
        classes.setdefault(counts[bank], []).append(bank)
    while True:
        count = max(classes, key=lambda n: (ceil(n / alloc[classes[n][0]]), n))
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

    order = sorted(groups, key=lambda b: (-ceil(counts[b] / alloc[b]), -counts[b], b))
    passes = max(ceil(counts[b] / alloc[b]) for b in order)
    remaining = {bank: list(groups[bank]) for bank in order}
    chunks: List[List[int]] = []
    for _ in range(passes):
        chunk: List[int] = []
        for slot in range(max(alloc.values())):
            for bank in order:
                if slot < alloc[bank] and remaining[bank]:
                    chunk.append(remaining[bank].pop(0))
        if chunk:
            chunks.append(chunk)

    # Per-ordinal stability: a physical core must carry ONE bank across every
    # pass it appears in (chunks must shrink as prefixes of pass 0's layout).
    reference = [int(instances[i].bank_id) for i in chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk) > len(reference):
            return None
        if any(
            int(instances[index].bank_id) != reference[ordinal]
            for ordinal, index in enumerate(chunk)
        ):
            return None
    return chunks
