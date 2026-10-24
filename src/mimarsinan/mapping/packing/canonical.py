from __future__ import annotations

from mimarsinan.mapping.platform.core_residency import ungrouped_fallback_id

from typing import List, Optional, Protocol, Sequence, TypeVar


class SoftCoreLike(Protocol):
    def get_input_count(self) -> int: ...

    def get_output_count(self) -> int: ...


class HardCoreLike(Protocol):
    def get_input_count(self) -> int: ...

    def get_output_count(self) -> int: ...


SoftT = TypeVar("SoftT", bound=SoftCoreLike)
HardT = TypeVar("HardT", bound=HardCoreLike)


def _capacity(hc: HardCoreLike) -> int:
    """Total capacity (area) of a hardware core."""
    return int(hc.get_input_count()) * int(hc.get_output_count())


def _read_latency(obj) -> int | None:
    """Read latency tolerating runtime ``latency`` or layout ``latency_tag``; None if neither."""
    lat = getattr(obj, "latency", None)
    if lat is not None:
        return int(lat)
    lat = getattr(obj, "latency_tag", None)
    return int(lat) if lat is not None else None


def _read_residency_class(obj, *, fallback_id: int | None = None) -> int:
    """Canonical residency-class id read; ungrouped objects get a unique ``-(id+1)`` fallback."""
    tg = getattr(obj, "residency_class_id", None)
    if tg is not None:
        return int(tg)
    if fallback_id is None:
        fallback_id = int(getattr(obj, "id", 0))
    return ungrouped_fallback_id(fallback_id)


def canonical_fuse_hardcores(
    hcs: List[HardT],
    *,
    make_fused,
) -> HardT:
    """Shared fusion protocol: sum axons across fused cores, keep the first core's neuron width.

    Type-specific instance construction is delegated to ``make_fused`` so the
    layout and runtime packers take identical branching decisions.
    """
    if not hcs:
        raise ValueError("Cannot fuse empty list of hardcores")
    first = hcs[0]
    fused_axons = sum(int(hc.get_input_count()) for hc in hcs)
    return make_fused(
        axons=fused_axons,
        neurons=int(first.get_output_count()),
        template=first,
        components=list(hcs),
    )


def canonical_split_softcore(
    softcore: SoftT,
    available_neurons: int,
    *,
    make_fragments,
) -> tuple[SoftT, SoftT]:
    """Shared neuron-dimension split: fragments of shape ``(axons, available_neurons)`` and the remainder.

    Both packers cut at the same boundary; ``make_fragments`` builds the
    type-specific fragment instances.
    """
    total_neurons = int(softcore.get_output_count())
    remaining = total_neurons - int(available_neurons)
    if remaining <= 0 or available_neurons <= 0:
        raise ValueError(
            f"Invalid split: available_neurons={available_neurons}, "
            f"total_neurons={total_neurons}"
        )
    return make_fragments(
        softcore=softcore,
        first_neurons=int(available_neurons),
        remaining_neurons=int(remaining),
    )


#: N2 core roles: a dynamic operand bank's instance, or every other (static) soft core.
OPERAND_ROLE, STATIC_ROLE = "operand", "static"


def declared_core_roles(cores_config: Sequence[dict]) -> List[Optional[frozenset]]:
    """Per declared core type, the roles its cores accept: on a chip that declares an operand core type beside
    static ones, operand types take operand banks only and the others static tiles only; otherwise every role."""
    operand = [ct.get("operand") is not None for ct in cores_config]
    if not any(operand) or all(operand):
        return [None] * len(operand)
    return [frozenset({OPERAND_ROLE}) if flag else frozenset({STATIC_ROLE}) for flag in operand]


def fused_core_roles(template_roles: Optional[frozenset]) -> frozenset:
    """The roles a fused core accepts: its template's, never the operand role (an operand bank sits on a whole
    core, where its per-sample payload is written)."""
    return frozenset(template_roles if template_roles is not None else (OPERAND_ROLE, STATIC_ROLE)) - {OPERAND_ROLE}


def canonical_is_mapping_possible(softcore: SoftCoreLike, hardcore: HardCoreLike) -> bool:
    """Single-source-of-truth feasibility predicate shared by the layout and runtime packers.

    A hardcore accepts a softcore only when residency-class, latency, the core role (N2, where the chip
    declares separate operand cores) and the remaining axon/neuron budget all match.
    """
    roles = getattr(hardcore, "accepted_roles", None)
    if roles is not None and (getattr(softcore, "core_role", None) or STATIC_ROLE) not in roles:
        return False
    hc_tg = getattr(hardcore, "residency_class_id", None)
    if hc_tg is not None:
        sc_tg = _read_residency_class(softcore)
        if sc_tg != int(hc_tg):
            return False

    hc_lat = _read_latency(hardcore)
    if hc_lat is not None:
        sc_lat = _read_latency(softcore)
        if sc_lat is None or sc_lat != hc_lat:
            return False

    avail_a = getattr(hardcore, "available_axons", None)
    avail_n = getattr(hardcore, "available_neurons", None)
    if avail_a is None:
        avail_a = int(hardcore.get_input_count())
    if avail_n is None:
        avail_n = int(hardcore.get_output_count())
    return (
        int(softcore.get_input_count()) <= int(avail_a)
        and int(softcore.get_output_count()) <= int(avail_n)
    )


#: Every hard-core attribute ``canonical_is_mapping_possible`` reads besides the core's shape (``get_input_count``,
#: ``get_output_count``); the best fit's cost for a softcore without a weight bank and the split into a used core read
#: no other. Two cores alike on these and on their shape answer every such softcore alike: the engine's core-state
#: index (``greedy/used_index.py``) keys on exactly this set. A bank-affine softcore also reads the core's banks:
#: on the index it reads the cores hosting its bank, then one core per state (``greedy/pack._best_used_core``).
CANONICAL_CORE_READS = ("residency_class_id", "latency", "latency_tag", "available_axons", "available_neurons",
                        "accepted_roles")


def _placement_waste(soft: SoftCoreLike, hard: HardCoreLike) -> int:
    """L-shaped diagonal-packing dead-zone area ``h_a·s_n + s_a·h_n − 2·s_a·s_n``; penalises aspect-ratio mismatch."""
    s_a = int(soft.get_input_count())
    s_n = int(soft.get_output_count())
    h_a = int(hard.get_input_count())
    h_n = int(hard.get_output_count())
    return h_a * s_n + s_a * h_n - 2 * s_a * s_n


def _remaining_capacity(soft: SoftCoreLike, hard: HardCoreLike) -> int:
    """Remaining usable area after placing *soft* into *hard*; lower means a tighter fit."""
    avail_a = getattr(hard, "available_axons", int(hard.get_input_count()))
    avail_n = getattr(hard, "available_neurons", int(hard.get_output_count()))
    s_a = int(soft.get_input_count())
    s_n = int(soft.get_output_count())
    return (avail_a - s_a) * (avail_n - s_n)


def pick_best_softcore(unmapped_cores: List[SoftT]) -> SoftT:
    """Pick the core whose max input/output count is most extreme."""
    if not unmapped_cores:
        raise ValueError("unmapped_cores is empty")

    core_a = max(unmapped_cores, key=lambda c: c.get_input_count())
    core_b = max(unmapped_cores, key=lambda c: c.get_output_count())

    if core_a.get_input_count() > core_b.get_output_count():
        return core_a
    return core_b

