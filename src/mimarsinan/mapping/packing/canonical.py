from __future__ import annotations

from typing import List, Protocol, TypeVar


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


def _read_threshold_group(obj, *, fallback_id: int | None = None) -> int:
    """Canonical threshold-group id read; ungrouped objects get a unique ``-(id+1)`` fallback."""
    tg = getattr(obj, "threshold_group_id", None)
    if tg is not None:
        return int(tg)
    if fallback_id is None:
        fallback_id = int(getattr(obj, "id", 0))
    return -(fallback_id + 1)


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


def canonical_is_mapping_possible(softcore: SoftCoreLike, hardcore: HardCoreLike) -> bool:
    """Single-source-of-truth feasibility predicate shared by the layout and runtime packers.

    A hardcore accepts a softcore only when threshold-group, latency, and the
    remaining axon/neuron budget all match.
    """
    hc_tg = getattr(hardcore, "threshold_group_id", None)
    if hc_tg is not None:
        sc_tg = _read_threshold_group(softcore)
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

