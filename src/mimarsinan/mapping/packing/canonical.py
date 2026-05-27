from __future__ import annotations

from collections import Counter
from typing import Callable, List, Protocol, TypeVar


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
    """Read latency from a softcore or hardcore, tolerating either the
    runtime-side ``latency`` attribute or the layout-side ``latency_tag``
    attribute.  Returns ``None`` if neither is set.
    """
    lat = getattr(obj, "latency", None)
    if lat is not None:
        return int(lat)
    lat = getattr(obj, "latency_tag", None)
    return int(lat) if lat is not None else None


def _read_threshold_group(obj, *, fallback_id: int | None = None) -> int:
    """Canonical threshold-group id read.

    Uses the object's ``threshold_group_id`` when set; otherwise falls
    back to ``-(fallback_id + 1)`` (or ``-(id + 1)`` when the object has
    an ``id`` attribute).  Both are unique-per-object, consistent with
    how ``LayoutIRMapping._finalize_softcores`` and
    how ``LayoutIRMapping._finalize_softcores`` have historically disambiguated
    ungrouped softcores.
    """
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
    """Shared fusion protocol used by both the layout packer and the real
    hard-core mapper.

    The algorithm is identical — sum axons across the fused cores, keep
    the first core's neuron width.  Only the *construction* of the fused
    instance is type-specific (real ``HardCore`` must carry over
    ``threshold``/``activation_scale``/``hardware_bias``/``fused_component_axons``;
    the layout ``LayoutHardCoreInstance`` only needs dimensions).  That
    work is delegated to ``make_fused`` so both paths take the same
    branching decisions inside ``greedy_pack_softcores``.
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
    """Shared neuron-dimension split protocol.

    Produces fragment shapes ``(axons, available_neurons)`` and
    ``(axons, remaining)`` where ``remaining = total_neurons - available_neurons``.
    Both the layout packer and the runtime packer cut at the same
    boundary (the caller's ``available_neurons``).  ``make_fragments`` is
    type-specific: layout builds two ``LayoutSoftCoreSpec``s with shape
    + carried-over group/latency, runtime builds two ``SoftCore``s that
    additionally carry sliced ``core_matrix`` / ``axon_sources`` /
    ``hardware_bias`` / bank metadata.  The split *decision* stays in one
    place so the downstream packing order is identical in both paths.
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
    """Single-source-of-truth feasibility predicate for bin-packing.

    Both the layout-level ``pack_layout`` and the runtime-level
    hard-core mapper call this via the shared
    ``greedy_pack_softcores`` algorithm, so the wizard's feasibility
    estimate and the pipeline's real packing decision follow identical
    rules:

    * **Threshold-group**: once a hardcore is claimed by a softcore's
      threshold group, it may only accept softcores from the same group.
      Empty hardcores (``hardcore.threshold_group_id is None``) accept
      any softcore.  The softcore group id is read canonically
      (``_read_threshold_group``); ungrouped softcores get a per-object
      unique fallback so they never mistakenly share a hardcore.
    * **Latency**: once a hardcore is pinned to a latency, it accepts
      only softcores with that same latency.  Read tolerates either
      ``.latency`` (runtime) or ``.latency_tag`` (layout) attribute
      names so both types flow through the same check.
    * **Dimensions**: the softcore's input/output counts must fit the
      hardcore's remaining axon/neuron budget (``available_axons`` /
      ``available_neurons`` on runtime HardCore, or the full dimensions
      reported by ``LayoutHardCoreInstance.get_input_count()`` /
      ``get_output_count()`` which already reflect remaining capacity).
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
    """
    Estimate the area wasted by placing *soft* into *hard*.

    In diagonal packing each softcore occupies a block of axons × neurons
    along the diagonal.  The dead zone for a single placement is the
    L-shaped region that neither the softcore nor any future placement
    can use::

        waste = (h_a − s_a) · s_n + s_a · (h_n − s_n)
              = h_a · s_n + s_a · h_n − 2 · s_a · s_n

    This naturally penalises aspect-ratio mismatches: a softcore that is
    narrow in axons but wide in neurons will score much lower on a
    narrow-axon core type than on a square core of the same total area.
    """
    s_a = int(soft.get_input_count())
    s_n = int(soft.get_output_count())
    h_a = int(hard.get_input_count())
    h_n = int(hard.get_output_count())
    return h_a * s_n + s_a * h_n - 2 * s_a * s_n


def _remaining_capacity(soft: SoftCoreLike, hard: HardCoreLike) -> int:
    """
    Remaining usable area after placing *soft* into *hard*.

    Uses ``available_axons`` / ``available_neurons`` when present (used
    cores), otherwise falls back to the full core dimensions (unused cores).

    A lower value indicates a tighter fit — the softcore fills up the
    remaining space more completely, reducing fragmentation.
    """
    avail_a = getattr(hard, "available_axons", int(hard.get_input_count()))
    avail_n = getattr(hard, "available_neurons", int(hard.get_output_count()))
    s_a = int(soft.get_input_count())
    s_n = int(soft.get_output_count())
    return (avail_a - s_a) * (avail_n - s_n)


def pick_best_softcore(unmapped_cores: List[SoftT]) -> SoftT:
    """
    Heuristic used by the existing HardCoreMapping:
    - pick the core with max input count
    - pick the core with max output count
    - choose whichever is more extreme
    """
    if not unmapped_cores:
        raise ValueError("unmapped_cores is empty")

    # Single linear scan is O(n) vs two full sorts O(n log n) — same winner.
    core_a = max(unmapped_cores, key=lambda c: c.get_input_count())
    core_b = max(unmapped_cores, key=lambda c: c.get_output_count())

    if core_a.get_input_count() > core_b.get_output_count():
        return core_a
    return core_b

