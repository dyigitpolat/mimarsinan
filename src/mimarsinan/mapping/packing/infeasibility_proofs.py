"""Sound infeasibility proofs for ONE pack: what a refusal PROVES, never what it suspects."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, Tuple

#: Every softcore was placed.
VERDICT_FEASIBLE = "feasible"
#: No placement of this pack exists — a prover below says so.
VERDICT_PROVEN_INFEASIBLE = "proven_infeasible"
#: The greedy engine refused this pack; some other placement order may still take it.
VERDICT_HEURISTIC_FAILED = "heuristic_failed"

_VERDICTS: Tuple[str, ...] = (
    VERDICT_PROVEN_INFEASIBLE,
    VERDICT_HEURISTIC_FAILED,
    VERDICT_FEASIBLE,
)


class SoftcoreExtent(Protocol):
    """The axon x neuron extent a proof reads off a softcore."""

    @property
    def input_count(self) -> int: ...

    @property
    def output_count(self) -> int: ...


class CoreTypeDeclaration(Protocol):
    """The geometry and multiplicity a proof reads off a declared core type."""

    @property
    def max_axons(self) -> int: ...

    @property
    def max_neurons(self) -> int: ...

    @property
    def count(self) -> int: ...


def largest_extents(softcores: Sequence[SoftcoreExtent]) -> Tuple[int, int]:
    """The largest axon and the largest neuron extent any single softcore demands."""
    if not softcores:
        return (0, 0)
    return (
        max(int(sc.input_count) for sc in softcores),
        max(int(sc.output_count) for sc in softcores),
    )


def no_core_type_fits(
    softcores: Sequence[SoftcoreExtent],
    core_types: Sequence[CoreTypeDeclaration],
    *,
    neurons_may_split: bool,
    axons_may_spread: bool,
) -> bool:
    """Sound: a placed unit lands on ONE declared type, whose ``max_neurons`` bounds
    its neuron extent unless splitting may cut that dimension and whose ``max_axons``
    bounds its axon extent unless a spreading mechanism (declared coalescing, or the
    placement engine's hardcore fusion) may distribute it — so a demand no declared
    type covers in a dimension nothing relaxes has no placement at all.
    """
    if not softcores:
        return False
    max_axons_needed, max_neurons_needed = largest_extents(softcores)
    return not any(
        (axons_may_spread or int(ct.max_axons) >= max_axons_needed)
        and (neurons_may_split or int(ct.max_neurons) >= max_neurons_needed)
        for ct in core_types
    )


def cells_exceed_total_capacity(
    softcores: Sequence[SoftcoreExtent],
    core_types: Sequence[CoreTypeDeclaration],
) -> bool:
    """Sound: splitting and coalescing PARTITION a softcore's cells (the fragments'
    extents sum back to the original) and fusion only sums same-type crossbars, so no
    permission shrinks the committed cell total or grows the declared one — and one
    pack never reuses a core, so a demand above the declared total cannot be placed.
    """
    demand = sum(int(sc.input_count) * int(sc.output_count) for sc in softcores)
    declared = sum(
        int(ct.max_axons) * int(ct.max_neurons) * int(ct.count) for ct in core_types
    )
    return demand > declared


def failed_pack_verdict(
    softcores: Sequence[SoftcoreExtent],
    core_types: Sequence[CoreTypeDeclaration],
    *,
    neurons_may_split: bool,
    axons_may_spread: bool,
) -> str:
    """The class a FAILED pack belongs to: proven when a prover fires, refused otherwise."""
    proven = no_core_type_fits(
        softcores, core_types,
        neurons_may_split=neurons_may_split, axons_may_spread=axons_may_spread,
    ) or cells_exceed_total_capacity(softcores, core_types)
    return VERDICT_PROVEN_INFEASIBLE if proven else VERDICT_HEURISTIC_FAILED


def tag_with_verdict(verdict: str, message: str) -> str:
    """A refusal message stating its class up front, so a census self-classifies."""
    return f"[{verdict}] {message}"


def _declared_verdict(message: str) -> Optional[str]:
    for verdict in _VERDICTS:
        if message.startswith(f"[{verdict}]"):
            return verdict
    return None


def verdict_of_message(message: str) -> str:
    """The class a refusal message declares; an untagged message proves nothing."""
    return _declared_verdict(message) or VERDICT_HEURISTIC_FAILED


def ensure_verdict_tag(message: str) -> str:
    """The message with its class stated — propagated when present, unproven when absent."""
    if _declared_verdict(message) is not None:
        return message
    return tag_with_verdict(VERDICT_HEURISTIC_FAILED, message)
