"""The soma-axes fold and its cross-key contract (bits-driven, like WQ)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple

from mimarsinan.chip_simulation.soma_axes import (
    FIRING_GRANULARITY_KEY,
    MEMBRANE_ARITHMETIC_KEY,
    MEMBRANE_BITS_KEY,
    MEMBRANE_SIGNED_KEY,
    SATURATING_SIGNED_MEMBRANE,
    SATURATING_UNSIGNED_MEMBRANE,
    UNBOUNDED_MEMBRANE,
    resolved_firing_granularity,
    resolved_membrane_arithmetic,
    resolved_membrane_bits,
    resolved_membrane_signed,
)
from mimarsinan.chip_simulation.soma_law import FIRING_MODE_KEY, SomaLaw
from mimarsinan.mapping.platform.platform_constraints import bias_mode_for_cores

_PARAM_ENCODED_BIAS = "param_encoded"
_MEMBRANE_INIT_KEY = "lif_membrane_init"

# Which keys a violation's one-click remedy may clear, per offending key.
_REMEDY_KEYS: Dict[str, Tuple[str, ...]] = {
    MEMBRANE_ARITHMETIC_KEY: (
        MEMBRANE_ARITHMETIC_KEY, MEMBRANE_BITS_KEY, MEMBRANE_SIGNED_KEY),
    FIRING_GRANULARITY_KEY: (FIRING_GRANULARITY_KEY,),
    FIRING_MODE_KEY: (FIRING_MODE_KEY, FIRING_GRANULARITY_KEY),
    _MEMBRANE_INIT_KEY: (_MEMBRANE_INIT_KEY, FIRING_GRANULARITY_KEY),
}


def fold_soma_axes(dp: MutableMapping[str, Any]) -> None:
    """Fold the resolved soma axes into ``dp`` (idempotent).

    Runs EARLY — beside ``fold_spiking_axes``, before the recipe fold — so the
    ``sim_enables`` derivation and every downstream consumer see a resolved
    point instead of a raw config key. Total over every historical shape:
    axes-only, legacy-only, neither, and value-domain documents (where both
    axes resolve to today's inert law).
    """
    dp[FIRING_GRANULARITY_KEY] = resolved_firing_granularity(dp)
    dp[MEMBRANE_ARITHMETIC_KEY] = resolved_membrane_arithmetic(dp)


def enforce_soma_axes_contract(cfg: Mapping[str, Any]) -> None:
    """Cross-key rules a NON-default soma point must satisfy (raises).

    Inert at the default point, so no pre-existing configuration pays for the
    axes: only a declared width or a declared per-event law is judged here.
    """
    for _key, message in _contract_violations(cfg):
        raise ValueError(message)


def soma_contract_error_rows(
    dp: Mapping[str, Any], pc: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """The same rules as KEYED, remediable rows for the resolve channel."""
    cfg: Dict[str, Any] = {**dict(pc), **dict(dp)}
    return [
        {
            "key": key,
            "message": message,
            "rule_id": "soma_law_contract",
            "remedies": [
                {"label": f"Clear {name}", "action": "clear", "key": name}
                for name in _REMEDY_KEYS[key]
            ],
        }
        for key, message in _contract_violations(cfg)
    ]


def _contract_violations(cfg: Mapping[str, Any]) -> Iterator[Tuple[str, str]]:
    """Every ``(offending key, message)`` this resolved point violates."""
    law = SomaLaw.resolve(cfg)
    if law.is_default_point:
        return
    bits = resolved_membrane_bits(cfg)
    if bits > 0 and law.membrane_arithmetic == UNBOUNDED_MEMBRANE:
        yield MEMBRANE_ARITHMETIC_KEY, (
            f"membrane_arithmetic={UNBOUNDED_MEMBRANE!r} contradicts the "
            f"declared membrane_bits={bits}: the membrane law is bits-driven "
            f"(a declared width IS a saturating register, exactly as "
            f"weight_bits declares a quantized artifact). Remove "
            f"membrane_arithmetic to accept the derivation, or drop "
            f"membrane_bits."
        )
    if bits == 0 and law.saturates:
        yield MEMBRANE_ARITHMETIC_KEY, (
            f"membrane_arithmetic={law.membrane_arithmetic!r} needs a declared "
            f"membrane_bits width: a register that saturates has to say where. "
            f"Declare platform membrane_bits, or remove membrane_arithmetic."
        )
    if bits > 0 and law.saturates:
        declared_signed = resolved_membrane_signed(cfg)
        if declared_signed != law.is_signed_membrane:
            yield MEMBRANE_ARITHMETIC_KEY, (
                f"membrane_arithmetic={law.membrane_arithmetic!r} contradicts "
                f"the declared membrane_signed={declared_signed}: signedness is "
                f"physical structure of the register, declared once beside its "
                f"width, and the arithmetic falls out of the pair "
                f"({SATURATING_SIGNED_MEMBRANE!r} when signed, "
                f"{SATURATING_UNSIGNED_MEMBRANE!r} when not). Remove "
                f"membrane_arithmetic to accept the derivation, or declare the "
                f"register the target actually has."
            )
    if not law.is_per_event:
        return
    if law.is_signed_membrane:
        yield MEMBRANE_ARITHMETIC_KEY, (
            f"membrane_arithmetic={SATURATING_SIGNED_MEMBRANE!r} is refused "
            f"under firing_granularity='per_event': the event-serial law is "
            f"realized as ROW PAIRS whose zero-magnitude member is a no-op "
            f"only against a register that floors at zero, and no executor "
            f"here folds events on a two's-complement membrane. The signed "
            f"register exists to make a per-CYCLE window hold the same number "
            f"the unbounded accumulator holds. Declare membrane_signed=false, "
            f"or deploy per_cycle."
        )
    if law.firing_mode != law.required_firing_mode:
        yield FIRING_MODE_KEY, (
            f"firing_granularity='per_event' requires "
            f"firing_mode={law.required_firing_mode!r} (the hard-zero reset), "
            f"but the point declares {law.firing_mode!r}. The per-event law is "
            f"realized as ROW PAIRS — a zero-magnitude row folded beside every "
            f"event row — and that row is a no-op only while every fire leaves "
            f"the membrane below theta. The zero reset guarantees it; a "
            f"subtractive reset leaves m >= theta whenever one event carries "
            f">= 2*theta, and the deployed count would then depend on which "
            f"rows happen to be masked. Clear firing_mode to derive "
            f"{law.required_firing_mode!r}, or deploy per_cycle."
        )
    if bias_mode_for_cores(cfg.get("cores")) != _PARAM_ENCODED_BIAS:
        yield FIRING_GRANULARITY_KEY, (
            "firing_granularity='per_event' requires a param_encoded bias: the "
            "per-event fold consumes the bias as an always-on axon row at the "
            "declared slot, while an on-chip bias lane adds it once per cycle. "
            "Declare has_bias=false on the core grid, or deploy per_cycle."
        )
    init = cfg.get(_MEMBRANE_INIT_KEY, 0.0)
    try:
        value = float(init)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if not 0.0 <= value < 1.0:
        yield _MEMBRANE_INIT_KEY, (
            f"lif_membrane_init={value} is outside [0, 1) and "
            f"firing_granularity='per_event' needs a window-start membrane "
            f"inside the unit threshold window (the row-pair lemma's "
            f"precondition: a zero-magnitude event is a no-op only while the "
            f"membrane is below threshold). Set lif_membrane_init in [0, 1), "
            f"or deploy per_cycle."
        )
