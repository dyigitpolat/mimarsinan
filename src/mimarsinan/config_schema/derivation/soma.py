"""The soma-axes fold and its cross-key contract (bits-driven, like WQ)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple

from mimarsinan.chip_simulation.soma_axes import (
    FIRING_GRANULARITY_KEY,
    MEMBRANE_ARITHMETIC_KEY,
    MEMBRANE_BITS_KEY,
    SATURATING_UNSIGNED_MEMBRANE,
    UNBOUNDED_MEMBRANE,
    resolved_firing_granularity,
    resolved_membrane_arithmetic,
    resolved_membrane_bits,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.platform.platform_constraints import bias_mode_for_cores

_PARAM_ENCODED_BIAS = "param_encoded"
_MEMBRANE_INIT_KEY = "lif_membrane_init"

# Which keys a violation's one-click remedy may clear, per offending key.
_REMEDY_KEYS: Dict[str, Tuple[str, ...]] = {
    MEMBRANE_ARITHMETIC_KEY: (MEMBRANE_ARITHMETIC_KEY, MEMBRANE_BITS_KEY),
    FIRING_GRANULARITY_KEY: (FIRING_GRANULARITY_KEY,),
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
            f"membrane_arithmetic={SATURATING_UNSIGNED_MEMBRANE!r} needs a "
            f"declared membrane_bits width: an unsigned register that "
            f"saturates has to say where. Declare platform membrane_bits, or "
            f"remove membrane_arithmetic."
        )
    if not law.is_per_event:
        return
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
