"""THE nevresim C++ policy type names: reset, compare, fire, integration.

One home for every string that becomes a C++ template argument in a generated
``main.cpp``. The reset law and the comparator each used to be resolved twice —
in ``code_generation/generate_main`` and on ``NeuralBehaviorConfig`` — with
*opposite* fallbacks for an unrecognized value ([D2]): one silently answered
``SubtractiveReset``, the other silently answered ``ZeroReset``, so an
unexpected firing mode produced a different chip depending on which caller
asked. Both call sites now delegate here, and an unknown value RAISES: a
typo in a semantics key must never resolve to somebody's physics.
"""

from __future__ import annotations

from typing import Optional

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_semantics import (
    DEFAULT_FIRING_MODE,
    NOVENA_FIRING_MODE,
    THRESHOLDING_MODES,
    TTFS_FIRING_MODE,
)

ZERO_RESET = "ZeroReset"
SUBTRACTIVE_RESET = "SubtractiveReset"
STRICT_COMPARE = "StrictCompare"
INCLUSIVE_COMPARE = "InclusiveCompare"

WHOLE_VECTOR_INTEGRATE = "WholeVectorIntegrate"
"""The default integration policy: one reduction per cycle, one compare. The
name a defaulted C++ template parameter already resolves to, so it is NEVER
emitted — an emitted argument would change the text of every existing
``main.cpp``."""

EVENT_SERIAL_INTEGRATE = "EventSerialIntegrate"
"""The event-serial fold on a saturating UNSIGNED register (stock ODIN)."""

WHOLE_VECTOR_SATURATING_SIGNED = "WholeVectorSaturatingSigned"
"""[ODIN P6] the sync-fire law: today's whole-vector reduction and single
compare, on a fixed-width TWO'S-COMPLEMENT register that asserts rather than
saturates — so it holds exactly the number the unbounded accumulator holds."""

#: Integration policies whose cycle can emit MORE than one spike per neuron.
#: The wire's alphabet is a consequence of the law, so it is declared here once
#: rather than inferred from "is it the default".
_COUNTED_POLICIES = frozenset({EVENT_SERIAL_INTEGRATE})

# The reset law each firing mode deploys. TTFS neurons never read this string
# (their compute policies take the comparator alone), but the codegen path
# builds it unconditionally, so the table is total over the firing-mode
# vocabulary and answers with the string the emitter has always produced.
_RESET_BY_FIRING_MODE = {
    NOVENA_FIRING_MODE: ZERO_RESET,
    DEFAULT_FIRING_MODE: SUBTRACTIVE_RESET,
    TTFS_FIRING_MODE: SUBTRACTIVE_RESET,
}

_COMPARE_BY_THRESHOLDING_MODE = {
    "<": STRICT_COMPARE,
    "<=": INCLUSIVE_COMPARE,
}


class NevresimPolicyTypeError(ValueError):
    """A semantics value nevresim has no C++ policy type for."""


def nevresim_reset_policy(firing_mode: str) -> str:
    """The C++ reset policy for ``firing_mode``."""
    try:
        return _RESET_BY_FIRING_MODE[str(firing_mode)]
    except KeyError:
        raise NevresimPolicyTypeError(
            f"no nevresim reset policy for firing_mode={firing_mode!r}; "
            f"known: {sorted(_RESET_BY_FIRING_MODE)}. The reset law decides "
            f"what the membrane holds after a spike, so an unrecognized "
            f"firing mode may not fall back to either answer."
        ) from None


def nevresim_compare_policy(thresholding_mode: str) -> str:
    """The C++ compare policy for ``thresholding_mode``."""
    try:
        return _COMPARE_BY_THRESHOLDING_MODE[str(thresholding_mode)]
    except KeyError:
        raise NevresimPolicyTypeError(
            f"no nevresim compare policy for "
            f"thresholding_mode={thresholding_mode!r}; known: "
            f"{list(THRESHOLDING_MODES)}. On an integer-lattice chip '<' and "
            f"'<=' are different physics at every exact tie."
        ) from None


def nevresim_lif_fire_policy(firing_mode: str, thresholding_mode: str) -> str:
    """The C++ ``LIFirePolicy`` instantiation for one (reset, compare) point."""
    return (
        f"LIFirePolicy<{nevresim_reset_policy(firing_mode)}, "
        f"{nevresim_compare_policy(thresholding_mode)}>"
    )


def nevresim_integration_policy(soma_law: Optional[SomaLaw]) -> str:
    """The C++ integration policy for one resolved soma point.

    The default point resolves to the name the C++ default template argument
    already carries, so nothing is emitted and every historical ``main.cpp``
    stays byte-identical. The event-serial fold is nevresim's ONE non-default
    executor and it is the saturating fixed-width one: the two half-declared
    points refuse by name rather than run somebody else's arithmetic.
    """
    if soma_law is None or soma_law.is_default_point:
        return WHOLE_VECTOR_INTEGRATE
    if soma_law.is_per_event and soma_law.is_signed_membrane:
        raise NevresimPolicyTypeError(
            f"nevresim folds events on a register that FLOORS at zero; the "
            f"point declares firing_granularity='per_event' with "
            f"membrane_arithmetic={soma_law.membrane_arithmetic!r}, and no "
            f"executor here folds events on a two's-complement membrane. The "
            f"signed register is the per-CYCLE sync-fire law."
        )
    if soma_law.is_per_event and soma_law.saturates:
        return f"{EVENT_SERIAL_INTEGRATE}<{int(soma_law.membrane_bits)}>"
    if soma_law.is_signed_membrane:
        return (
            f"{WHOLE_VECTOR_SATURATING_SIGNED}<{int(soma_law.membrane_bits)}>"
        )
    if soma_law.is_per_event:
        raise NevresimPolicyTypeError(
            f"nevresim runs the event-serial fold on a FIXED-WIDTH saturating "
            f"membrane; the point declares firing_granularity='per_event' "
            f"with membrane_arithmetic={soma_law.membrane_arithmetic!r} and "
            f"no register width, and an unbounded per-event accumulator is a "
            f"law no executor here implements. Declare membrane_bits."
        )
    raise NevresimPolicyTypeError(
        f"nevresim's per-cycle integration is the UNBOUNDED accumulator; the "
        f"point declares membrane_arithmetic={soma_law.membrane_arithmetic!r} "
        f"(membrane_bits={soma_law.membrane_bits}) with "
        f"firing_granularity={soma_law.firing_granularity!r}, and a "
        f"saturating per-cycle register is a law no executor here implements. "
        f"Declare firing_granularity='per_event', or drop the width."
    )


def integration_policy_base(integration_policy: str) -> str:
    """The policy's TEMPLATE name, without its register-width argument."""
    return str(integration_policy).split("<", 1)[0]


def counts_on_the_wire(integration_policy: str) -> bool:
    """Whether this integration policy can emit more than one spike per cycle.

    The wire's alphabet is a consequence of the LAW, never a separate switch:
    the counted raster, the counted carry seam and the cache sub-object all
    arm off this one predicate. It is NOT "is this the default policy" — the
    sync-fire register is a non-default law that still fires at most once per
    cycle, and binarizing its raster is lossless.
    """
    return integration_policy_base(integration_policy) in _COUNTED_POLICIES


def emits_integration_policy(integration_policy: str) -> bool:
    """Whether the emitter must NAME this policy in the generated ``main.cpp``.

    Only the C++ default template argument is omitted; naming it would change
    the text of every program that predates the axis.
    """
    return integration_policy_base(integration_policy) != WHOLE_VECTOR_INTEGRATE
