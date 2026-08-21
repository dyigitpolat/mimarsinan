"""Point-keyed backend capability: a backend refuses a soma law it cannot execute."""

from __future__ import annotations

from typing import Optional

from mimarsinan.chip_simulation.soma_axes import (
    PER_EVENT_FIRING,
    SATURATING_SIGNED_MEMBRANE,
    SATURATING_UNSIGNED_MEMBRANE,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_semantics import backend_capabilities


class BackendSomaLawError(ValueError):
    """A backend was asked to execute a soma law it does not implement."""


def unsupported_soma_law_reason(
    backend: str, soma_law: Optional[SomaLaw]
) -> Optional[str]:
    """WHICH declared axis value ``backend`` cannot execute, or ``None``.

    A missing law is the legacy mode-string surface: it carries no point, so
    it is judged by the mode matrix alone and never refused here.
    """
    if soma_law is None or soma_law.is_default_point:
        return None
    caps = backend_capabilities(backend)
    if soma_law.is_per_event and not caps.per_event_firing:
        return f"firing_granularity={PER_EVENT_FIRING!r}"
    # The signed register is asked about FIRST: it is also a saturating one,
    # and a refusal must name the value the config actually declares.
    if soma_law.is_signed_membrane:
        if not (caps.saturating_membrane and caps.signed_membrane):
            return (
                f"membrane_arithmetic={SATURATING_SIGNED_MEMBRANE!r} "
                f"(membrane_bits={soma_law.membrane_bits})"
            )
        return None
    if soma_law.saturates and not caps.saturating_membrane:
        return (
            f"membrane_arithmetic={SATURATING_UNSIGNED_MEMBRANE!r} "
            f"(membrane_bits={soma_law.membrane_bits})"
        )
    return None


def supports_soma_law(backend: str, soma_law: Optional[SomaLaw]) -> bool:
    """Whether ``backend`` declares an executor for this resolved point."""
    return unsupported_soma_law_reason(backend, soma_law) is None


def require_soma_law_supported(
    soma_law: Optional[SomaLaw], *, backend: str, context: str
) -> None:
    """Refuse the point BY NAME, naming the backend and the axis value."""
    reason = unsupported_soma_law_reason(backend, soma_law)
    if reason is None:
        return
    raise BackendSomaLawError(
        f"{context}: backend {backend!r} does not implement {reason} — no "
        f"executor runs this soma law, so the run would report a DIFFERENT "
        f"physics as the deployed number. Remove the declaration to deploy "
        f"the per-cycle unbounded law, or run a backend that declares it."
    )
