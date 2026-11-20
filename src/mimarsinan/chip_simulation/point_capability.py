"""[U-LIF2 W3, W5] Whether a backend executes a resolved spiking point: one judgment over the backend's capability row
(the mode columns, the soma-law columns, the window-firing column) and the execution extensions it implements.

Every capability question reads it: the conversion recipe derives a backend that does not execute the point OFF with
the reason this judgment names, and the registry's assembly check and every runner refuse through
``require_backend_executes`` as guards (a derived config never reaches them with a point its backends refuse)."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from mimarsinan.chip_simulation.backend_extensions import BackendExtensionRefusal, implemented_extensions
from mimarsinan.chip_simulation.soma_capability import require_soma_law_supported, unsupported_soma_law_reason
from mimarsinan.chip_simulation.spiking_point import SpikingPoint, UnresolvedPointRefusal
from mimarsinan.chip_simulation.spiking_semantics import (
    backend_capabilities,
    require_spiking_mode_supported,
    supports_spiking_mode,
)
from mimarsinan.chip_simulation.window_firing import WINDOW_FIRING_KEY
from mimarsinan.common.refusal import ToolchainRefusal


class BackendWindowFiringRefusal(ToolchainRefusal, ValueError):
    """A backend asked to execute a window firing its capability row does not declare, refused by name."""


def _unrun_window_firing(backend: str, point: SpikingPoint) -> Optional[str]:
    value = point.window_firing
    if value is None or value in backend_capabilities(backend).window_firings:
        return None
    return f"{WINDOW_FIRING_KEY}={value!r}"


def _unrun_extensions(backend: str, point: SpikingPoint) -> list:
    implements = implemented_extensions(backend)
    return [key for key in point.execution_extensions if key not in implements]


def require_resolved_point(point: object, *, context: str) -> SpikingPoint:
    """[U-LIF2 r2] ``point`` itself when it is a resolved point; anything else (a mode alone) is refused by name: no
    capability judgment and no policy build takes a point the run did not resolve."""
    if not isinstance(point, SpikingPoint):
        raise UnresolvedPointRefusal(
            f"{context} takes the run's resolved point (a plan, a contract or a SpikingPoint), never a mode alone: "
            f"got {point!r}")
    return point


def backend_refusal_reason(backend: str, point: SpikingPoint) -> Optional[str]:
    """WHICH value of ``point`` ``backend`` does not execute (the mode, a soma axis, the window firing or a declared
    execution extension, in that order), or ``None`` when it executes the whole point."""
    point = require_resolved_point(point, context="a backend question")
    if not supports_spiking_mode(backend, point.spiking_mode):
        return f"spiking_mode={point.spiking_mode!r}"
    soma = unsupported_soma_law_reason(backend, point.soma_law)
    if soma is not None:
        return soma
    window = _unrun_window_firing(backend, point)
    if window is not None:
        return window
    missing = _unrun_extensions(backend, point)
    return f"the declared execution extension(s) {missing}" if missing else None


#: The backend each capability-derived enable turns on, in the recipe's order. [N5 2026-08-09] streamed Loihi is enabled:
#: the wave runner's per-core replay is free-running-equivalent on gap-1 graphs (relays guarantee gap 1; per-sample
#: window resets; the integer-theta lattice makes vth exactly representable; the step's spike-parity gate is FATAL).
#: [ODIN P7a/P8] the point-keyed capability decides whether the crossbar executes a law; the opt-in set decides a run
#: reaches for the hardware (or freezes a bundle for it) only when asked.
SIM_ENABLE_BACKENDS: Mapping[str, str] = {
    "enable_nevresim_simulation": "nevresim",
    "enable_sanafe_simulation": "sanafe",
    "enable_loihi_simulation": "loihi",
    "enable_odin_fpga_simulation": "odin_fpga",
    "enable_odin_hacc_export": "odin_hacc",
}
#: nevresim has no genuine synchronized-window (TTFS) backend: the one enable a mode rule turns off beside the row.
_NEVRESIM_SYNCHRONIZED_TTFS = "spiking_variant='synchronized' of the ttfs family (no synchronized-window backend)"


def sim_enable_refusals(point: SpikingPoint) -> Dict[str, Optional[str]]:
    """Each capability-derived enable key with the value of ``point`` its backend does not execute (``None``: the
    backend executes the whole point): the recipe's enables and the surface's WHY texts read this one table."""
    refusals = {key: backend_refusal_reason(backend, point) for key, backend in SIM_ENABLE_BACKENDS.items()}
    if point.is_synchronized_ttfs and refusals["enable_nevresim_simulation"] is None:
        refusals["enable_nevresim_simulation"] = _NEVRESIM_SYNCHRONIZED_TTFS
    return refusals


def executor_refusal_message(context: str, backend: str, what: str) -> str:
    """The one refusal text of a backend that does not execute a declared value of the point."""
    return (f"{context}: backend {backend!r} does not execute {what}: it would run another function than the one the "
            f"run declares. Derive the backend off (drop its explicit enable) or declare a point it executes")


def require_backend_executes(point: SpikingPoint, *, backend: str, context: str) -> None:
    """Refuse by name, as a guard, a point ``backend`` does not execute: the mode and the soma law with their own typed
    refusals, the window firing and the execution extensions with the one refusal text."""
    point = require_resolved_point(point, context=context)
    require_spiking_mode_supported(point.spiking_mode, backend=backend, context=context)
    require_soma_law_supported(point.soma_law, backend=backend, context=context)
    window = _unrun_window_firing(backend, point)
    if window is not None:
        raise BackendWindowFiringRefusal(executor_refusal_message(context, backend, window))
    missing = _unrun_extensions(backend, point)
    if missing:
        raise BackendExtensionRefusal(executor_refusal_message(
            context, backend, f"the declared execution extension(s) {missing} (the HCM implements them)"))
