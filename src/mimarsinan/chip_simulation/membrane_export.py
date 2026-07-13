"""[C2] Deployed membrane-readout honesty gate over backend export capability."""

from __future__ import annotations

from typing import Any, Mapping

from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
from mimarsinan.chip_simulation.spiking_semantics import is_lif


def backend_exports_final_membrane(name: str) -> bool:
    """Whether backend ``name`` can export end-of-window membrane potentials
    of output neurons (declared on the backend registry)."""
    return bool(BACKEND_REGISTRY.get(name).exports_final_membrane)


def enabled_backend_names(plan: Any) -> tuple[str, ...]:
    """Names of the simulation backends this plan enables, in registry order."""
    return tuple(
        backend.name
        for backend in BACKEND_REGISTRY.simulation_backends()
        if backend.enabled(plan)
    )


def half_step_charge_from_config(config: Mapping[str, Any]) -> float:
    """[C1] the half-step charge (in theta units) the wire-bias fold bakes into
    the terminal charge; the membrane decode removes exactly this much."""
    return 0.5 if bool(config.get("lif_half_step_bias", False)) else 0.0


def _backend_vetoes_membrane_decode(backend: Any) -> bool:
    """A backend vetoes the decode iff it produces a deployed ACCURACY read
    it cannot realize with membranes; parity-currency backends (Loihi/Lava,
    SANA-FE) compare per-neuron counts the logits decode never touches."""
    return bool(getattr(backend, "decodes_accuracy", False)) and not bool(
        getattr(backend, "exports_final_membrane", False)
    )


def deployed_membrane_readout_enabled(
    config: Mapping[str, Any], plan: Any,
) -> bool:
    """[C2] honesty gate for the deployed membrane-augmented logits decode.

    True only when ``lif_membrane_readout`` is armed, the mode is LIF, and
    every enabled ACCURACY-PRODUCING backend exports final membranes (the
    in-repo nevresim read port). Parity-only backends never veto: their
    compared currency is raw counts, untouched by the decode. A run with no
    enabled backend keeps the decode (realizable through nevresim).
    """
    if not bool(config.get("lif_membrane_readout", False)):
        return False
    if not is_lif(str(getattr(plan, "spiking_mode", "lif"))):
        return False
    return not any(
        _backend_vetoes_membrane_decode(BACKEND_REGISTRY.get(name))
        for name in enabled_backend_names(plan)
    )
