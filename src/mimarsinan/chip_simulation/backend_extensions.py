"""[W15, W33] The execution extensions a backend must implement to run a plan: function-bearing contract fields the
reference executor (the HCM) honours beyond the semantics, the window, the comparator, the firing mode and the
encoder. Each is declared by its registry key, which the decision-class registry classes function-bearing
(``test_backend_extensions``). A backend declares the extensions it implements; an enabled backend that does not
implement one a plan declares refuses the plan by name at assembly instead of running another function (measured:
under phase dither and under a membrane guard, 15 of 16 programs differed on nevresim, SANA-FE and Lava,
``RA/11_wave2/exact/review_soundness.md`` M3)."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Tuple

from mimarsinan.chip_simulation.spiking_semantics import lif_config, lif_membrane_init, spike_phase_dither_enabled
from mimarsinan.common.refusal import ToolchainRefusal


#: Registry key -> whether a config declares the extension (its value away from the neutral one). Both act only on
#: LIF runs: the dither rotates uniform encode combs, the guard pre-charges the LIF membrane (``SpikingHybridCoreFlow``).
EXECUTION_EXTENSIONS: Mapping[str, Callable[[Mapping[str, Any]], bool]] = {
    "spike_phase_dither": lambda config: lif_config(config) and spike_phase_dither_enabled(config),
    "lif_membrane_init": lambda config: lif_config(config) and lif_membrane_init(config) != 0.0,
}


class BackendExtensionRefusal(ToolchainRefusal, ValueError):
    """An enabled backend asked to run an execution extension it does not implement, refused by name."""


def declared_extensions(config: Mapping[str, Any]) -> Tuple[str, ...]:
    """The execution extensions ``config`` declares, by registry key, sorted."""
    return tuple(sorted(key for key, declared in EXECUTION_EXTENSIONS.items() if declared(config)))


def require_extensions(backend: str, implements, config: Mapping[str, Any], *, context: str) -> None:
    """Refuse by name a plan that declares an extension ``backend`` does not implement."""
    missing = [key for key in declared_extensions(config) if key not in implements]
    if missing:
        raise BackendExtensionRefusal(
            f"{context}: backend {backend!r} does not implement the declared execution extension(s) {missing} "
            f"(the HCM does): it would run another function. Disable the backend (its enable_* key) or the "
            f"extension")
