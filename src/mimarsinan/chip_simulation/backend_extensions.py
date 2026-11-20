"""[W15, W33] The execution extensions a backend must implement to run a plan: function-bearing contract fields the
reference executor (the HCM) honours beyond the semantics, the window, the comparator, the firing mode and the
encoder. Each is declared by its registry key, which the decision-class registry classes function-bearing
(``test_backend_extensions``). A backend declares the extensions it implements (``IMPLEMENTED_EXTENSIONS``), read by
the one capability judgment (``point_capability``) with its capability row: [U-LIF2 W3] a backend that does not
implement one a plan declares is derived off where the run is resolved, and the assembly check refuses it by name as
a guard, instead of running another function (measured: under phase dither and under a membrane guard, 15 of 16
programs differed on nevresim, SANA-FE and Lava, ``RA/11_wave2/exact/review_soundness.md`` M3)."""

from __future__ import annotations

from typing import Any, Callable, FrozenSet, Mapping, Tuple

from mimarsinan.chip_simulation.spiking_semantics import lif_config, lif_membrane_init, spike_phase_dither_enabled
from mimarsinan.common.refusal import ToolchainRefusal


#: Registry key -> whether a config declares the extension (its value away from the neutral one). Both act only on
#: LIF runs: the dither rotates uniform encode combs, the guard pre-charges the LIF membrane (``SpikingHybridCoreFlow``).
EXECUTION_EXTENSIONS: Mapping[str, Callable[[Mapping[str, Any]], bool]] = {
    "spike_phase_dither": lambda config: lif_config(config) and spike_phase_dither_enabled(config),
    "lif_membrane_init": lambda config: lif_config(config) and lif_membrane_init(config) != 0.0,
}


#: [U-LIF2 W3] The extensions each backend implements, the column the one capability judgment reads beside the backend's
#: capability row (``point_capability``): the torch executors run both as the HCM does; every other name implements none.
_TORCH_EXECUTORS = ("hcm", "unified", "hybrid", "training")
IMPLEMENTED_EXTENSIONS: Mapping[str, FrozenSet[str]] = {
    name: frozenset(EXECUTION_EXTENSIONS) for name in _TORCH_EXECUTORS}


def implemented_extensions(backend: str) -> FrozenSet[str]:
    """The execution extensions ``backend`` implements (none for a name no row declares)."""
    return IMPLEMENTED_EXTENSIONS.get(backend, frozenset())


class BackendExtensionRefusal(ToolchainRefusal, ValueError):
    """An enabled backend asked to run an execution extension it does not implement, refused by name."""


def declared_extensions(config: Mapping[str, Any]) -> Tuple[str, ...]:
    """The execution extensions ``config`` declares, by registry key, sorted."""
    return tuple(sorted(key for key, declared in EXECUTION_EXTENSIONS.items() if declared(config)))
