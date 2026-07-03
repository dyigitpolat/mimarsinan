"""Backend interface + capability-validated registry keyed by name."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

__all__ = [
    "Backend",
    "SimulationBackend",
    "BackendRegistry",
    "BACKEND_REGISTRY",
]


def _spiking_mode_of(contract: Any) -> str:
    """The spiking mode of a contract / plan / raw mode string."""
    mode = getattr(contract, "spiking_mode", contract)
    return str(mode or "lif")


class Backend(ABC):
    """A capability-declared deployment target with a build/run/parity-gate driver seam."""

    name: str

    def supports(self, contract: Any) -> bool:
        """Whether this backend supports the contract's spiking mode (matrix-driven)."""
        return policy_for_spiking_mode(_spiking_mode_of(contract)).supports_backend(
            self.name
        )

    def require_supported(self, contract: Any, *, context: str) -> None:
        """Raise an actionable error if this backend cannot run the contract's mode."""
        policy_for_spiking_mode(_spiking_mode_of(contract)).require_backend_supported(
            backend=self.name, context=context
        )

    @abstractmethod
    def build(self, *args: Any, **kwargs: Any) -> Any:
        """Construct the per-sample runner / chip for this backend."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the built runner on a sample and return its record/metric."""

    @abstractmethod
    def parity_gate(self, *args: Any, **kwargs: Any) -> Any:
        """Check this backend's record against the HCM/golden reference."""


class SimulationBackend(Backend):
    """A simulation backend backed by an existing PipelineStep class.

    The step class IS the build+run+parity driver; the registry only routes
    selection + up-front validation. ``applies`` carries the enable predicate.
    """

    def __init__(
        self,
        name: str,
        *,
        step_name: str,
        step_class: type,
        enabled_for: Callable[[Any], bool],
        applies_for: Optional[Callable[[Any], bool]] = None,
        unsupported_error: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self.name = name
        self.step_name = step_name
        self.step_class = step_class
        self._enabled_for = enabled_for
        self._applies_for = applies_for
        self._unsupported_error = unsupported_error

    def enabled(self, plan: Any) -> bool:
        """Whether the plan's ``enable_*`` flag turns this backend on."""
        return bool(self._enabled_for(plan))

    def applies(self, plan: Any) -> bool:
        """Whether this backend's step is appended (enable flag + extra gates)."""
        if not self.enabled(plan):
            return False
        return True if self._applies_for is None else bool(self._applies_for(plan))

    def require_supported(self, contract: Any, *, context: str) -> None:
        """Validate the backend×mode, preferring a backend-specific error message."""
        if self.supports(contract):
            return
        if self._unsupported_error is not None:
            raise ValueError(self._unsupported_error(contract))
        super().require_supported(contract, context=context)

    def step_spec(self) -> tuple[str, type]:
        """The ``(step_name, step_class)`` this backend contributes."""
        return (self.step_name, self.step_class)

    def build(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.name} build is driven by {self.step_class.__name__}"
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.name} run is driven by {self.step_class.__name__}"
        )

    def parity_gate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.name} parity_gate is driven by {self.step_class.__name__}"
        )


class BackendRegistry:
    """Ordered registry of deployment backends, keyed by name."""

    def __init__(self, backends: list[Backend]) -> None:
        self._backends: list[Backend] = list(backends)
        self._by_name: dict[str, Backend] = {b.name: b for b in self._backends}

    def get(self, name: str) -> Backend:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown backend {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def all(self) -> tuple[Backend, ...]:
        return tuple(self._backends)

    def simulation_backends(self) -> tuple[SimulationBackend, ...]:
        return tuple(b for b in self._backends if isinstance(b, SimulationBackend))

    def selected_step_specs(self, plan: Any) -> list[tuple[str, type]]:
        """Validate every enabled backend against the capability matrix, then return its step specs.

        An enabled backend that does not support the plan's spiking mode raises here
        (at assembly, in registry order) before any step is appended.
        """
        for backend in self.simulation_backends():
            if backend.enabled(plan):
                backend.require_supported(plan, context=backend.step_name)
        return [
            backend.step_spec()
            for backend in self.simulation_backends()
            if backend.applies(plan)
        ]


def _loihi_ttfs_error(contract: Any) -> str:
    return (
        f"enable_loihi_simulation is not supported for "
        f"spiking_mode={_spiking_mode_of(contract)!r}; "
        "Loihi/Lava only implements LIF dynamics."
    )


def _build_default_registry() -> BackendRegistry:
    from mimarsinan.pipelining.pipeline_steps import (
        LoihiSimulationStep,
        SanafeSimulationStep,
        SimulationStep,
    )

    return BackendRegistry([
        SimulationBackend(
            "nevresim",
            step_name="Simulation",
            step_class=SimulationStep,
            enabled_for=lambda plan: plan.enable_nevresim_simulation,
            # nevresim has no genuine synchronized-window backend, so skip it only there.
            applies_for=lambda plan: not plan.is_synchronized_ttfs,
        ),
        SimulationBackend(
            "loihi",
            step_name="Loihi Simulation",
            step_class=LoihiSimulationStep,
            enabled_for=lambda plan: plan.enable_loihi_simulation,
            unsupported_error=_loihi_ttfs_error,
        ),
        SimulationBackend(
            "sanafe",
            step_name="SANA-FE Simulation",
            step_class=SanafeSimulationStep,
            enabled_for=lambda plan: plan.enable_sanafe_simulation,
        ),
    ])


class _LazyBackendRegistry:
    """Registry proxy that defers step-class imports until first use.

    Building the default registry eagerly would create an import cycle
    (chip_simulation ← pipelining ← chip_simulation); this resolves it on first access.
    """

    def __init__(self) -> None:
        self._registry: Optional[BackendRegistry] = None

    def _resolve(self) -> BackendRegistry:
        if self._registry is None:
            self._registry = _build_default_registry()
        return self._registry

    def __getattr__(self, item: str) -> Any:
        return getattr(self._resolve(), item)

    def __contains__(self, name: str) -> bool:
        return name in self._resolve()


BACKEND_REGISTRY: Any = _LazyBackendRegistry()
