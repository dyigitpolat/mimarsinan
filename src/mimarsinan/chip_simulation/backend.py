"""Backend interface + capability-validated registry (Vector V3).

A deployment ``Backend`` is a named target whose *capabilities* are declared in
the ``_BACKEND_CAPS`` matrix (``spiking_semantics.py``). Until V3 that matrix was
**informational**: backend selection happened through ``enable_*`` flags and
validation was reactive inside ``step.process()`` (the path the SANA-FE C++
incompatibility took before it core-dumped). V3 makes selection + validation
*consult the matrix up-front at pipeline assembly*:

- ``Backend.supports(contract)`` answers "does this backend support this spiking
  mode" by reading the same ``_BACKEND_CAPS`` matrix.
- ``Backend.require_supported(contract, context=...)`` raises an actionable error
  at assembly for an unsupported backend×mode — never reactively mid-run.
- ``Backend.build`` / ``run`` / ``parity_gate`` are the documented per-target
  driver seam; the existing simulation *step classes* (nevresim / SANA-FE /
  Loihi) remain the implementations the runner drives, registered here.

``BackendRegistry`` maps backend name → ``Backend`` and resolves which backends
a ``DeploymentPlan`` enables (the verbatim ``enable_*`` precedence). The pipeline
step planner consults it to validate-then-append, so adding a backend is one
registry entry, not scattered ``if enable_*`` edits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from mimarsinan.chip_simulation.spiking_semantics import (
    require_spiking_mode_supported,
    supports_spiking_mode,
)

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
    """A capability-declared deployment target.

    Concrete backends declare a ``name`` (the key in the ``_BACKEND_CAPS``
    capability matrix) and implement the build/run/parity-gate driver seam.
    ``supports`` / ``require_supported`` consult the matrix; the registry calls
    ``require_supported`` at assembly so an unsupported backend×mode fails loud
    before any expensive setup.
    """

    name: str

    def supports(self, contract: Any) -> bool:
        """Whether this backend supports the contract's spiking mode (matrix-driven)."""
        return supports_spiking_mode(self.name, _spiking_mode_of(contract))

    def require_supported(self, contract: Any, *, context: str) -> None:
        """Raise an actionable error if this backend cannot run the contract's mode."""
        require_spiking_mode_supported(
            _spiking_mode_of(contract), backend=self.name, context=context
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
    """A pipeline simulation backend backed by an existing PipelineStep class.

    The step class IS the build+run+parity driver (verbatim, unchanged): V3 keeps
    them as the implementations and only routes *selection + up-front validation*
    through the registry. ``applies`` carries the backend's enable predicate (the
    former ``if plan.enable_*`` condition) so the step planner reads data, not a
    per-backend ``if``.
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

    # ── driver seam (implemented by the step class, delegated for the ABC) ──
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
        """Validate every enabled backend up-front, then return its step specs.

        Mirrors the pre-V3 ``if plan.enable_*: specs.append(...)`` block but with
        the capability matrix consulted *first*: an enabled backend that does not
        support the plan's spiking mode raises here (at assembly), in registry
        order, before any step is appended — instead of crashing mid-run.

        Only the plan's ``spiking_mode`` is needed to gate against the capability
        matrix, so the plan is consulted directly — the full spiking contract
        (which needs ``simulation_steps``) is not built at step-ordering time.
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
            # nevresim runs the genuine fire-once-latch cascade, but has no
            # genuine synchronized-window backend yet — skip it only there.
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

    The default registry references pipeline step classes; building it eagerly at
    import time would create an import cycle (chip_simulation ← pipelining ←
    chip_simulation). This proxy resolves the concrete registry on first access.
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
