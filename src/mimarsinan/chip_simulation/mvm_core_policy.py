"""The value-domain (MVM) core policy: typed answers on exactly the consumed seams."""

from __future__ import annotations

from mimarsinan.chip_simulation.core_semantics import CORE_SEMANTICS_MVM
from mimarsinan.chip_simulation.spiking_mode_policy import SpikingModePolicy


class MvmCorePolicy(SpikingModePolicy):
    """Policy for value-domain MVM cores: pure weight-stationary matmul units.

    Implements the seams the pipeline consults on a value plan; every
    event-domain seam (decode_mode, calibration_forward, soma/nevresim
    codegen) stays the base class's loud ``NotImplementedError``.
    """

    def __init__(self) -> None:
        # [U-LIF2 W4] The value domain has no spiking point: the policy names its domain and carries no soma law,
        # and it overrides every point-keyed judgment of the base below (no spiking backend runs it).
        self.spiking_mode = CORE_SEMANTICS_MVM
        self.schedule = None
        self.soma_law = None

    def certification_observable(self) -> tuple[str, "str | None"]:
        """[cert-plan W3 shape] per-neuron post-affine VALUES, never counts."""
        return ("values", None)

    @property
    def runs_per_cycle_cascade(self) -> bool:
        """Value cores compute one matrix product per read: no cycle, no window."""
        return False

    def requires_activation_alignment(self) -> bool:
        """No on-chip activations exist, so the alignment ladder is inert."""
        return False

    def training_forward_kind(self) -> str:
        return "value"

    def supports_backend(self, backend: str) -> bool:
        """No spiking simulator runs the value family; the value twin is in-process."""
        del backend
        return False

    def require_backend_supported(self, *, backend: str, context: str) -> None:
        raise ValueError(
            f"{context}: backend {backend!r} cannot run the value-domain "
            f"(core_semantics='mvm') family — spiking simulators execute "
            f"event physics; the value twin executor carries the deployed read."
        )
