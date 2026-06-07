"""Cross-side deployment-semantics SSOT shared by torch NF and chip simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.spiking_semantics import (
    is_analytical_ttfs,
    is_cascaded_ttfs,
    is_synchronized_ttfs,
    ttfs_cycle_schedule,
)


@dataclass(frozen=True)
class SpikingDeploymentContract:
    """One object answering every schedule/mode-derived deployment question.

    ``from_pipeline_config`` is the ONLY place these config keys are read;
    everything downstream takes the contract. Derived getters accept a reserved
    ``core=None`` kwarg so per-core heterogeneity can land without re-plumbing.
    """

    behavior: NeuralBehaviorConfig
    simulation_steps: int
    ttfs_cycle_schedule: str
    encoding_layer_placement: str
    bias_mode: str

    # ── identity axes (delegated to the composed NeuralBehaviorConfig) ──────
    @property
    def spiking_mode(self) -> str:
        return self.behavior.spiking_mode

    @property
    def firing_mode(self) -> str:
        return self.behavior.firing_mode

    @property
    def thresholding_mode(self) -> str:
        return self.behavior.thresholding_mode

    @property
    def spike_generation_mode(self) -> str:
        return self.behavior.spike_generation_mode

    @property
    def spike_encoding_seed(self) -> int | None:
        return self.behavior.spike_encoding_seed

    @classmethod
    def from_pipeline_config(cls, cfg: dict[str, Any]) -> "SpikingDeploymentContract":
        from mimarsinan.pipelining.core.platform_constraints_resolver import (
            resolve_bias_mode,
        )

        return cls(
            behavior=NeuralBehaviorConfig.from_deployment_config(cfg),
            simulation_steps=int(cfg["simulation_steps"]),
            ttfs_cycle_schedule=ttfs_cycle_schedule(cfg.get("ttfs_cycle_schedule")),
            encoding_layer_placement=str(
                cfg.get("encoding_layer_placement", "subsume")
            ),
            bias_mode=resolve_bias_mode(cfg),
        )

    # ── derived behavior (the D4 killer) ────────────────────────────────────
    def is_synchronized(self, *, core: Any = None) -> bool:
        return is_synchronized_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)

    def is_cascaded(self, *, core: Any = None) -> bool:
        return is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)

    def quantize_stage_input_to_grid(self, *, core: Any = None) -> bool:
        """The synchronized wire rule q(x), decided HERE, not per caller."""
        return self.is_synchronized(core=core)

    def wire(self, *, core: Any = None):
        """Wire-op kernel bundle (staircase / spike-time / grid-snap twins)."""
        from mimarsinan.models.spiking.wire_semantics import WireSemantics

        return WireSemantics(
            simulation_steps=self.simulation_steps,
            compare_mode=self.thresholding_mode,
        )

    def training_forward_kind(self, *, core: Any = None) -> str:
        """NF algorithm the fine-tuners must train through for this deployment.

        ``segment_spike``: cascaded single-spike segment walk.
        ``analytical_staircase``: staircase composition (synchronized /
        analytical TTFS). ``lif_cycle`` / ``rate``: the LIF-family forwards.
        """
        if self.is_cascaded(core=core):
            return "segment_spike"
        if self.is_synchronized(core=core) or is_analytical_ttfs(self.spiking_mode):
            return "analytical_staircase"
        if self.spiking_mode == "rate":
            return "rate"
        return "lif_cycle"
