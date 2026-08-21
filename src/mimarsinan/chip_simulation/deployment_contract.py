"""Cross-side deployment-semantics SSOT shared by torch NF and chip simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.activation_semantics import is_streamed_lif
from mimarsinan.chip_simulation.soma_axes import (
    PER_CYCLE_FIRING,
    UNBOUNDED_MEMBRANE,
    resolved_firing_granularity,
    resolved_membrane_arithmetic,
    resolved_membrane_bits,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.support.schedule.pass_cut import transfer_for
from mimarsinan.chip_simulation.spiking_semantics import (
    is_cascaded_ttfs,
    is_synchronized_ttfs,
    lif_execution_synchronized,
    lif_membrane_init,
    spike_phase_dither_enabled,
    ttfs_cycle_schedule,
    uses_ttfs_floor_ceil_convention,
)
from mimarsinan.models.nn.activations.autograd import (
    ChipInputQuantizer,
    TTFSInputGridQuantizer,
)
from mimarsinan.models.spiking.wire_semantics import WireSemantics
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.compute_boundary import normalize_boundary_value


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
    # Ops bound, not semantics: the wall cap simulator backends run under.
    simulation_step_timeout_s: float | None = None
    # The evaluator that decides host-op staircase ties. A hosted ComputeOp
    # (a subsumed encoding layer, say) ends in a staircase of step theta/T, so
    # a pre-activation within f32 reduction noise of a step edge resolves to
    # DIFFERENT steps on cuda and cpu — and one step is exactly one spike after
    # the segment-boundary transcode. Every backend that re-derives a host op
    # must therefore evaluate it where the census/HCM reference did.
    host_compute_device: Any = None
    # [E3] carry the +θ/(2S) mid-tread offset in the compare ladder, not the bias.
    comparator_half_step: bool = False
    # [calculus 15.11] deployed-composition physics: decorrelated encode combs
    # and the window-start membrane guard (normalized threshold units).
    spike_phase_dither: bool = False
    lif_membrane_init: float = 0.0
    lif_execution_synchronized: bool = False
    # [P3] end-to-end event-streamed LIF: no boundary normalization between
    # encode and readout; NF↔SCM holds EXACTLY (atol=0), single-program only.
    lif_streamed: bool = False
    # [ODIN P1] the soma axes: WHEN the threshold is evaluated and WHAT the
    # membrane's arithmetic is. The defaults are today's law exactly; consumers
    # take ``soma_law()``, never these fields.
    firing_granularity: str = PER_CYCLE_FIRING
    membrane_arithmetic: str = UNBOUNDED_MEMBRANE
    membrane_bits: int = 0

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

        timeout = cfg.get("simulation_step_timeout_s")
        return cls(
            behavior=NeuralBehaviorConfig.from_deployment_config(cfg),
            simulation_steps=int(cfg["simulation_steps"]),
            ttfs_cycle_schedule=ttfs_cycle_schedule(cfg.get("ttfs_cycle_schedule")),
            encoding_layer_placement=str(
                cfg.get("encoding_layer_placement", "subsume")
            ),
            bias_mode=resolve_bias_mode(cfg),
            simulation_step_timeout_s=float(timeout) if timeout is not None else None,
            comparator_half_step=bool(cfg.get("comparator_half_step", False)),
            spike_phase_dither=spike_phase_dither_enabled(cfg),
            lif_membrane_init=lif_membrane_init(cfg),
            lif_execution_synchronized=lif_execution_synchronized(cfg),
            lif_streamed=is_streamed_lif(cfg),
            firing_granularity=resolved_firing_granularity(cfg),
            membrane_arithmetic=resolved_membrane_arithmetic(cfg),
            membrane_bits=resolved_membrane_bits(cfg),
            host_compute_device=cfg.get("device"),
        )

    def soma_law(self, *, core: Any = None) -> SomaLaw:
        """The resolved per-neuron firing law this deployment executes."""
        del core
        return SomaLaw.resolve(self)

    def is_streamed_lif(self, *, core: Any = None) -> bool:
        return self.lif_streamed

    def pass_boundary_transfer(self, *, core: Any = None) -> str:
        """What a pass boundary INSIDE a neural segment owes this semantics.

        A pass is a physical unit (one chip program); a segment is a semantic one.
        Streamed execution has no interior transcode, so its pass boundaries must
        replay the raster verbatim — collapsing to counts there would be the
        windowed transcode wearing a scheduling hat. The windowed disciplines
        normalize timing at every boundary anyway, so a pass costs them nothing.
        """
        return transfer_for(streamed=self.is_streamed_lif(core=core))

    def is_synchronized(self, *, core: Any = None) -> bool:
        return is_synchronized_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)

    def is_cascaded(self, *, core: Any = None) -> bool:
        return is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)

    def uses_ttfs_floor_ceil_convention(self, *, core: Any = None) -> bool:
        """Modes that train the floor + half-step-bias convention and deploy the ceil TTFS kernel."""
        return uses_ttfs_floor_ceil_convention(
            self.spiking_mode, self.ttfs_cycle_schedule
        )

    def quantize_stage_input_to_grid(self, *, core: Any = None) -> bool:
        """The synchronized wire rule q(x), decided HERE, not per caller."""
        return self.is_synchronized(core=core)

    def wire(self, *, core: Any = None):
        """Wire-op kernel bundle (staircase / spike-time / grid-snap twins)."""
        return WireSemantics(
            simulation_steps=self.simulation_steps,
            compare_mode=self.thresholding_mode,
            comparator_half_step=self.comparator_half_step,
        )

    def mode_policy(self, *, core: Any = None):
        """The behavior-carrying ``SpikingModePolicy`` for this (firing × sync × point)."""
        return policy_for_spiking_mode(
            self.spiking_mode, self.ttfs_cycle_schedule,
            soma_law=self.soma_law(core=core),
        )

    def boundary_config(
        self, *, cycle_accurate: bool, core: Any = None
    ) -> BoundaryConfig:
        """The segment-boundary encode/decode config for this wire."""
        return BoundaryConfig(
            simulation_length=self.simulation_steps,
            spiking_mode=self.spiking_mode,
            cycle_accurate=cycle_accurate,
            spike_mode=self.spike_generation_mode,
            thresholding_mode=self.thresholding_mode,
            firing_mode=self.firing_mode,
            phase_dither=self.spike_phase_dither,
        )

    def entry_quantizer(self, theta, *, core: Any = None) -> ChipInputQuantizer:
        """The trained entry op == the deployed seam composition for this wire
        (synchronized TTFS snaps to the grid; rate/LIF rounds). Sigma-free:
        the negative-boundary shift is walk-applied and bias-baked."""
        cls = (
            TTFSInputGridQuantizer
            if self.is_synchronized(core=core)
            else ChipInputQuantizer
        )
        return cls(T=self.simulation_steps, activation_scale=theta)

    def seam_transcode(self, *, core: Any = None):
        """The one boundary value->wire transcode kernel (SSOT)."""
        return normalize_boundary_value

    def calibration_pipeline(self, config, *, distmatch_driven=False, core: Any = None):
        """The conversion-health ``CalibrationPipeline`` for this (firing × sync) cell.

        The ENABLE is the contract's (firing × sync) decision (the cascaded cycle opts
        in; LIF / analytical / synchronized get the inert pipeline).
        """
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        return CalibrationPipeline.for_mode(
            config,
            mode_policy=self.mode_policy(core=core),
            distmatch_driven=distmatch_driven,
        )

    def training_forward_kind(self, *, core: Any = None) -> str:
        """NF algorithm the fine-tuners must train through for this deployment.

        ``segment_spike``: cascaded single-spike segment walk.
        ``analytical_staircase``: staircase composition (synchronized /
        analytical TTFS). ``lif_cycle`` / ``rate``: the LIF-family forwards.
        """
        return self.mode_policy(core=core).training_forward_kind()
