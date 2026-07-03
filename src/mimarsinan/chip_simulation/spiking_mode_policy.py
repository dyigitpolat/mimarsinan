"""One behavior-carrying policy per (firing × sync) spiking mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_analytical_ttfs,
    is_synchronized_ttfs,
    is_ttfs_cycle_based,
    require_known_spiking_mode,
    require_spiking_mode_supported,
    requires_ttfs_firing,
    supports_spiking_mode,
    ttfs_cycle_schedule,
)

__all__ = [
    "SpikingModePolicy",
    "LifModePolicy",
    "TtfsAnalyticalModePolicy",
    "TtfsSyncCycleModePolicy",
    "TtfsCascadeModePolicy",
    "policy_for_spiking_mode",
    "ExecPolicySpec",
    "NevresimExecParams",
]


@dataclass(frozen=True)
class ExecPolicySpec:
    """C++ compute policy type and execution alias for nevresim main.cpp."""

    compute_policy: str
    exec_decl: str


@dataclass(frozen=True)
class NevresimExecParams:
    """Codegen scalars for one nevresim ``main.cpp`` exec policy.

    Comparator/reset strings (selected by the firing-strategy SSOT, never here)
    plus the chip-shape constants the C++ templates need.
    """

    compare: str
    lif_fire_policy: str
    spike_gen_mode: str
    weight_type: str
    simulation_length: int
    latency: int
    output_count: int

    @property
    def spike_generator(self) -> str:
        return f"{self.spike_gen_mode}SpikeGenerator"


class SpikingModePolicy:
    """Behavior carried per (firing × sync) deployment mode.

    Concrete subclasses override the firing-/schedule-derived behavior; the base
    resolves only the shared capability gate. ``spiking_mode`` captures the
    sub-variation (``ttfs`` vs ``ttfs_quantized``); ``schedule`` is the normalized
    ``ttfs_cycle_schedule`` for the TTFS-cycle family (``None`` otherwise).
    """

    def __init__(self, spiking_mode: str, schedule: Optional[str] = None) -> None:
        self.spiking_mode = spiking_mode
        self.schedule = schedule

    @classmethod
    def from_contract(cls, contract: Any) -> "SpikingModePolicy":
        return policy_for_spiking_mode(
            contract.spiking_mode, contract.ttfs_cycle_schedule
        )

    def training_forward_kind(self) -> str:
        """NF algorithm the fine-tuners must train through for this deployment."""
        raise NotImplementedError

    def calibration_forward(self):
        """NF forward that produces this mode's boundary values for calibration."""
        raise NotImplementedError

    def decode_mode(self) -> str:
        """How a segment's output is decoded: ``count`` (rate) vs ``timing`` (TTFS)."""
        raise NotImplementedError

    def nevresim_exec_policy(self, params: "NevresimExecParams") -> "ExecPolicySpec":
        """The nevresim ``(ComputePolicy, Execution)`` C++ types for this mode."""
        raise NotImplementedError

    @property
    def single_step_activation_replacement(self) -> bool:
        """Whether this mode has a dedicated LIF/TTFS-cycle tuning step."""
        return False

    @property
    def does_conversion_health_calibration(self) -> bool:
        """Whether this (firing × sync) cell runs the conversion-health calibration steps.

        Default inert (byte-identical for LIF, analytical, synchronized cycle); only
        the cell whose deployed cascade needs depth-attenuation/distribution correction
        overrides it.
        """
        return False

    def soma_hw_name(self) -> str:
        """Name of the SANA-FE soma plugin for this mode."""
        raise NotImplementedError

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        """The ``model_attributes`` dict for one SANA-FE neuron of this mode."""
        raise NotImplementedError

    @property
    def log_potential(self) -> bool:
        """Whether SANA-FE must log the membrane potential trace (TTFS V-decode)."""
        return False

    @property
    def requires_ttfs_firing(self) -> bool:
        return requires_ttfs_firing(self.spiking_mode)

    def supports_backend(self, backend: str) -> bool:
        """Whether ``backend``'s capabilities support this mode."""
        return supports_spiking_mode(backend, self.spiking_mode)

    def require_backend_supported(self, *, backend: str, context: str) -> None:
        """Raise an actionable error if ``backend`` cannot run this mode."""
        require_spiking_mode_supported(
            self.spiking_mode, backend=backend, context=context
        )

    def valid_backends(self, candidates) -> tuple[str, ...]:
        """Subset of ``candidates`` whose capabilities support this mode."""
        return tuple(b for b in candidates if self.supports_backend(b))


class LifModePolicy(SpikingModePolicy):
    """LIF family: per-cycle integrate-and-fire, count decode."""

    @property
    def single_step_activation_replacement(self) -> bool:
        return True

    def training_forward_kind(self) -> str:
        return "lif_cycle"

    def calibration_forward(self):
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

        return chip_aligned_segment_forward

    def decode_mode(self) -> str:
        return "count"

    def nevresim_exec_policy(self, params: NevresimExecParams) -> ExecPolicySpec:
        lif = params.lif_fire_policy
        return ExecPolicySpec(
            compute_policy=f"SpikingCompute<{lif}>",
            exec_decl=(
                f"using exec = SpikingExecution<"
                f"{params.simulation_length}, {params.latency}, "
                f"{params.output_count}, "
                f"{params.spike_generator}, {params.weight_type}, "
                f"{lif}>;"
            ),
        )

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import SOMA_LIF_NAME

        return SOMA_LIF_NAME

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import lif_model_attributes

        return lif_model_attributes(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
            firing_mode=firing_mode,
        )


class TtfsAnalyticalModePolicy(SpikingModePolicy):
    """Closed-form analytical TTFS (``ttfs`` continuous / ``ttfs_quantized``)."""

    def training_forward_kind(self) -> str:
        return "analytical_staircase"

    def calibration_forward(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            _analytical_segment_calibration_forward,
        )

        return _analytical_segment_calibration_forward

    def decode_mode(self) -> str:
        return "timing"

    @property
    def _is_quantized(self) -> bool:
        return forces_activation_quantization(self.spiking_mode)

    def nevresim_exec_policy(self, params: NevresimExecParams) -> ExecPolicySpec:
        if not self._is_quantized:
            return ExecPolicySpec(
                compute_policy="TTFSAnalyticalCompute",
                exec_decl="using exec = TTFSContinuousExecution;",
            )
        return ExecPolicySpec(
            compute_policy=(
                f"TTFSQuantizedCompute<{params.simulation_length}, "
                f"{params.compare}>"
            ),
            exec_decl=(
                f"using exec = TTFSExecution<"
                f"{params.simulation_length}, {params.latency}, "
                f"{params.compare}>;"
            ),
        )

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import (
            SOMA_TTFS_CONTINUOUS_NAME,
            SOMA_TTFS_QUANTIZED_NAME,
        )

        return (
            SOMA_TTFS_QUANTIZED_NAME if self._is_quantized else SOMA_TTFS_CONTINUOUS_NAME
        )

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_continuous_model_attributes,
            ttfs_quantized_model_attributes,
        )

        builder = (
            ttfs_quantized_model_attributes
            if self._is_quantized
            else ttfs_continuous_model_attributes
        )
        return builder(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
        )

    @property
    def log_potential(self) -> bool:
        return True


class _TtfsCycleModePolicy(SpikingModePolicy):
    """Shared base for the genuine single-spike (ttfs_cycle_based) schedules."""

    @property
    def single_step_activation_replacement(self) -> bool:
        return True

    def calibration_forward(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            _ttfs_segment_calibration_forward,
        )

        return _ttfs_segment_calibration_forward


class TtfsSyncCycleModePolicy(_TtfsCycleModePolicy):
    """Genuine single-spike, synchronized schedule (V-reconstruction soma)."""

    def training_forward_kind(self) -> str:
        return "analytical_staircase"

    def decode_mode(self) -> str:
        return "timing"

    def nevresim_exec_policy(self, params: NevresimExecParams) -> ExecPolicySpec:
        raise ValueError(
            "nevresim does not run the synchronized ttfs_cycle_based schedule; "
            "any nevresim run of ttfs_cycle_based is the cascaded schedule"
        )

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import SOMA_TTFS_CYCLE_NAME

        return SOMA_TTFS_CYCLE_NAME

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_cycle_model_attributes,
        )

        return ttfs_cycle_model_attributes(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
        )

    @property
    def log_potential(self) -> bool:
        return True


class TtfsCascadeModePolicy(_TtfsCycleModePolicy):
    """Genuine fire-once-latch, cascaded schedule (count-based decode, LIF-like)."""

    @property
    def does_conversion_health_calibration(self) -> bool:
        return True

    def training_forward_kind(self) -> str:
        return "segment_spike"

    def decode_mode(self) -> str:
        return "count"

    def nevresim_exec_policy(self, params: NevresimExecParams) -> ExecPolicySpec:
        return ExecPolicySpec(
            compute_policy=f"TTFSCascadeCompute<{params.compare}>",
            exec_decl=(
                f"using exec = TTFSCascadeExecution<"
                f"{params.simulation_length}, {params.latency}, "
                f"{params.output_count}, "
                f"{params.spike_generator}, {params.weight_type}, "
                f"{params.compare}>;"
            ),
        )

    def soma_hw_name(self) -> str:
        from mimarsinan.chip_simulation.sanafe.presets import SOMA_TTFS_CASCADE_NAME

        return SOMA_TTFS_CASCADE_NAME

    def soma_model_attributes(
        self,
        *,
        threshold: float,
        hardware_bias: Optional[float] = None,
        active_start: Optional[int] = None,
        active_length: Optional[int] = None,
        firing_mode: str = "Default",
    ) -> dict:
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_cascade_model_attributes,
        )

        return ttfs_cascade_model_attributes(
            threshold=threshold,
            hardware_bias=hardware_bias,
            active_start=active_start,
            active_length=active_length,
        )

    @property
    def log_potential(self) -> bool:
        return False


def policy_for_spiking_mode(
    spiking_mode: str, schedule=None
) -> SpikingModePolicy:
    """SSOT dispatch resolving the ``(firing × sync)`` policy for ``(spiking_mode, schedule)``.

    Rejects modes outside ``ALL_SPIKING_MODES`` (incl. the removed ``rate``).
    """
    mode = require_known_spiking_mode(spiking_mode)
    if is_ttfs_cycle_based(mode):
        if is_synchronized_ttfs(mode, schedule):
            return TtfsSyncCycleModePolicy(mode, ttfs_cycle_schedule(schedule))
        return TtfsCascadeModePolicy(mode, ttfs_cycle_schedule(schedule))
    if is_analytical_ttfs(mode):
        return TtfsAnalyticalModePolicy(mode)
    return LifModePolicy(mode)
