"""Derived plan predicates — the questions steps ask a resolved plan.

Split out of ``deployment_plan`` so the dataclass + resolution stay within
the module budget and each concern has one home [D5].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
from mimarsinan.chip_simulation.spiking_semantics import (
    uses_ttfs_floor_ceil_convention as _uses_ttfs_floor_ceil_convention,
)
from mimarsinan.tuning.orchestration.temporal_allocation import (
    TemporalAllocationResolver,
)
from mimarsinan.tuning.orchestration.optimization_driver import (
    OPTIMIZATION_DRIVER_FAST,
)


class PlanPredicates:
    """Mixin: every derived answer a step may ask of a resolved plan."""

    # Supplied by the DeploymentPlan dataclass this mixin composes into.
    # TYPE_CHECKING-guarded: a bare annotation here would be inherited
    # as a dataclass FIELD and silently reshape the plan's signature.
    if TYPE_CHECKING:
        activation_quantization: Any
        config: Any
        core_semantics: Any
        is_synchronized_ttfs: Any
        is_ttfs_cycle_based: Any
        optimization_driver: Any
        requires_ttfs_firing: Any
        spiking_mode: Any
        ttfs_cycle_schedule: Any

    @property
    def declared_simulation_batch_size(self) -> "int | None":
        """The census bound the CONFIG declared, or None when it is silent.

        ``simulation_batch_size`` derives to 8, which funds the OOM retry and
        is not a statement about the first read's batch — so the declared and
        resolved values are different questions and get different accessors.
        """
        declared = self.config.get("simulation_batch_size")
        return int(declared) if declared else None

    @property
    def is_mvm(self) -> bool:
        """Whether this plan targets the value-domain MVM core family."""
        return is_mvm_core_semantics(self.core_semantics)

    def mode_policy(self):
        """The behavior-carrying mode policy for this plan (domain-first dispatch)."""
        if self.is_mvm:
            from mimarsinan.chip_simulation.mvm_core_policy import MvmCorePolicy

            return MvmCorePolicy()
        from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

        return policy_for_spiking_mode(self.spiking_mode, self.ttfs_cycle_schedule)

    @property
    def conversion_recipe(self):
        """The recipe SSOT for this plan: the mvm recipe, else ``(spiking_mode, schedule)``."""
        if self.is_mvm:
            from mimarsinan.tuning.orchestration.mvm_conversion import derive_mvm_recipe

            return derive_mvm_recipe()
        from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

        return ConversionPolicy.derive(self.spiking_mode, self.ttfs_cycle_schedule)

    @property
    def is_fast_driver(self) -> bool:
        """Whether the resolved optimization-driver axis is the fast ladder (E2)."""
        return self.optimization_driver == OPTIMIZATION_DRIVER_FAST

    def optimization_driver_for_family(
        self, *, rates, steps_per_rate, eta_min_factor=0.0,
    ):
        """The family's ``controller | fast`` ``OptimizationDriver``, read from the pipeline-wide axis.

        ``fast`` is the plan's ``is_fast_driver`` decision; the family supplies its ladder.
        """
        from mimarsinan.tuning.orchestration.optimization_driver import (
            OptimizationDriver,
        )

        return OptimizationDriver.for_family(
            fast=self.is_fast_driver,
            rates=rates,
            steps_per_rate=steps_per_rate,
            eta_min_factor=eta_min_factor,
        )

    @property
    def is_cascaded_ttfs(self) -> bool:
        """ttfs_cycle_based on the cascaded schedule (the complement of ``is_synchronized_ttfs``)."""
        return self.is_ttfs_cycle_based and not self.is_synchronized_ttfs

    @property
    def uses_ttfs_floor_ceil_convention(self) -> bool:
        """Whether the NF trains the floor + half-step-bias convention and deploys
        the ceil TTFS kernel (ttfs_quantized and the synchronized floor-collapse)."""
        return _uses_ttfs_floor_ceil_convention(
            self.spiking_mode, self.ttfs_cycle_schedule
        )

    @property
    def is_lif_style(self) -> bool:
        """Whether this plan has a dedicated LIF/TTFS-cycle tuning step."""
        return self.mode_policy().single_step_activation_replacement

    @property
    def runs_cycle_accurate_activation_tuner(self) -> bool:
        """Whether LIF/TTFS-cycle fine-tuning follows activation preconditioning."""
        return self.spiking_mode == "lif" or self.is_ttfs_cycle_based

    @property
    def requires_clamp_preconditioning(self) -> bool:
        """Clamp before TTFS firing, activation quantization, or cycle tuning."""
        return not self.is_mvm and (
            self.runs_cycle_accurate_activation_tuner
            or self.activation_quantization
            or self.requires_ttfs_firing
        )

    @property
    def requires_activation_quantization_preconditioning(self) -> bool:
        """Run shift/AQ before cycle tuning or when activation quantization is enabled."""
        return not self.is_mvm and (
            self.runs_cycle_accurate_activation_tuner or self.activation_quantization
        )

    def spiking_contract(self):
        """The spiking-semantics sub-part SSOT (needs ``simulation_steps``)."""
        if self.is_mvm:
            raise RuntimeError(
                "value-domain (core_semantics='mvm') plans have no spiking "
                "contract; event machinery must never be reached on this path."
            )
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        return SpikingDeploymentContract.from_pipeline_config(self.config)

    def calibration_pipeline(self, *, distmatch_driven=False):
        """The conversion-health ``CalibrationPipeline`` for this plan's (firing × sync) cell."""
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        return CalibrationPipeline.for_mode(
            self.config,
            mode_policy=self.mode_policy(),
            distmatch_driven=distmatch_driven,
        )

    def temporal_allocation(self, *, depth: int):
        """The per-depth temporal-allocation map (the reserved per-layer-S seam);
        ``s_allocation='uniform'`` returns the global ``simulation_steps``."""
        return TemporalAllocationResolver.from_config(self.config).resolve(depth=depth)
