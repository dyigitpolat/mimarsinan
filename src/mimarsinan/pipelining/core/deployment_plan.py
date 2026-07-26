"""Single contract-resolution layer for the deployment config; the pipeline reads the resolved decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.chip_simulation.core_semantics import (
    INERT_SPIKING_MODE, is_mvm_core_semantics, resolve_core_semantics,
)
from mimarsinan.chip_simulation.spiking_semantics import (
    is_synchronized_ttfs,
    is_ttfs_cycle_based,
    requires_ttfs_firing,
    ttfs_cycle_schedule,
    uses_ttfs_floor_ceil_convention as _uses_ttfs_floor_ceil_convention,
)
from mimarsinan.common.pretrained import (
    preload_regime_error,
    select_weight_set,
    selected_source,
)
from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.transformations.channel_scale_equalization import DEFAULT_CLIP_RATIO
from mimarsinan.tuning.orchestration.optimization_driver import (
    OPTIMIZATION_DRIVER_FAST,
    resolve_optimization_driver,
)
from mimarsinan.tuning.orchestration.temporal_allocation import (
    TemporalAllocationResolver,
    resolve_s_allocation_mode,
)

def resolve_weight_source(config: dict[str, Any]) -> Any:
    """THE weight-source resolution: an explicit ``weight_source`` > the chosen
    builder-registered weight set's source > None. The regime with nothing
    selectable fails LOUD (the configurator locks the switch, so it is
    unauthorable there and never 500s)."""
    explicit = config.get("weight_source")
    if explicit:
        return explicit
    if bool(config.get("preload_weights", False)):
        source = selected_source(config)
        if source is None:
            raise preload_regime_error(config)
        return source
    return None


@dataclass(frozen=True)
class DeploymentPlan:
    """Resolved deployment decisions; the rest of the pipeline reads THIS."""

    config: dict[str, Any]

    search_mode: str
    model_type: str
    model_category: str | None
    weight_source: Any
    pretrained_weight_set: dict[str, Any] | None

    core_semantics: str
    spiking_mode: str
    ttfs_cycle_schedule: str
    requires_ttfs_firing: bool
    is_synchronized_ttfs: bool
    is_ttfs_cycle_based: bool

    activation_quantization: bool
    weight_quantization: bool
    enable_training_noise: bool
    cycle_accurate_lif_forward: bool

    optimization_driver: str

    s_allocation: str

    pruning: bool
    pruning_fraction: float
    pruning_enabled: bool
    prune_sparsity: float
    scale_migration_enabled: bool
    scale_migration_clip_ratio: float

    enable_nevresim_simulation: bool
    enable_loihi_simulation: bool
    enable_sanafe_simulation: bool

    degradation_tolerance: float
    scm_degradation_tolerance: float | None
    degradation_budget_total: float
    cuda_debug: bool

    deployment_metric_full_eval: bool
    max_simulation_samples: int
    simulation_batch_count: Any
    simulation_batch_size: int
    eval_max_samples: int
    seed: int

    model_name: str
    workload: ResolvedWorkloadProfile

    @classmethod
    def resolve(cls, config: dict[str, Any]) -> "DeploymentPlan":
        get = config.get

        core_semantics = resolve_core_semantics(config)
        mvm = is_mvm_core_semantics(core_semantics)
        spiking = INERT_SPIKING_MODE if mvm else get("spiking_mode", "lif")
        schedule_raw = get("ttfs_cycle_schedule")

        pruning = get("pruning", False)
        pruning_fraction = float(get("pruning_fraction", 0.0))
        prune_sparsity = float(get("prune_sparsity", 0.0) or 0.0)
        degradation_tolerance = float(get("degradation_tolerance", 0.05))
        scm_dt = get("scm_degradation_tolerance")
        default_budget = 2.0 * degradation_tolerance
        model_type = get("model_type", "")
        model_category = ModelRegistry.get_category(model_type)
        if mvm and model_category == "native":
            raise ValueError(
                f"core_semantics='mvm' supports torch-category models only: "
                f"{model_type!r} is a native builder whose authored perceptron "
                f"activations affine cores would silently drop (de-fusion is "
                f"the designed follow-up seam).")
        if not mvm:
            from mimarsinan.chip_simulation import firing_strategy

            firing_strategy.require_chip_faithful_lif_forward(config, spiking)
        workload = ResolvedWorkloadProfile.from_config(config)

        return cls(
            config=config,
            search_mode=derive_search_mode(config),
            model_type=model_type,
            model_category=model_category,
            weight_source=resolve_weight_source(config),
            pretrained_weight_set=select_weight_set(config),
            core_semantics=core_semantics,
            spiking_mode=spiking,
            ttfs_cycle_schedule=ttfs_cycle_schedule(schedule_raw),
            requires_ttfs_firing=requires_ttfs_firing(spiking),
            is_synchronized_ttfs=is_synchronized_ttfs(spiking, schedule_raw),
            is_ttfs_cycle_based=is_ttfs_cycle_based(spiking),
            activation_quantization=bool(get("activation_quantization", False)),
            weight_quantization=bool(get("weight_quantization", False)),
            enable_training_noise=bool(get("enable_training_noise", False)),
            cycle_accurate_lif_forward=bool(get("cycle_accurate_lif_forward", False)),
            optimization_driver=resolve_optimization_driver(config),
            s_allocation=resolve_s_allocation_mode(config),
            pruning=bool(pruning),
            pruning_fraction=pruning_fraction,
            pruning_enabled=bool(pruning) and pruning_fraction > 0,
            prune_sparsity=prune_sparsity,
            scale_migration_enabled=bool(get("scale_migration", False)),
            scale_migration_clip_ratio=float(
                get("scale_migration_clip_ratio", DEFAULT_CLIP_RATIO)),
            enable_nevresim_simulation=bool(get("enable_nevresim_simulation", True)),
            enable_loihi_simulation=bool(get("enable_loihi_simulation", False)),
            enable_sanafe_simulation=bool(get("enable_sanafe_simulation", False)),
            degradation_tolerance=degradation_tolerance,
            scm_degradation_tolerance=None if scm_dt is None else float(scm_dt),
            degradation_budget_total=float(
                get("degradation_budget_total", default_budget)),
            cuda_debug=bool(get("cuda_debug", False)),
            deployment_metric_full_eval=bool(get("deployment_metric_full_eval", True)),
            max_simulation_samples=int(get("max_simulation_samples", 0) or 0),
            simulation_batch_count=get("simulation_batch_count", None),
            simulation_batch_size=int(get("simulation_batch_size", 8)),
            eval_max_samples=int(get("eval_max_samples", 10000)),  # registry SSOT default
            seed=int(get("seed", 0)),
            model_name=get("model_name") or model_type,
            workload=workload,
        )

    @classmethod
    def of(cls, pipeline) -> "DeploymentPlan":
        """Resolve the plan for a pipeline (reads ``pipeline.config``)."""
        return cls.resolve(pipeline.config)

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
