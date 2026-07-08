"""Single contract-resolution layer for the deployment config; the pipeline reads the resolved decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.chip_simulation.spiking_semantics import (
    is_synchronized_ttfs,
    is_ttfs_cycle_based,
    requires_ttfs_firing,
    ttfs_cycle_schedule,
    uses_ttfs_floor_ceil_convention as _uses_ttfs_floor_ceil_convention,
)
from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.tuning.orchestration.temporal_allocation import (
    TemporalAllocationResolver,
    resolve_s_allocation_mode,
)

OPTIMIZATION_DRIVER_CONTROLLER = "controller"
OPTIMIZATION_DRIVER_FAST = "fast"

_LEGACY_FAST_SWITCHES = (
    "lif_blend_fast", "ttfs_genuine_blend_fast", "ttfs_blend_fast",
)


def resolve_optimization_driver(config: dict[str, Any]) -> str:
    """The pipeline-wide ``controller | fast`` driver axis: explicit ``optimization_driver`` wins, else a legacy fast switch, else ``controller``."""
    explicit = config.get("optimization_driver")
    if explicit:
        value = str(explicit).lower()
        if value in (OPTIMIZATION_DRIVER_CONTROLLER, OPTIMIZATION_DRIVER_FAST):
            return value
        raise ValueError(
            f"optimization_driver must be '{OPTIMIZATION_DRIVER_CONTROLLER}' "
            f"or '{OPTIMIZATION_DRIVER_FAST}', got {explicit!r}"
        )
    if any(bool(config.get(switch, False)) for switch in _LEGACY_FAST_SWITCHES):
        return OPTIMIZATION_DRIVER_FAST
    return OPTIMIZATION_DRIVER_CONTROLLER


def _resolve_weight_source(config: dict[str, Any], workload: ResolvedWorkloadProfile) -> Any:
    """``weight_source`` with the F3 ``preload_weights`` regime flag folded in:
    the regime resolves to the builder-registered ``pretrained_weight_source``
    (fail-loud when nothing is registered)."""
    explicit = config.get("weight_source")
    if explicit:
        return explicit
    if bool(config.get("preload_weights", False)):
        if workload.pretrained_weight_source is None:
            raise ValueError(
                "preload_weights=true but no weight_source is set and the model "
                "builder registers no pretrained_weight_source "
                "(ModelWorkloadProfile). Declare weight_source explicitly."
            )
        return workload.pretrained_weight_source
    return explicit


@dataclass(frozen=True)
class DeploymentPlan:
    """Resolved deployment decisions; the rest of the pipeline reads THIS."""

    config: dict[str, Any]

    search_mode: str
    model_type: str
    model_category: str | None
    weight_source: Any

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
    seed: int

    model_name: str
    workload: ResolvedWorkloadProfile

    @classmethod
    def resolve(cls, config: dict[str, Any]) -> "DeploymentPlan":
        get = config.get

        spiking = get("spiking_mode", "lif")
        schedule_raw = get("ttfs_cycle_schedule")

        pruning = get("pruning", False)
        pruning_fraction = float(get("pruning_fraction", 0.0))
        prune_sparsity = float(get("prune_sparsity", 0.0) or 0.0)

        degradation_tolerance = float(get("degradation_tolerance", 0.05))
        scm_dt = get("scm_degradation_tolerance")
        default_budget = 2.0 * degradation_tolerance
        model_type = get("model_type", "")
        cls._require_chip_faithful_lif_forward(config, spiking)
        workload = ResolvedWorkloadProfile.from_config(config)

        return cls(
            config=config,
            search_mode=derive_search_mode(config),
            model_type=model_type,
            model_category=ModelRegistry.get_category(model_type),
            weight_source=_resolve_weight_source(config, workload),
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
            enable_nevresim_simulation=bool(get("enable_nevresim_simulation", True)),
            enable_loihi_simulation=bool(get("enable_loihi_simulation", False)),
            enable_sanafe_simulation=bool(get("enable_sanafe_simulation", False)),
            degradation_tolerance=degradation_tolerance,
            scm_degradation_tolerance=(
                None if scm_dt is None else float(scm_dt)
            ),
            degradation_budget_total=float(
                get("degradation_budget_total", default_budget)
            ),
            cuda_debug=bool(get("cuda_debug", False)),
            deployment_metric_full_eval=bool(
                get("deployment_metric_full_eval", True)
            ),
            max_simulation_samples=int(get("max_simulation_samples", 0) or 0),
            simulation_batch_count=get("simulation_batch_count", None),
            simulation_batch_size=int(get("simulation_batch_size", 8)),
            seed=int(get("seed", 0)),
            model_name=get("model_name") or model_type,
            workload=workload,
        )

    @staticmethod
    def _require_chip_faithful_lif_forward(config: dict[str, Any], spiking: str) -> None:
        """LIF-family gate: a Novena deployment must run the chip-faithful cycle-accurate forward (skipped for TTFS)."""
        if requires_ttfs_firing(spiking):
            return
        from mimarsinan.chip_simulation.firing_strategy import FiringStrategyFactory

        strategy = FiringStrategyFactory.from_config({
            "spiking_mode": spiking,
            "firing_mode": config.get("firing_mode", "Default"),
            "thresholding_mode": config.get("thresholding_mode", "<="),
        })
        strategy.require_chip_faithful_lif_forward(
            cycle_accurate_lif_forward=bool(
                config.get("cycle_accurate_lif_forward", True)
            ),
        )

    @classmethod
    def of(cls, pipeline) -> "DeploymentPlan":
        """Resolve the plan for a pipeline (reads ``pipeline.config``)."""
        return cls.resolve(pipeline.config)

    def mode_policy(self):
        """The behavior-carrying ``SpikingModePolicy`` for this plan (resolved without ``simulation_steps``)."""
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )

        return policy_for_spiking_mode(self.spiking_mode, self.ttfs_cycle_schedule)

    @property
    def conversion_recipe(self):
        """The ConversionPolicy SSOT recipe for this plan's resolved ``(spiking_mode, schedule)`` mode."""
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
        return (
            self.runs_cycle_accurate_activation_tuner
            or self.activation_quantization
            or self.requires_ttfs_firing
        )

    @property
    def requires_activation_quantization_preconditioning(self) -> bool:
        """Run shift/AQ before cycle tuning or when activation quantization is enabled."""
        return self.runs_cycle_accurate_activation_tuner or self.activation_quantization

    def spiking_contract(self):
        """The spiking-semantics sub-part SSOT (needs ``simulation_steps``)."""
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
        """The per-depth temporal-allocation map (the reserved per-layer-S axis seam).

        ``s_allocation='uniform'`` (default) returns the global ``simulation_steps`` for
        every depth; ``depth`` is supplied by the caller.
        """
        return TemporalAllocationResolver.from_config(self.config).resolve(depth=depth)
