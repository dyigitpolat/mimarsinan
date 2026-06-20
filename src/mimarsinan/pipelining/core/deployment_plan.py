"""One contract-resolution layer for the whole deployment_parameters config.

``DeploymentPlan.resolve`` generalises the ``TtfsAdaptationPlan`` /
``SpikingDeploymentContract`` pattern to EVERY deployment axis: it is the single
place the scattered ``config.get(...)`` reads of a deployment flag are resolved,
so the rest of the pipeline reads the resolved decision instead of re-deriving it.

The spiking-semantics sub-part stays the ``SpikingDeploymentContract`` (the SSOT
for schedule/firing/wire questions); the plan composes it lazily via
``spiking_contract`` so a plan can be resolved from a config that has not yet had
``simulation_steps`` filled in (e.g. at pipeline-step-ordering time). The
schedule-derived booleans the step planner needs (``requires_ttfs_firing``,
``is_synchronized``) are resolved directly from the taxonomy, no sim length needed.

Behaviour is moved VERBATIM from the former inline reads; the inline defaults are
preserved key-for-key so the resolved plan is byte-identical to the old reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.chip_simulation.spiking_semantics import (
    is_synchronized_ttfs,
    is_ttfs_cycle_based,
    requires_ttfs_firing,
    ttfs_cycle_schedule,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode


@dataclass(frozen=True)
class DeploymentPlan:
    """Resolved deployment decisions; the rest of the pipeline reads THIS."""

    config: dict[str, Any]

    # ── workload / search ──────────────────────────────────────────────────
    search_mode: str
    model_type: str
    model_category: str | None
    weight_source: Any

    # ── spiking semantics (schedule-derived; full contract is lazy) ────────
    spiking_mode: str
    ttfs_cycle_schedule: str
    requires_ttfs_firing: bool
    is_synchronized_ttfs: bool
    is_ttfs_cycle_based: bool

    # ── conversion process ─────────────────────────────────────────────────
    activation_quantization: bool
    weight_quantization: bool
    enable_training_noise: bool
    cycle_accurate_lif_forward: bool

    # ── pruning ────────────────────────────────────────────────────────────
    pruning: bool
    pruning_fraction: float
    pruning_enabled: bool

    # ── deployment targets ─────────────────────────────────────────────────
    enable_nevresim_simulation: bool
    enable_loihi_simulation: bool
    enable_sanafe_simulation: bool

    # ── tolerances / budget ────────────────────────────────────────────────
    degradation_tolerance: float
    scm_degradation_tolerance: float | None
    degradation_budget_total: float
    cuda_debug: bool

    # ── deployment metric sampling ─────────────────────────────────────────
    deployment_metric_full_eval: bool
    max_simulation_samples: int
    simulation_batch_count: Any
    simulation_batch_size: int
    seed: int

    # ── identity ───────────────────────────────────────────────────────────
    model_name: str

    @classmethod
    def resolve(cls, config: dict[str, Any]) -> "DeploymentPlan":
        get = config.get

        spiking = get("spiking_mode", "lif")
        schedule_raw = get("ttfs_cycle_schedule")

        pruning = get("pruning", False)
        pruning_fraction = float(get("pruning_fraction", 0.0))

        degradation_tolerance = float(get("degradation_tolerance", 0.05))
        scm_dt = get("scm_degradation_tolerance")
        default_budget = 2.0 * degradation_tolerance

        model_type = get("model_type", "")

        return cls(
            config=config,
            search_mode=derive_search_mode(config),
            model_type=model_type,
            model_category=ModelRegistry.get_category(model_type),
            weight_source=get("weight_source"),
            spiking_mode=spiking,
            ttfs_cycle_schedule=ttfs_cycle_schedule(schedule_raw),
            requires_ttfs_firing=requires_ttfs_firing(spiking),
            is_synchronized_ttfs=is_synchronized_ttfs(spiking, schedule_raw),
            is_ttfs_cycle_based=is_ttfs_cycle_based(spiking),
            activation_quantization=bool(get("activation_quantization", False)),
            weight_quantization=bool(get("weight_quantization", False)),
            enable_training_noise=bool(get("enable_training_noise", False)),
            cycle_accurate_lif_forward=bool(get("cycle_accurate_lif_forward", False)),
            pruning=bool(pruning),
            pruning_fraction=pruning_fraction,
            pruning_enabled=bool(pruning) and pruning_fraction > 0,
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
        )

    @classmethod
    def of(cls, pipeline) -> "DeploymentPlan":
        """Resolve the plan for a pipeline (reads ``pipeline.config``)."""
        return cls.resolve(pipeline.config)

    def spiking_contract(self):
        """The spiking-semantics sub-part SSOT (needs ``simulation_steps``)."""
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        return SpikingDeploymentContract.from_pipeline_config(self.config)
