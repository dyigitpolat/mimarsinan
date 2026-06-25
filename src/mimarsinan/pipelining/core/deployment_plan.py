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
from mimarsinan.tuning.orchestration.temporal_allocation import (
    resolve_s_allocation_mode,
)

OPTIMIZATION_DRIVER_CONTROLLER = "controller"
OPTIMIZATION_DRIVER_FAST = "fast"

# The per-family fast switches that selected the fast ladder BEFORE E2 unbound the
# driver into a single axis. The pipeline-wide `optimization_driver` key (default
# `controller`) wins when set; otherwise the axis reflects whichever legacy switch a
# config still carries, so a plan resolved from an existing config reports the SAME
# driver the tuner runs (byte-identical: all default off ⇒ `controller`).
_LEGACY_FAST_SWITCHES = (
    "lif_blend_fast",
    "ttfs_genuine_blend_fast",
    "ttfs_blend_fast",
    "ttfs_staircase_ste_fast",
)


def resolve_optimization_driver(config: dict[str, Any]) -> str:
    """The pipeline-wide ``controller | fast`` driver axis (E2). Explicit
    ``optimization_driver`` wins; else derived from any legacy per-family fast
    switch; else ``controller`` (default ⇒ byte-identical).

    Public so the consuming half (EF1) can resolve the axis directly from a config
    dict — e.g. ``TtfsAdaptationPlan.resolve``'s back-compat path — without holding a
    ``DeploymentPlan`` instance."""
    explicit = config.get("optimization_driver")
    if explicit:
        value = str(explicit).lower()
        if value in (OPTIMIZATION_DRIVER_CONTROLLER, OPTIMIZATION_DRIVER_FAST):
            return value
        raise ValueError(
            f"optimization_driver must be "
            f"'{OPTIMIZATION_DRIVER_CONTROLLER}' or '{OPTIMIZATION_DRIVER_FAST}', "
            f"got {explicit!r}"
        )
    if any(bool(config.get(switch, False)) for switch in _LEGACY_FAST_SWITCHES):
        return OPTIMIZATION_DRIVER_FAST
    return OPTIMIZATION_DRIVER_CONTROLLER


# Back-compat private alias (the dataclass body below referenced the old name).
_resolve_optimization_driver = resolve_optimization_driver

# F3 dual-regime axis: ``preload_weights`` is the publication-batch boolean for
# the pretrained regime. An EXPLICIT ``weight_source`` always wins; the flag only
# derives the default torchvision-pretrained source when no source is set, so an
# unset flag is byte-identical (``weight_source`` unchanged => from_scratch).
PRETRAINED_WEIGHT_SOURCE = "torchvision"


def _resolve_weight_source(config: dict[str, Any]) -> Any:
    """``weight_source`` with the F3 ``preload_weights`` regime flag folded in."""
    explicit = config.get("weight_source")
    if explicit:
        return explicit
    if bool(config.get("preload_weights", False)):
        return PRETRAINED_WEIGHT_SOURCE
    return explicit


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

    # ── optimization driver (E2: how the rate is driven 0→1, pipeline-wide) ──
    optimization_driver: str

    # ── per-layer-S temporal allocation (EW1: declared intent; map RESERVED) ──
    s_allocation: str

    # ── pruning ────────────────────────────────────────────────────────────
    pruning: bool
    pruning_fraction: float
    pruning_enabled: bool
    # D4: structured magnitude-pruning fraction applied BEFORE mapping (default
    # 0.0 ⇒ no structural pruning ⇒ byte-identical). Distinct from the in-loop
    # ``pruning``/``pruning_fraction`` mask machinery above; this is the deployment
    # knob the coverage ledger enumerates as ``pruning=dense`` vs ``pruning=pruned``.
    prune_sparsity: float

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
        prune_sparsity = float(get("prune_sparsity", 0.0) or 0.0)

        degradation_tolerance = float(get("degradation_tolerance", 0.05))
        scm_dt = get("scm_degradation_tolerance")
        default_budget = 2.0 * degradation_tolerance

        model_type = get("model_type", "")

        cls._require_chip_faithful_lif_forward(config, spiking)

        return cls(
            config=config,
            search_mode=derive_search_mode(config),
            model_type=model_type,
            model_category=ModelRegistry.get_category(model_type),
            weight_source=_resolve_weight_source(config),
            spiking_mode=spiking,
            ttfs_cycle_schedule=ttfs_cycle_schedule(schedule_raw),
            requires_ttfs_firing=requires_ttfs_firing(spiking),
            is_synchronized_ttfs=is_synchronized_ttfs(spiking, schedule_raw),
            is_ttfs_cycle_based=is_ttfs_cycle_based(spiking),
            activation_quantization=bool(get("activation_quantization", False)),
            weight_quantization=bool(get("weight_quantization", False)),
            enable_training_noise=bool(get("enable_training_noise", False)),
            cycle_accurate_lif_forward=bool(get("cycle_accurate_lif_forward", False)),
            optimization_driver=_resolve_optimization_driver(config),
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
        )

    @staticmethod
    def _require_chip_faithful_lif_forward(config: dict[str, Any], spiking: str) -> None:
        """LIF-family capability gate (V3): a Novena deployment must run the
        chip-faithful cycle-accurate forward. Skipped for the TTFS family (firing
        is TTFS there); the FiringStrategy owns the (firing × cycle-accurate) rule."""
        if requires_ttfs_firing(spiking):
            return
        from mimarsinan.chip_simulation.firing_strategy import FiringStrategyFactory

        strategy = FiringStrategyFactory.from_config({
            "spiking_mode": spiking,
            "firing_mode": config.get("firing_mode", "Default"),
            "thresholding_mode": config.get("thresholding_mode", "<="),
        })
        # Absent ⇒ the LIF default (chip-faithful cascade) the pipeline setdefaults
        # to True; only an EXPLICIT opt-out trips the gate. Keeps a config that
        # merely omits the key (relying on the default) resolvable.
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
        """The behavior-carrying ``SpikingModePolicy`` (V2) for this plan.

        Resolved directly from the schedule-derived axes — no ``simulation_steps``
        needed — so the step planner can compose with the policy at
        pipeline-step-ordering time (the full contract is lazy via
        ``spiking_contract``)."""
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )

        return policy_for_spiking_mode(self.spiking_mode, self.ttfs_cycle_schedule)

    @property
    def is_fast_driver(self) -> bool:
        """Whether the resolved optimization-driver axis is the fast ladder (E2)."""
        return self.optimization_driver == OPTIMIZATION_DRIVER_FAST

    def optimization_driver_for_family(
        self, *, rates, steps_per_rate, eta_min_factor=0.0,
    ):
        """The family's ``controller | fast`` ``OptimizationDriver``, READ from the
        pipeline-wide axis (EF1, the consuming half).

        The single-switch families (LIF + the analytical clamp/shift/activation-quant/
        activation-adaptation chain + the manager-rate family) resolve their driver
        through THIS one seam instead of each reading a hard-coded per-family fast
        switch (or, for the analytical/manager families, never reading the axis at
        all). ``fast`` is the pipeline-wide ``is_fast_driver`` decision; the family
        supplies its uniform value-domain ladder. Default ``controller`` ⇒ the ladder
        is carried but disabled (``_setup_fast_ladder(enabled=False)``) ⇒
        byte-identical."""
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
    def is_lif_style(self) -> bool:
        """LIF or ttfs_cycle: one activation-replacement step subsumes the
        clamp/shift/activation-quantization chain (it clamps + quantises
        internally), so that chain is skipped. The (firing × sync) branch the
        step planner reads to choose the activation-adaptation family —
        composed from the V2 ``SpikingModePolicy``."""
        return self.mode_policy().single_step_activation_replacement

    def spiking_contract(self):
        """The spiking-semantics sub-part SSOT (needs ``simulation_steps``)."""
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        return SpikingDeploymentContract.from_pipeline_config(self.config)

    def calibration_pipeline(self, *, distmatch_driven=False):
        """The conversion-health ``CalibrationPipeline`` for this plan's (firing ×
        sync) cell (E3).

        Pipeline-wide: every conversion tuner reads ITS calibration through this one
        contract-keyed resolver. Resolved from the schedule-derived policy (no
        ``simulation_steps`` needed) and the plan's config; the cascaded cycle opts
        into the conversion-health steps, LIF / analytical / synchronized get the
        inert pipeline (default-off ⇒ byte-identical)."""
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        return CalibrationPipeline.for_mode(
            self.config,
            mode_policy=self.mode_policy(),
            distmatch_driven=distmatch_driven,
        )

    def temporal_allocation(self, *, depth: int):
        """The EW1 per-depth temporal-allocation map for a model of ``depth`` cascade
        depths / latency groups — the RESERVED per-layer-S axis seam.

        DEFAULT-OFF / byte-identical: ``s_allocation='uniform'`` (the default) returns
        the SAME global ``simulation_steps`` for every depth, so nothing threads a
        non-uniform map. ``explicit`` validates a declared per-depth list; ``budget`` is
        a no-op that returns uniform + a ``derivation_deferred`` marker (the budget
        allocator's derivation is deferred to the ConversionPolicy keystone, research).
        ``depth`` is supplied by the caller (this layer does not introspect the model).
        Gated by the ``allow_per_layer_s`` chip capability; nothing reads the map yet."""
        from mimarsinan.tuning.orchestration.temporal_allocation import (
            TemporalAllocationResolver,
        )

        return TemporalAllocationResolver.from_config(self.config).resolve(depth=depth)

    def conversion_policy(self, *, model=None, characterizer=None, context=None):
        """The E4 characterization-and-policy decision for this plan's (firing ×
        sync) cell — the keystone seam (propose → confirm → escalate).

        DEFAULT-OFF / byte-identical: until ``conversion_policy`` is set in config
        the returned ``ConversionDecision`` names the CURRENT behavior
        (driver=controller, no characterization run) so nothing changes. When opted
        in, the plan's policy proposes the recipe, the ``characterizer`` confirms it
        on ``model`` (calibration batches drawn from ``context``, the trainer), and a
        mismatch escalates to the controller fallback rather than shipping a silent
        regression. ``context`` is a backward-compatible keyword (default None ⇒
        inert ⇒ byte-identical). This is the scaffolding Fix B switches on later; it
        is exposed, NOT enabled."""
        from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

        return ConversionPolicy.resolve(
            self.config,
            mode_policy=self.mode_policy(),
            model=model,
            characterizer=characterizer,
            context=context,
        )
