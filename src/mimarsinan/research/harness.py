"""Typed contracts for promoting research prototypes into production recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mimarsinan.chip_simulation.ledger_schema import normalize_planned_ledger_row

ACCEPTANCE_DEPLOYED_ACC = 0.97
ACCEPTANCE_RELATIVE_TIME = 1.0

MIXER_DIAGNOSTIC_STEP_NAMES = (
    "Activation Analysis",
    "Activation Adaptation",
    "Clamp Adaptation",
    "Activation Shifting",
    "Activation Quantization",
    "LIF/TTFS Cycle Tuning",
    "Weight Quantization",
    "Normalization Fusion",
    "Soft Core Mapping",
    "Hard Core Mapping",
    "Simulation",
)


def _provider_name(dataset: str) -> str:
    name = str(dataset)
    if name.endswith("_DataProvider"):
        return name
    aliases = {
        "cifar10": "CIFAR10_DataProvider",
        "cifar100": "CIFAR100_DataProvider",
        "mnist": "MNIST_DataProvider",
        "fmnist": "FashionMNIST_DataProvider",
        "kmnist": "KMNIST_DataProvider",
        "svhn": "SVHN_DataProvider",
    }
    return aliases.get(name.lower(), name)


@dataclass(frozen=True)
class VehicleSpec:
    """A model/dataset vehicle a prototype or production run targets."""

    name: str
    model_type: str
    dataset: str
    depth: int | None = None
    input_shape: tuple[int, ...] | None = None
    num_classes: int | None = None
    residual: bool | None = None
    extra_model_config: Mapping[str, Any] = field(default_factory=dict)

    def config_overlay(self) -> dict[str, Any]:
        model_config = dict(self.extra_model_config)
        if self.depth is not None:
            model_config["depth"] = int(self.depth)
        if self.input_shape is not None:
            model_config["input_shape"] = [int(v) for v in self.input_shape]
        if self.num_classes is not None:
            model_config["num_classes"] = int(self.num_classes)
        if self.residual is not None:
            model_config["residual"] = bool(self.residual)
        out = {
            "model_type": self.model_type,
            "data_provider_name": _provider_name(self.dataset),
        }
        if model_config:
            out["model_config"] = model_config
        return out


@dataclass(frozen=True)
class FixRecipe:
    """A named mechanism with default-off config flags for production promotion."""

    name: str
    mechanism: str
    config_flags: Mapping[str, Any] = field(default_factory=dict)
    owner: str | None = None
    rationale: str = ""

    def config_overlay(self) -> dict[str, Any]:
        return dict(self.config_flags)


def recipe_presets() -> dict[str, FixRecipe]:
    """Return named default-off recipes that emit production config overlays."""
    return {
        "sync_qat_fast_bn": FixRecipe(
            name="sync_qat_fast_bn",
            mechanism="synchronized_qat",
            config_flags={
                "spiking_mode": "ttfs_cycle_based",
                "ttfs_cycle_schedule": "synchronized",
                "ttfs_sync_genuine_qat": True,
                "ttfs_blend_fast": True,
                "fast_ladder_freeze_bn": True,
            },
            owner="conversion",
            rationale="Synchronized TTFS genuine-QAT fast path with frozen BN.",
        ),
        "lif_qat_fast_bn": FixRecipe(
            name="lif_qat_fast_bn",
            mechanism="lif_qat",
            config_flags={
                "spiking_mode": "lif",
                "lif_blend_fast": True,
                "fast_ladder_freeze_bn": True,
                "cycle_accurate_lif_forward": True,
            },
            owner="conversion",
            rationale="Adapted LIF QAT fast path with cycle-accurate forward.",
        ),
    }


def recipe_preset(name: str) -> FixRecipe:
    """Return one named recipe preset or raise ``KeyError`` with known names."""
    presets = recipe_presets()
    try:
        return presets[name]
    except KeyError as exc:
        raise KeyError(f"unknown recipe preset {name!r}; known: {sorted(presets)}") from exc


@dataclass(frozen=True)
class BudgetSchedule:
    """Timing and adaptation-budget envelope for a conversion recipe."""

    name: str
    max_tuning_wall_s: float | None
    max_ft_pass_wall_s: float | None = None
    max_adaptation_steps: int | None = None
    stabilization_steps: int | None = None
    scale_ramp_steps: bool = False

    def config_overlay(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tuning_budget_scale_ramp_steps": bool(self.scale_ramp_steps),
        }
        if self.max_tuning_wall_s is not None:
            out["tuning_budget_max_wall_s"] = float(self.max_tuning_wall_s)
        if self.max_ft_pass_wall_s is not None:
            out["max_ft_pass_wall_s"] = float(self.max_ft_pass_wall_s)
        if self.max_adaptation_steps is not None:
            out["tuning_max_adaptation_steps"] = int(self.max_adaptation_steps)
        if self.stabilization_steps is not None:
            out["tuning_stabilization_steps"] = int(self.stabilization_steps)
        return out


@dataclass(frozen=True)
class ExperimentContext:
    """One prototype or closure-run context, expressible as production config."""

    vehicle: VehicleSpec
    recipe: FixRecipe
    budget: BudgetSchedule
    seed: int
    simulation_steps: int
    platform: Mapping[str, Any] = field(default_factory=dict)

    def config_overlay(self) -> dict[str, Any]:
        out = {}
        out.update(self.vehicle.config_overlay())
        out.update(self.recipe.config_overlay())
        out.update(self.budget.config_overlay())
        out["seed"] = int(self.seed)
        out["simulation_steps"] = int(self.simulation_steps)
        if self.platform:
            out["platform_constraints"] = dict(self.platform)
        return out


@dataclass(frozen=True)
class MechanismResult:
    """Measured result of one recipe on one vehicle."""

    ann_acc: float
    deployed_acc: float
    wall_s: float
    tuning_wall_s: float
    parity_mismatch: float
    validity_tier: str


def normalize_retention(result: MechanismResult) -> float:
    if result.ann_acc <= 0.0:
        return 0.0
    return float(result.deployed_acc) / float(result.ann_acc)


def promotion_record(
    *, context: ExperimentContext, result: MechanismResult, source: str,
) -> dict[str, Any]:
    retention = normalize_retention(result)
    timing_ok = (
        context.budget.max_tuning_wall_s is None
        or result.tuning_wall_s <= context.budget.max_tuning_wall_s
    )
    return {
        "vehicle": context.vehicle.name,
        "recipe": context.recipe.name,
        "mechanism": context.recipe.mechanism,
        "budget": context.budget.name,
        "source": source,
        "ann_acc": float(result.ann_acc),
        "deployed_acc": float(result.deployed_acc),
        "retention": retention,
        "wall_s": float(result.wall_s),
        "tuning_wall_s": float(result.tuning_wall_s),
        "parity_mismatch": float(result.parity_mismatch),
        "validity_tier": result.validity_tier,
        "passes_retention_gate": retention >= 0.85,
        "passes_parity_gate": result.parity_mismatch == 0.0,
        "passes_timing_gate": bool(timing_ok),
    }


@dataclass(frozen=True)
class AcceptanceVerdict:
    accepted: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcceptanceGate:
    min_deployed_acc: float = ACCEPTANCE_DEPLOYED_ACC
    max_relative_time: float = ACCEPTANCE_RELATIVE_TIME

    def to_dict(self) -> dict[str, float]:
        return {
            "min_deployed_acc": float(self.min_deployed_acc),
            "max_relative_time": float(self.max_relative_time),
        }

    def evaluate(self, row: Mapping[str, Any]) -> AcceptanceVerdict:
        reasons: list[str] = []
        if int(row.get("returncode", 0)) != 0:
            reasons.append("returncode")
        acc = row.get("deployed_acc")
        if acc is None or float(acc) < self.min_deployed_acc:
            reasons.append("deployed_acc")
        rel = row.get("relative_time")
        if rel is None or float(rel) >= self.max_relative_time:
            reasons.append("relative_time")
        return AcceptanceVerdict(accepted=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class MixerBudgetSchedule:
    """Separated budget accounting for mixer recipe comparison."""

    ramp_steps: int
    recovery_steps: int
    stabilization_steps: int
    eval_sample_count: int
    max_tuning_wall_s: float | None = None
    max_total_wall_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ramp_steps": int(self.ramp_steps),
            "recovery_steps": int(self.recovery_steps),
            "stabilization_steps": int(self.stabilization_steps),
            "eval_sample_count": int(self.eval_sample_count),
            "max_tuning_wall_s": self.max_tuning_wall_s,
            "max_total_wall_s": self.max_total_wall_s,
        }


@dataclass(frozen=True)
class RecipePreset:
    recipe_id: str
    label: str
    mode_family: str
    base_overrides: Mapping[str, Any]
    budget_schedule: MixerBudgetSchedule
    required_probes: tuple[str, ...] = ()
    acceptance: AcceptanceGate = field(default_factory=AcceptanceGate)
    notes: str = ""

    def to_tags(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_family": self.mode_family,
            "budget_schedule": self.budget_schedule.to_dict(),
            "required_probes": list(self.required_probes),
        }


@dataclass(frozen=True)
class DiagnosticCell:
    cell_id: str
    template: str
    firing: str
    sync: str
    role: str
    recipe_ids: tuple[str, ...]
    dataset: str = "MNIST_DataProvider"
    vehicle: str = "mlp_mixer_core"
    backend: str = "sanafe"
    acceptance: AcceptanceGate = field(default_factory=AcceptanceGate)

    def axes(self) -> dict[str, Any]:
        return {
            "vehicle": self.vehicle,
            "dataset": self.dataset,
            "firing": self.firing,
            "sync": self.sync,
            "backend": self.backend,
            "quantization": "weight_activation"
            if self.firing == "ttfs_quantized"
            else "weight",
        }


@dataclass(frozen=True)
class DiagnosticManifest:
    manifest_id: str
    cells: tuple[DiagnosticCell, ...]
    seeds: tuple[int, ...]
    acceptance: AcceptanceGate = field(default_factory=AcceptanceGate)

    def cell_by_id(self, cell_id: str) -> DiagnosticCell:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(cell_id)


def _mixer_recipe(
    recipe_id: str,
    label: str,
    mode_family: str,
    overrides: Mapping[str, Any],
    budget: MixerBudgetSchedule,
    *,
    probes: Sequence[str] = (),
    notes: str = "",
) -> RecipePreset:
    merged = {
        "deployment_parameters.recipe_id": recipe_id,
        "deployment_parameters.acceptance_min_deployed_acc": ACCEPTANCE_DEPLOYED_ACC,
        "deployment_parameters.acceptance_max_relative_time": ACCEPTANCE_RELATIVE_TIME,
        "deployment_parameters.relative_timing_required": True,
    }
    merged.update(dict(overrides))
    return RecipePreset(
        recipe_id=recipe_id,
        label=label,
        mode_family=mode_family,
        base_overrides=merged,
        budget_schedule=budget,
        required_probes=tuple(probes),
        notes=notes,
    )


def _lif_genuine_qat_recipe(recipe_id: str, *, alpha: float) -> RecipePreset:
    """Genuine-cascade QAT for LIF: a BN-frozen fast ladder driven through the
    cycle-accurate LIF forward with a KD+CE blend. Production port of the proven
    CIFAR deep-residual fix; ``alpha`` sweeps the CE/KD weighting (kd_ce_alpha)."""
    return _mixer_recipe(
        recipe_id,
        f"Mixer LIF genuine-cascade QAT (kd_ce_alpha={alpha})",
        "lif",
        {
            "deployment_parameters.optimization_driver": "fast",
            "deployment_parameters.lif_blend_fast": True,
            "deployment_parameters.lif_blend_fast_stabilize_steps": 600,
            "deployment_parameters.cycle_accurate_lif_forward": True,
            "deployment_parameters.fast_ladder_freeze_bn": True,
            "deployment_parameters.kd_ce_alpha": alpha,
            "deployment_parameters.kd_temperature": 4.0,
        },
        MixerBudgetSchedule(
            ramp_steps=4 * 120,
            recovery_steps=0,
            stabilization_steps=600,
            eval_sample_count=5000,
            max_tuning_wall_s=900.0,
        ),
        probes=("genuine_endpoint",),
        notes=(
            "Production port of the proven genuine-cascade QAT (BN-frozen, KD+CE) "
            "that closed deep-residual LIF conversion on CIFAR. The a03/a05/a07 "
            "variants sweep the kd_ce_alpha blend to locate the conversion optimum."
        ),
    )


def _lif_theta_qat_recipe(recipe_id: str, *, alpha: float) -> RecipePreset:
    """Round-2 LIF gap closer: compose the two MEASURED-faithful levers — per-channel
    trainable theta (the decode scale, which survives LIF deployment) co-trained by the
    gradual blend ramp, AND the genuine-cascade KD+CE QAT objective — on the
    SHORT-stabilize fine ladder. ``alpha`` sweeps the CE/KD weighting (kd_ce_alpha)."""
    return _mixer_recipe(
        recipe_id,
        f"Mixer LIF per-channel theta + genuine-cascade QAT (kd_ce_alpha={alpha})",
        "lif",
        {
            "deployment_parameters.optimization_driver": "fast",
            "deployment_parameters.lif_blend_fast": True,
            "deployment_parameters.lif_blend_fast_rates": [
                0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0,
            ],
            "deployment_parameters.lif_blend_fast_steps_per_rate": 120,
            "deployment_parameters.cycle_accurate_lif_forward": True,
            "deployment_parameters.fast_ladder_freeze_bn": True,
            "deployment_parameters.lif_theta_cotrain": True,
            "deployment_parameters.kd_ce_alpha": alpha,
            "deployment_parameters.kd_temperature": 4.0,
            "deployment_parameters.lif_blend_fast_stabilize_steps": 200,
        },
        MixerBudgetSchedule(
            ramp_steps=7 * 120,
            recovery_steps=0,
            stabilization_steps=200,
            eval_sample_count=5000,
            max_tuning_wall_s=900.0,
        ),
        probes=("genuine_endpoint",),
        notes=(
            "Round-2 compose: faithful per-channel theta (decode scale, near-lossless "
            "at deployment per the theta-fidelity finding) + the proven KD+CE genuine "
            "QAT objective. Round-1 levers each clustered 0.952-0.968; the hypothesis "
            "is the faithful scale lever and the conversion-faithful objective compound "
            "over 0.97. Rides the SHORT-stabilize fine ladder, never a long polish."
        ),
    )


def recipe_registry() -> dict[str, RecipePreset]:
    """Composable mixer-specific recipe presets for queue generation."""
    return {
        "mixer_lif_fast_minimal": _mixer_recipe(
            "mixer_lif_fast_minimal",
            "Mixer LIF fast minimal",
            "lif",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.lif_blend_fast": True,
                "deployment_parameters.lif_blend_fast_stabilize_steps": 0,
            },
            MixerBudgetSchedule(
                ramp_steps=4 * 120,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=5000,
                max_tuning_wall_s=600.0,
            ),
        ),
        "mixer_lif_fast_stabilized": _mixer_recipe(
            "mixer_lif_fast_stabilized",
            "Mixer LIF fast stabilized",
            "lif",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.lif_blend_fast": True,
                "deployment_parameters.lif_blend_fast_stabilize_steps": 600,
                "deployment_parameters.fast_ladder_freeze_bn": True,
            },
            MixerBudgetSchedule(
                ramp_steps=4 * 120,
                recovery_steps=0,
                stabilization_steps=600,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
        ),
        "mixer_lif_best_known": _mixer_recipe(
            "mixer_lif_best_known",
            "Mixer LIF best known flagged baseline",
            "lif",
            {
                "deployment_parameters.optimization_driver": "controller",
                "deployment_parameters.lif_blend_fast": True,
                "deployment_parameters.lif_blend_fast_stabilize_steps": 1200,
                "deployment_parameters.lif_distmatch": True,
                "deployment_parameters.lif_distmatch_bias_iters": 10,
                "deployment_parameters.tuning_keepbest_certified": True,
                "deployment_parameters.tuning_target_floor_on_real_target": True,
            },
            MixerBudgetSchedule(
                ramp_steps=0,
                recovery_steps=4000,
                stabilization_steps=1200,
                eval_sample_count=5000,
                max_tuning_wall_s=None,
            ),
            notes=(
                "Best known completed LIF line is still below 97%; kept as "
                "flagged baseline for future repair."
            ),
        ),
        "mixer_sync_ttfs_qat_minimal": _mixer_recipe(
            "mixer_sync_ttfs_qat_minimal",
            "Mixer synchronized TTFS QAT minimal",
            "ttfs_cycle_synchronized",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "synchronized",
                "deployment_parameters.ttfs_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 0,
                "deployment_parameters.ttfs_sync_genuine_qat": True,
                "deployment_parameters.nf_scm_parity_samples": 8,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=5000,
                max_tuning_wall_s=600.0,
            ),
            probes=("nf_scm_parity",),
        ),
        "mixer_sync_ttfs_relaxed_parity": _mixer_recipe(
            "mixer_sync_ttfs_relaxed_parity",
            "Mixer synchronized TTFS relaxed parity diagnostic",
            "ttfs_cycle_synchronized",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "synchronized",
                "deployment_parameters.ttfs_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 0,
                "deployment_parameters.ttfs_sync_genuine_qat": True,
                "deployment_parameters.nf_scm_parity_samples": 8,
                "deployment_parameters.nf_scm_parity_max_mismatch_fraction": 0.15,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=5000,
                max_tuning_wall_s=600.0,
            ),
            probes=("nf_scm_parity", "torch_sim_parity"),
            notes=(
                "Diagnostic best known sync line: relaxes documented mixer "
                "per-neuron residual to reach the downstream torch/deployed gate."
            ),
        ),
        "mixer_cascaded_proxy_then_refine_minimal": _mixer_recipe(
            "mixer_cascaded_proxy_then_refine_minimal",
            "Mixer cascaded proxy then refine minimal",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_ste_refine": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 600,
                "deployment_parameters.tuning_full_transform_probe": True,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=600,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            probes=("tuning_full_transform_probe", "proxy_genuine_gap"),
        ),
        "mixer_cascaded_genuine_blend_fast": _mixer_recipe(
            "mixer_cascaded_genuine_blend_fast",
            "Mixer cascaded genuine blend fast",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_genuine_blend_ramp": True,
                "deployment_parameters.ttfs_genuine_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 300,
                "deployment_parameters.tuning_full_transform_probe": True,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=300,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            probes=("tuning_full_transform_probe", "genuine_endpoint"),
        ),
        "mixer_controller_baseline": _mixer_recipe(
            "mixer_controller_baseline",
            "Mixer controller baseline",
            "controller",
            {
                "deployment_parameters.optimization_driver": "controller",
                "deployment_parameters.conversion_policy": False,
            },
            MixerBudgetSchedule(
                ramp_steps=0,
                recovery_steps=4000,
                stabilization_steps=1200,
                eval_sample_count=5000,
                max_tuning_wall_s=None,
            ),
            notes="Diagnostic control only; not promotable unless it also beats timing.",
        ),
        "mixer_ttfs_quantized_q100_fast": _mixer_recipe(
            "mixer_ttfs_quantized_q100_fast",
            "Mixer TTFS quantized q=1.0 fast",
            "ttfs_quantized",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.activation_scale_quantile": 1.0,
                "deployment_parameters.manager_rate_fast_rates": [0.25, 0.5, 0.75, 1.0],
                "deployment_parameters.manager_rate_fast_steps_per_rate": 120,
            },
            MixerBudgetSchedule(
                ramp_steps=4 * 120,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=5000,
                max_tuning_wall_s=600.0,
            ),
            notes=(
                "Best known quantized line: clears 97% accuracy in current "
                "artifacts but remains too slow relative to analytical baseline."
            ),
        ),
        "mixer_ttfs_quantized_q100_fast_timing": _mixer_recipe(
            "mixer_ttfs_quantized_q100_fast_timing",
            "Mixer TTFS quantized q=1.0 fast timing repair",
            "ttfs_quantized",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.activation_scale_quantile": 1.0,
                "deployment_parameters.manager_rate_fast_rates": [0.5, 0.75, 1.0],
                "deployment_parameters.manager_rate_fast_steps_per_rate": 90,
                "deployment_parameters.tuning_budget_scale": 0.75,
            },
            MixerBudgetSchedule(
                ramp_steps=3 * 90,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=2000,
                max_tuning_wall_s=450.0,
            ),
            notes=(
                "Timing repair candidate: fewer AQ ramp steps and smaller "
                "eval subsample while preserving q=1.0 fast accuracy path."
            ),
        ),
        "mixer_lif_genuine_qat": _lif_genuine_qat_recipe(
            "mixer_lif_genuine_qat", alpha=0.5
        ),
        "mixer_lif_genuine_qat_a03": _lif_genuine_qat_recipe(
            "mixer_lif_genuine_qat_a03", alpha=0.3
        ),
        "mixer_lif_genuine_qat_a07": _lif_genuine_qat_recipe(
            "mixer_lif_genuine_qat_a07", alpha=0.7
        ),
        "mixer_sync_ttfs_faithfulness_repair": _mixer_recipe(
            "mixer_sync_ttfs_faithfulness_repair",
            "Mixer synchronized TTFS faithfulness + genuine-QAT repair",
            "ttfs_cycle_synchronized",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "synchronized",
                "deployment_parameters.ttfs_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 300,
                "deployment_parameters.ttfs_sync_genuine_qat": True,
                "deployment_parameters.fast_ladder_freeze_bn": True,
                "deployment_parameters.nf_scm_parity_samples": 8,
                "deployment_parameters.nf_scm_parity_max_mismatch_fraction": 0.15,
                "deployment_parameters.scm_torch_sim_parity_min_agreement": 0.96,
                "deployment_parameters.kd_ce_alpha": 0.5,
                "deployment_parameters.kd_temperature": 4.0,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=300,
                eval_sample_count=5000,
                max_tuning_wall_s=700.0,
            ),
            probes=("nf_scm_parity", "torch_sim_parity"),
            notes=(
                "Primary sync repair: relaxes the honest mixer weight-quant parity "
                "residual (NF proven bit-exact, transpose wiring sound) to the 0.15 "
                "documented mixer floor, and adds BN-frozen genuine QAT to recover "
                "the ~1-3pp accuracy gap to the 0.97 gate."
            ),
        ),
        "mixer_cascaded_genuine_qat": _mixer_recipe(
            "mixer_cascaded_genuine_qat",
            "Mixer cascaded genuine-cascade QAT conversion repair",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_genuine_blend_ramp": True,
                "deployment_parameters.ttfs_genuine_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 600,
                "deployment_parameters.fast_ladder_freeze_bn": True,
                "deployment_parameters.conversion_policy": True,
                "deployment_parameters.tuning_enable_characterization": True,
                "deployment_parameters.tuning_full_transform_probe": True,
                "deployment_parameters.kd_ce_alpha": 0.5,
                "deployment_parameters.kd_temperature": 4.0,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=600,
                eval_sample_count=5000,
                max_tuning_wall_s=1000.0,
            ),
            probes=("tuning_full_transform_probe", "genuine_endpoint"),
            notes=(
                "Cascaded conversion repair: genuine-blend cascade + BN-frozen QAT "
                "+ conversion policy/characterization to lift the 0.92-0.95 plateau "
                "above the 0.97 gate."
            ),
        ),
        "mixer_ttfs_quantized_q100_aggressive_timing": _mixer_recipe(
            "mixer_ttfs_quantized_q100_aggressive_timing",
            "Mixer TTFS quantized q=1.0 aggressive timing repair",
            "ttfs_quantized",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.activation_scale_quantile": 1.0,
                "deployment_parameters.manager_rate_fast_rates": [0.75, 1.0],
                "deployment_parameters.manager_rate_fast_steps_per_rate": 60,
                "deployment_parameters.tuning_budget_scale": 0.5,
            },
            MixerBudgetSchedule(
                ramp_steps=2 * 60,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=1000,
                max_tuning_wall_s=350.0,
            ),
            notes=(
                "Aggressive timing repair: 2 AQ rates x 60 steps + 0.5 budget "
                "scale to drop under the 428s analytical wall while holding the "
                "q=1.0 accuracy path."
            ),
        ),
        "mixer_ttfs_analytical_control": _mixer_recipe(
            "mixer_ttfs_analytical_control",
            "Mixer analytical TTFS timing/accuracy control",
            "ttfs_analytical",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.spiking_mode": "ttfs",
            },
            MixerBudgetSchedule(
                ramp_steps=4 * 120,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=5000,
                max_tuning_wall_s=600.0,
            ),
            notes=(
                "Immutable analytical control for relative timing and near-lossless "
                "accuracy reference on each mixer workload family."
            ),
        ),
        # -- SOLUTION-study wave: cascaded-collapse revival ---------------------
        # The cascaded genuine-QAT collapse is a REGRESSION: conversion_policy
        # silently vetoes the proven fast genuine-blend driver and routes the deep
        # cascade through the controller (the controller_baseline failure mode).
        "mixer_cascaded_policy_isolate": _mixer_recipe(
            "mixer_cascaded_policy_isolate",
            "Mixer cascaded conversion-policy trigger isolate",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_genuine_blend_ramp": True,
                "deployment_parameters.ttfs_genuine_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 300,
                "deployment_parameters.tuning_full_transform_probe": True,
                "deployment_parameters.conversion_policy": True,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=300,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            probes=("tuning_full_transform_probe", "genuine_endpoint"),
            notes=(
                "Adversarial isolate: the proven genuine_blend_fast base + "
                "conversion_policy ONLY. A collapse here pins the regression on that "
                "single knob (the silent fast->controller driver veto), not on "
                "BN-freeze / kd reweight / long stabilize."
            ),
        ),
        "mixer_cascaded_blend_theta": _mixer_recipe(
            "mixer_cascaded_blend_theta",
            "Mixer cascaded genuine-blend + per-channel theta co-train",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_genuine_blend_ramp": True,
                "deployment_parameters.ttfs_genuine_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 300,
                "deployment_parameters.ttfs_theta_cotrain": True,
                "deployment_parameters.tuning_full_transform_probe": True,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=300,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            probes=("tuning_full_transform_probe", "genuine_endpoint"),
            notes=(
                "Revival: the surviving fast genuine-blend ramp (NO conversion_policy) "
                "plus per-channel TRAINABLE theta co-trained through the cascade to "
                "lift the 0.92 plateau toward the 0.97 gate, non-destructively."
            ),
        ),
        "mixer_cascaded_staircase_ste_theta": _mixer_recipe(
            "mixer_cascaded_staircase_ste_theta",
            "Mixer cascaded staircase-STE fast + theta (near-lossless recipe)",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_staircase_ste": True,
                "deployment_parameters.ttfs_staircase_ste_fast": True,
                "deployment_parameters.ttfs_theta_cotrain": True,
                "deployment_parameters.tuning_full_transform_probe": True,
            },
            MixerBudgetSchedule(
                ramp_steps=1000,
                recovery_steps=0,
                stabilization_steps=0,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            probes=("tuning_full_transform_probe", "genuine_endpoint"),
            notes=(
                "Revival via the proven near-lossless deep-cascade recipe: the "
                "staircase-STE fast loop (clean staircase-gradient hedge, split "
                "weight/theta LR, progressive shallow->deep unfreeze) co-trained with "
                "per-channel theta. No conversion_policy, no controller."
            ),
        ),
        # -- SOLUTION-study wave: LIF gradual non-destructive adaptation --------
        # Close the ~1.5-2.5pp LIF rate-coding gap with PROPER gradual adaptation
        # (finer ramp, clip-bias/distmatch, the golden controller ramp) — NOT a
        # lengthy stabilize. Stabilization stays SHORT (a stat-tracking polish).
        "mixer_lif_fine_ladder": _mixer_recipe(
            "mixer_lif_fine_ladder",
            "Mixer LIF finer non-destructive rate ladder",
            "lif",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.lif_blend_fast": True,
                "deployment_parameters.lif_blend_fast_rates": [
                    0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0,
                ],
                "deployment_parameters.lif_blend_fast_steps_per_rate": 120,
                "deployment_parameters.cycle_accurate_lif_forward": True,
                "deployment_parameters.fast_ladder_freeze_bn": True,
                "deployment_parameters.lif_blend_fast_stabilize_steps": 200,
            },
            MixerBudgetSchedule(
                ramp_steps=7 * 120,
                recovery_steps=0,
                stabilization_steps=200,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            notes=(
                "Densifies the rate ladder near 1.0 (where the LIF quantization "
                "staircase cliff lives) so per-rung KD recovery absorbs the "
                "rate-quantization gradually instead of over a 0.25 cliff. SHORT "
                "stabilize (200) — the lever is the finer ramp, not a long polish."
            ),
        ),
        "mixer_lif_q100_distmatch": _mixer_recipe(
            "mixer_lif_q100_distmatch",
            "Mixer LIF q=1.0 clip-bias + distribution match",
            "lif",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.lif_blend_fast": True,
                "deployment_parameters.lif_blend_fast_rates": [
                    0.25, 0.5, 0.75, 0.9, 1.0,
                ],
                "deployment_parameters.lif_blend_fast_steps_per_rate": 120,
                "deployment_parameters.activation_scale_quantile": 1.0,
                "deployment_parameters.lif_distmatch": True,
                "deployment_parameters.lif_distmatch_bias_iters": 15,
                "deployment_parameters.cycle_accurate_lif_forward": True,
                "deployment_parameters.fast_ladder_freeze_bn": True,
                "deployment_parameters.lif_blend_fast_stabilize_steps": 200,
            },
            MixerBudgetSchedule(
                ramp_steps=5 * 120,
                recovery_steps=0,
                stabilization_steps=200,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            notes=(
                "Removes the 0.99-quantile firing-threshold clip bias (q=1.0 raises "
                "theta -> fewer spikes -> energy win) and DFQ-matches the deployed "
                "cascade's first moment to the teacher. Cheapest config-only "
                "accuracy+energy probe; control for the per-channel theta lever."
            ),
        ),
        "mixer_lif_controller_gradual": _mixer_recipe(
            "mixer_lif_controller_gradual",
            "Mixer LIF golden controller ramp (bounded short stabilize)",
            "lif",
            {
                "deployment_parameters.optimization_driver": "controller",
                "deployment_parameters.cycle_accurate_lif_forward": True,
                "deployment_parameters.tuning_refind_lr_on_miss": True,
                "deployment_parameters.tuning_recovery_lr_plateau": True,
                "deployment_parameters.tuning_tight_plateau": True,
                "deployment_parameters.tuning_recovery_check_divisor": 2,
                "deployment_parameters.tuning_keepbest_certified": True,
                "deployment_parameters.tuning_stabilization_bounded": True,
                "deployment_parameters.tuning_stabilization_ratio": 0.25,
                "deployment_parameters.kd_temperature": 4.0,
            },
            MixerBudgetSchedule(
                ramp_steps=0,
                recovery_steps=1500,
                stabilization_steps=300,
                eval_sample_count=5000,
                max_tuning_wall_s=1200.0,
            ),
            notes=(
                "The golden non-destructive ramp: the controller's fine 0.125 ladder "
                "with per-cycle KD recovery + rollback, made higher-QUALITY (LR "
                "refind-on-miss, plateau LR ladder, tight plateau, keep-best "
                "certified) rather than LONGER. Bounded SHORT stabilize (ratio 0.25) "
                "vs the 1200-step best_known baseline."
            ),
        ),
        "mixer_lif_theta_cotrain": _mixer_recipe(
            "mixer_lif_theta_cotrain",
            "Mixer LIF per-channel theta co-train (the #1 lever)",
            "lif",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.lif_blend_fast": True,
                "deployment_parameters.lif_blend_fast_rates": [
                    0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0,
                ],
                "deployment_parameters.lif_blend_fast_steps_per_rate": 120,
                "deployment_parameters.cycle_accurate_lif_forward": True,
                "deployment_parameters.fast_ladder_freeze_bn": True,
                "deployment_parameters.lif_theta_cotrain": True,
                "deployment_parameters.lif_blend_fast_stabilize_steps": 200,
            },
            MixerBudgetSchedule(
                ramp_steps=7 * 120,
                recovery_steps=0,
                stabilization_steps=200,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            notes=(
                "Rebinds each non-encoding perceptron's single-scalar firing threshold "
                "to a per-output-channel TRAINABLE theta so the gradual blend ramp "
                "co-optimises the firing-gain WITH the weights (a scalar threshold "
                "cannot serve wide+narrow channels). The principled LIF lever per WS3 "
                "4.2; rides the SHORT-stabilize fine ladder, no long polish."
            ),
        ),
        "mixer_lif_theta_distmatch": _mixer_recipe(
            "mixer_lif_theta_distmatch",
            "Mixer LIF per-channel theta + DFQ distribution match",
            "lif",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.lif_blend_fast": True,
                "deployment_parameters.lif_blend_fast_rates": [
                    0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0,
                ],
                "deployment_parameters.lif_blend_fast_steps_per_rate": 120,
                "deployment_parameters.activation_scale_quantile": 1.0,
                "deployment_parameters.lif_theta_cotrain": True,
                "deployment_parameters.lif_distmatch": True,
                "deployment_parameters.lif_distmatch_bias_iters": 15,
                "deployment_parameters.cycle_accurate_lif_forward": True,
                "deployment_parameters.fast_ladder_freeze_bn": True,
                "deployment_parameters.lif_blend_fast_stabilize_steps": 200,
            },
            MixerBudgetSchedule(
                ramp_steps=7 * 120,
                recovery_steps=0,
                stabilization_steps=200,
                eval_sample_count=5000,
                max_tuning_wall_s=900.0,
            ),
            notes=(
                "Composes the two orthogonal principled levers: per-channel trainable "
                "theta (scale) co-trained by the ramp + DFQ per-neuron bias match to "
                "the teacher's first moment, with the q=1.0 clip-bias removal. Tests "
                "whether scale-cotrain and bias-correction compound on the gap."
            ),
        ),
        "mixer_lif_theta_qat": _lif_theta_qat_recipe(
            "mixer_lif_theta_qat", alpha=0.5
        ),
        "mixer_lif_theta_qat_a03": _lif_theta_qat_recipe(
            "mixer_lif_theta_qat_a03", alpha=0.3
        ),
        "mixer_cascaded_blend_fast_finer": _mixer_recipe(
            "mixer_cascaded_blend_fast_finer",
            "Mixer cascaded genuine blend — finer rate→1.0 ramp + longer per-rung recovery",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_genuine_blend_ramp": True,
                "deployment_parameters.ttfs_genuine_blend_fast": True,
                "deployment_parameters.ttfs_blend_fast_rates": [
                    0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 1.0,
                ],
                "deployment_parameters.ttfs_blend_fast_steps_per_rate": 160,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 200,
                "deployment_parameters.tuning_full_transform_probe": True,
            },
            MixerBudgetSchedule(
                ramp_steps=7 * 160,
                recovery_steps=0,
                stabilization_steps=200,
                eval_sample_count=5000,
                max_tuning_wall_s=1200.0,
            ),
            probes=("tuning_full_transform_probe", "genuine_endpoint"),
            notes=(
                "Round-2 cascaded revival. The ONLY surviving faithful cascaded path is "
                "the fast genuine-blend ladder (genuine_blend_fast: 0.94 @ parity "
                "0.9961); genuine_qat / policy_isolate collapse THROUGH the controller. "
                "This stays on the fast driver and ramps FINER through the rate→1.0 "
                "collapse zone (7 rungs, 2 in [0.9,1.0)) with longer per-rung recovery, "
                "to lift the faithful ladder above ~0.94 without re-entering the "
                "controller collapse."
            ),
        ),
        "mixer_cascaded_blend_fast_kd": _mixer_recipe(
            "mixer_cascaded_blend_fast_kd",
            "Mixer cascaded genuine blend — finer ramp + stronger KD anchoring",
            "ttfs_cycle_cascaded",
            {
                "deployment_parameters.optimization_driver": "fast",
                "deployment_parameters.ttfs_cycle_schedule": "cascaded",
                "deployment_parameters.ttfs_genuine_blend_ramp": True,
                "deployment_parameters.ttfs_genuine_blend_fast": True,
                "deployment_parameters.ttfs_genuine_blend_ce_alpha": 0.1,
                "deployment_parameters.ttfs_blend_fast_rates": [
                    0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 1.0,
                ],
                "deployment_parameters.ttfs_blend_fast_steps_per_rate": 160,
                "deployment_parameters.ttfs_blend_fast_stabilize_steps": 200,
                "deployment_parameters.tuning_full_transform_probe": True,
            },
            MixerBudgetSchedule(
                ramp_steps=7 * 160,
                recovery_steps=0,
                stabilization_steps=200,
                eval_sample_count=5000,
                max_tuning_wall_s=1200.0,
            ),
            probes=("tuning_full_transform_probe", "genuine_endpoint"),
            notes=(
                "Round-2 cascaded revival, KD-anchored. Same finer ramp as "
                "blend_fast_finer but lowers ttfs_genuine_blend_ce_alpha 0.3->0.1 so the "
                "teacher anchors the genuine cascade tighter across the rate→1.0 "
                "collapse zone (more KD, less free CE drift)."
            ),
        ),
    }


def build_mnist_mixer_manifest(seeds: Sequence[int] = (0, 1, 2)) -> DiagnosticManifest:
    return build_mixer_manifest(
        manifest_id="mnist_mlp_mixer_core_diagnostics",
        dataset="MNIST_DataProvider",
        template_prefix="mnist_mmixcore",
        seeds=seeds,
    )


def build_mixer_manifest(
    *,
    manifest_id: str,
    dataset: str,
    template_prefix: str,
    seeds: Sequence[int] = (0, 1, 2),
) -> DiagnosticManifest:
    """Diagnostic manifest for mlp_mixer_core on a digit dataset."""
    return DiagnosticManifest(
        manifest_id=manifest_id,
        seeds=tuple(int(s) for s in seeds),
        cells=(
            DiagnosticCell(
                f"{template_prefix}_lif",
                f"templates/{template_prefix}_matrix_1_lif_rate.json",
                "lif",
                "none",
                "diagnostic",
                (
                    "mixer_lif_genuine_qat",
                    "mixer_lif_genuine_qat_a03",
                    "mixer_lif_genuine_qat_a07",
                    "mixer_lif_fast_stabilized",
                    "mixer_lif_fine_ladder",
                    "mixer_lif_q100_distmatch",
                    "mixer_lif_controller_gradual",
                    "mixer_lif_theta_cotrain",
                    "mixer_lif_theta_distmatch",
                    "mixer_lif_theta_qat",
                    "mixer_lif_theta_qat_a03",
                ),
                dataset=dataset,
            ),
            DiagnosticCell(
                f"{template_prefix}_ttfs_cycle_synchronized",
                f"templates/{template_prefix}_matrix_7_ttfs_cycle_synchronized.json",
                "ttfs_cycle_based",
                "synchronized",
                "diagnostic",
                (
                    "mixer_sync_ttfs_faithfulness_repair",
                    "mixer_sync_ttfs_relaxed_parity",
                ),
                dataset=dataset,
            ),
            DiagnosticCell(
                f"{template_prefix}_ttfs_cycle_cascaded",
                f"templates/{template_prefix}_matrix_6_ttfs_cycle_cascaded.json",
                "ttfs_cycle_based",
                "cascaded",
                "diagnostic",
                (
                    "mixer_cascaded_genuine_qat",
                    "mixer_cascaded_genuine_blend_fast",
                    "mixer_cascaded_policy_isolate",
                    "mixer_cascaded_blend_theta",
                    "mixer_cascaded_staircase_ste_theta",
                    "mixer_cascaded_blend_fast_finer",
                    "mixer_cascaded_blend_fast_kd",
                ),
                dataset=dataset,
            ),
            DiagnosticCell(
                f"{template_prefix}_ttfs_analytical_control",
                f"templates/{template_prefix}_matrix_4_ttfs_analytical.json",
                "ttfs",
                "analytical",
                "analytical_control",
                ("mixer_ttfs_analytical_control",),
                dataset=dataset,
            ),
            DiagnosticCell(
                f"{template_prefix}_ttfs_quantized_control",
                f"templates/{template_prefix}_matrix_5_ttfs_quantized_offload.json",
                "ttfs_quantized",
                "analytical",
                "diagnostic",
                (
                    "mixer_ttfs_quantized_q100_aggressive_timing",
                    "mixer_ttfs_quantized_q100_fast_timing",
                ),
                dataset=dataset,
            ),
        ),
    )


def build_fmnist_mixer_manifest(seeds: Sequence[int] = (0, 1, 2)) -> DiagnosticManifest:
    return build_mixer_manifest(
        manifest_id="fmnist_mlp_mixer_core_diagnostics",
        dataset="FashionMNIST_DataProvider",
        template_prefix="fmnist_mmixcore",
        seeds=seeds,
    )


def build_kmnist_mixer_manifest(seeds: Sequence[int] = (0, 1, 2)) -> DiagnosticManifest:
    return build_mixer_manifest(
        manifest_id="kmnist_mlp_mixer_core_diagnostics",
        dataset="KMNIST_DataProvider",
        template_prefix="kmnist_mmixcore",
        seeds=seeds,
    )


@dataclass(frozen=True)
class WorkloadDiagnosticCell:
    cell_id: str
    model_type: str
    dataset: str
    template: str
    spiking_mode: str
    schedule: str
    role: str
    recipe_ids: tuple[str, ...]
    depth: int = 4

    def axes(self) -> dict[str, Any]:
        return {
            "vehicle": self.model_type,
            "dataset": self.dataset,
            "firing": self.spiking_mode,
            "sync": self.schedule,
            "backend": "sanafe",
            "depth": self.depth,
        }


@dataclass(frozen=True)
class WorkloadDiagnosticManifest:
    manifest_id: str
    cells: tuple[WorkloadDiagnosticCell, ...]
    seeds: tuple[int, ...]
    acceptance: AcceptanceGate = field(default_factory=AcceptanceGate)


def build_lenet5_digit_manifest(seeds: Sequence[int] = (0,)) -> WorkloadDiagnosticManifest:
    return WorkloadDiagnosticManifest(
        manifest_id="lenet5_digit_diagnostics",
        seeds=tuple(int(s) for s in seeds),
        cells=(
            WorkloadDiagnosticCell(
                "mnist_lenet5_lif",
                "lenet5",
                "MNIST_DataProvider",
                "templates/mnist_lenet5_lif.json",
                "lif",
                "none",
                "diagnostic",
                ("mixer_lif_best_known",),
            ),
            WorkloadDiagnosticCell(
                "mnist_lenet5_ttfs_analytical",
                "lenet5",
                "MNIST_DataProvider",
                "templates/mnist_lenet5_ttfs.json",
                "ttfs",
                "analytical",
                "analytical_control",
                ("mixer_ttfs_analytical_control",),
            ),
        ),
    )


def build_deep_cnn_digit_manifest(seeds: Sequence[int] = (0,)) -> WorkloadDiagnosticManifest:
    datasets = (
        ("mnist", "MNIST_DataProvider"),
        ("fmnist", "FashionMNIST_DataProvider"),
        ("kmnist", "KMNIST_DataProvider"),
    )
    cells: list[WorkloadDiagnosticCell] = []
    for prefix, provider in datasets:
        cells.extend(
            [
                WorkloadDiagnosticCell(
                    f"{prefix}_deep_cnn_sync_d4",
                    "deep_cnn",
                    provider,
                    f"templates/{prefix}_deep_cnn_sync_d4.json",
                    "ttfs_cycle_based",
                    "synchronized",
                    "diagnostic",
                    ("mixer_sync_ttfs_relaxed_parity",),
                    depth=4,
                ),
                WorkloadDiagnosticCell(
                    f"{prefix}_deep_cnn_ttfs_analytical",
                    "deep_cnn",
                    provider,
                    f"templates/{prefix}_deep_cnn_ttfs_analytical_d4.json",
                    "ttfs",
                    "analytical",
                    "analytical_control",
                    ("mixer_ttfs_analytical_control",),
                    depth=4,
                ),
            ]
        )
    return WorkloadDiagnosticManifest(
        manifest_id="deep_cnn_digit_diagnostics",
        seeds=tuple(int(s) for s in seeds),
        cells=tuple(cells),
    )


def recipe_is_certified_for_promotion(
    recipe: RecipePreset,
    rows: Sequence[Mapping[str, Any]],
    *,
    required_seeds: Sequence[int] | None = None,
) -> bool:
    """Return True only when all required seeds pass the recipe acceptance gate."""
    required = tuple(required_seeds or ())
    gate = recipe.acceptance
    by_seed = {
        int(row.get("seed", -1)): row
        for row in rows
        if row.get("recipe_id") == recipe.recipe_id
    }
    seeds = required or tuple(sorted(by_seed))
    if not seeds:
        return False
    return all(
        gate.evaluate(by_seed[seed]).accepted
        for seed in seeds
        if seed in by_seed
    ) and len([seed for seed in seeds if seed in by_seed]) == len(seeds)


def default_recipe_for_cell(
    cell: DiagnosticCell,
    measured_rows: Sequence[Mapping[str, Any]],
    *,
    required_seeds: Sequence[int] | None = None,
) -> str | None:
    """Pick the first certified recipe for a cell; never promote without evidence."""
    registry = recipe_registry()
    for recipe_id in cell.recipe_ids:
        recipe = registry[recipe_id]
        if recipe_is_certified_for_promotion(
            recipe,
            measured_rows,
            required_seeds=required_seeds,
        ):
            return recipe_id
    return None


def planned_step_metrics() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "status": "planned",
            "metrics": {},
            "timing": {"wall_s": None, "relative_to_baseline": None},
        }
        for name in MIXER_DIAGNOSTIC_STEP_NAMES
    ]


def planned_mnist_mixer_ledger_row(
    *,
    run_id: str,
    cell: DiagnosticCell,
    recipe: RecipePreset,
    seed: int,
) -> dict[str, Any]:
    row = {
        "row_type": "planned",
        "study": "MNIST_MIXER_DIAGNOSTICS",
        "cluster": "MNIST_MIXER_CLOSURE",
        "run_id": run_id,
        "cell_id": cell.cell_id,
        "recipe_id": recipe.recipe_id,
        "seed": int(seed),
        "axes": cell.axes(),
        "acceptance": cell.acceptance.to_dict(),
        "budget_schedule": recipe.budget_schedule.to_dict(),
        "step_metrics": planned_step_metrics(),
        "probes": {
            "proxy_genuine": {
                "required": "tuning_full_transform_probe" in recipe.required_probes,
                "status": "planned",
                "metrics": {},
            },
            "decision_trace": {"status": "planned"},
            "conversion_decision": {
                "driver": recipe.base_overrides.get(
                    "deployment_parameters.optimization_driver"
                ),
                "recipe": recipe.recipe_id,
                "escalated": None,
                "characterized": None,
                "reason": None,
            },
        },
    }
    return normalize_planned_ledger_row(row)
