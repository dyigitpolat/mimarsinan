"""DeploymentPlan: the single resolver for the deployment_parameters config.

Locks the precedence / derivation rules (defaults, pruning_enabled,
schedule-derived spiking booleans, the budget default) and asserts the plan is
byte-identical to the inline ``config.get(...)`` reads it replaces.
"""

import re
from pathlib import Path

import pytest

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


def _resolve(**cfg):
    return DeploymentPlan.resolve(dict(cfg))


class TestDefaults:
    def test_empty_config_defaults(self):
        p = _resolve()
        assert p.spiking_mode == "lif"
        assert p.ttfs_cycle_schedule == "cascaded"
        assert p.activation_quantization is False
        assert p.weight_quantization is False
        assert p.enable_training_noise is False
        assert p.cycle_accurate_lif_forward is False
        assert p.pruning is False
        assert p.pruning_fraction == 0.0
        assert p.pruning_enabled is False
        assert p.enable_nevresim_simulation is True
        assert p.enable_loihi_simulation is False
        assert p.enable_sanafe_simulation is False
        assert p.cuda_debug is False
        assert p.deployment_metric_full_eval is True
        assert p.max_simulation_samples == 0
        assert p.simulation_batch_count is None
        assert p.simulation_batch_size == 8
        assert p.seed == 0
        assert p.weight_source is None
        assert p.model_type == ""

    def test_does_not_require_simulation_steps(self):
        # The step planner resolves a plan from a config without sim length.
        p = _resolve(spiking_mode="ttfs_cycle_based")
        assert p.requires_ttfs_firing is True


class TestSpikingDerived:
    def test_requires_ttfs_firing(self):
        assert _resolve(spiking_mode="lif").requires_ttfs_firing is False
        for m in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert _resolve(spiking_mode=m).requires_ttfs_firing is True

    def test_schedule_normalized_and_default(self):
        assert _resolve().ttfs_cycle_schedule == "cascaded"
        assert _resolve(ttfs_cycle_schedule="bogus").ttfs_cycle_schedule == "cascaded"
        assert (
            _resolve(ttfs_cycle_schedule="synchronized").ttfs_cycle_schedule
            == "synchronized"
        )

    def test_synchronized_only_for_ttfs_cycle(self):
        assert _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized"
        ).is_synchronized_ttfs is True
        assert _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded"
        ).is_synchronized_ttfs is False
        # schedule is inert for non-cycle modes.
        assert _resolve(
            spiking_mode="ttfs", ttfs_cycle_schedule="synchronized"
        ).is_synchronized_ttfs is False

    def test_is_ttfs_cycle_based(self):
        assert _resolve(spiking_mode="ttfs_cycle_based").is_ttfs_cycle_based is True
        assert _resolve(spiking_mode="lif").is_ttfs_cycle_based is False

    def test_is_cascaded_ttfs_is_the_synchronized_complement(self):
        # is_cascaded_ttfs is the greedy-schedule cell within the cycle family —
        # the complement of is_synchronized_ttfs (drives TTFSCycleAdaptationStep).
        assert _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded"
        ).is_cascaded_ttfs is True
        assert _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized"
        ).is_cascaded_ttfs is False
        # default schedule is cascaded; non-cycle modes are never cascaded_ttfs.
        assert _resolve(spiking_mode="ttfs_cycle_based").is_cascaded_ttfs is True
        assert _resolve(spiking_mode="lif").is_cascaded_ttfs is False

    def test_uses_ttfs_floor_ceil_convention(self):
        # The floor+half-step-bias convention: ttfs_quantized AND the synchronized
        # floor-collapse train the floor NF and deploy the mode-derived ceil kernel.
        assert _resolve(spiking_mode="ttfs_quantized").uses_ttfs_floor_ceil_convention
        assert _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized"
        ).uses_ttfs_floor_ceil_convention
        # cascaded, continuous ttfs, and lif keep their own conventions.
        assert not _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded"
        ).uses_ttfs_floor_ceil_convention
        assert not _resolve(spiking_mode="ttfs").uses_ttfs_floor_ceil_convention
        assert not _resolve(spiking_mode="lif").uses_ttfs_floor_ceil_convention


class TestNovenaChipFaithfulGate:
    """Novena's zero-reset needs the cycle-accurate cascade forward to deploy
    faithfully; the analytical rate forward (cycle_accurate_lif_forward=False)
    diverges from the deployed HCM (~12pp on mmixcore). Resolve fails loud (V3)."""

    def test_novena_without_cycle_accurate_raises(self):
        with pytest.raises(ValueError, match="cycle_accurate_lif_forward"):
            _resolve(
                spiking_mode="lif",
                firing_mode="Novena",
                thresholding_mode="<",
                cycle_accurate_lif_forward=False,
            )

    def test_novena_with_cycle_accurate_resolves(self):
        p = _resolve(
            spiking_mode="lif",
            firing_mode="Novena",
            thresholding_mode="<",
            cycle_accurate_lif_forward=True,
        )
        assert p.cycle_accurate_lif_forward is True

    def test_novena_default_cycle_accurate_resolves(self):
        # Omitting the key relies on the LIF default (chip-faithful cascade); the
        # gate must not reject it.
        p = _resolve(spiking_mode="lif", firing_mode="Novena", thresholding_mode="<")
        assert p.config["firing_mode"] == "Novena"

    def test_default_firing_rate_forward_unaffected(self):
        p = _resolve(
            spiking_mode="lif",
            firing_mode="Default",
            cycle_accurate_lif_forward=False,
        )
        assert p.cycle_accurate_lif_forward is False

    def test_ttfs_family_not_gated_by_lif_rule(self):
        # TTFS family fires with firing_mode='TTFS'; the LIF Novena rule is inert.
        p = _resolve(
            spiking_mode="ttfs_cycle_based",
            firing_mode="TTFS",
            cycle_accurate_lif_forward=False,
        )
        assert p.is_ttfs_cycle_based is True


class TestPruningDerivation:
    def test_pruning_enabled_requires_positive_fraction(self):
        assert _resolve(pruning=True, pruning_fraction=0.0).pruning_enabled is False
        assert _resolve(pruning=True, pruning_fraction=0.3).pruning_enabled is True
        assert _resolve(pruning=False, pruning_fraction=0.3).pruning_enabled is False


class TestToleranceAndBudget:
    def test_budget_defaults_to_twice_tolerance(self):
        assert _resolve(degradation_tolerance=0.05).degradation_budget_total == 0.10
        assert _resolve(degradation_tolerance=0.03).degradation_budget_total == 0.06

    def test_explicit_budget_overrides_default(self):
        assert (
            _resolve(degradation_tolerance=0.05, degradation_budget_total=0.42)
            .degradation_budget_total == 0.42
        )

    def test_scm_tolerance_optional(self):
        assert _resolve().scm_degradation_tolerance is None
        assert _resolve(scm_degradation_tolerance=0.02).scm_degradation_tolerance == 0.02


class TestModelIdentity:
    def test_model_name_falls_back_to_model_type(self):
        assert _resolve(model_type="mlp_mixer").model_name == "mlp_mixer"
        assert (
            _resolve(model_type="mlp_mixer", model_name="my_run").model_name == "my_run"
        )


class TestByteIdentityWithInlineReads:
    """Each resolved field must equal the inline read it replaces."""

    _CFGS = [
        {},
        {"spiking_mode": "ttfs", "activation_quantization": True},
        {
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
            "weight_quantization": True,
            "pruning": True,
            "pruning_fraction": 0.5,
            "enable_loihi_simulation": False,
            "enable_sanafe_simulation": True,
            "enable_nevresim_simulation": False,
            "enable_training_noise": True,
        },
        {
            "degradation_tolerance": 0.07,
            "scm_degradation_tolerance": 0.02,
            "cuda_debug": True,
            "deployment_metric_full_eval": False,
            "max_simulation_samples": 256,
            "simulation_batch_count": 4,
            "simulation_batch_size": 16,
            "seed": 7,
            "weight_source": "/tmp/w.pt",
            "model_type": "torch_custom",
            "cycle_accurate_lif_forward": True,
        },
    ]

    @pytest.mark.parametrize("cfg", _CFGS)
    def test_matches_inline_reads(self, cfg):
        p = DeploymentPlan.resolve(cfg)
        g = cfg.get
        assert p.spiking_mode == g("spiking_mode", "lif")
        assert p.activation_quantization == bool(g("activation_quantization", False))
        assert p.weight_quantization == bool(g("weight_quantization", False))
        assert p.pruning == g("pruning", False)
        assert p.pruning_fraction == float(g("pruning_fraction", 0.0))
        assert p.weight_source == g("weight_source")
        assert p.model_type == g("model_type", "")
        assert p.enable_loihi_simulation == bool(g("enable_loihi_simulation", False))
        assert p.enable_sanafe_simulation == bool(g("enable_sanafe_simulation", False))
        assert p.enable_nevresim_simulation == bool(
            g("enable_nevresim_simulation", True)
        )
        assert p.enable_training_noise == bool(g("enable_training_noise", False))
        assert p.cuda_debug == bool(g("cuda_debug", False))
        assert p.simulation_batch_size == int(g("simulation_batch_size", 8))
        assert p.seed == int(g("seed", 0))


class TestOptimizationDriverAxis:
    """E2 — the pipeline-wide ``controller | fast`` optimization-driver axis.

    Default ``controller`` ⇒ byte-identical. Explicit ``optimization_driver`` wins;
    else a legacy per-family fast switch (lif_blend_fast / ttfs_*_fast) is honoured
    so a plan resolved from an existing config reports the driver the tuner runs.
    """

    def test_default_is_controller(self):
        p = _resolve()
        assert p.optimization_driver == "controller"
        assert p.is_fast_driver is False

    def test_explicit_fast(self):
        p = _resolve(optimization_driver="fast")
        assert p.optimization_driver == "fast"
        assert p.is_fast_driver is True

    def test_explicit_controller(self):
        p = _resolve(optimization_driver="controller")
        assert p.optimization_driver == "controller"
        assert p.is_fast_driver is False

    def test_case_insensitive(self):
        assert _resolve(optimization_driver="FAST").optimization_driver == "fast"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            _resolve(optimization_driver="bisect")

    @pytest.mark.parametrize(
        "switch",
        ["lif_blend_fast", "ttfs_genuine_blend_fast", "ttfs_blend_fast"],
    )
    def test_legacy_fast_switch_implies_fast(self, switch):
        assert _resolve(**{switch: True}).optimization_driver == "fast"

    def test_explicit_controller_overrides_legacy_switch(self):
        # An explicit axis wins over a stray legacy switch.
        p = _resolve(optimization_driver="controller", lif_blend_fast=True)
        assert p.optimization_driver == "controller"

    def test_legacy_switch_off_stays_controller(self):
        assert _resolve(lif_blend_fast=False).optimization_driver == "controller"


class TestTemporalAllocationAxis:
    """EW1 — the RESERVED per-layer-S temporal-allocation axis.

    Default ``s_allocation='uniform'`` ⇒ the SAME global ``simulation_steps`` for every
    cascade depth ⇒ byte-identical. ``explicit`` validates a declared per-depth list;
    ``budget`` is a no-op returning uniform + a deferred marker. Nothing threads the
    map into the forwards/sim yet (research).
    """

    def test_default_axis_is_uniform(self):
        p = _resolve()
        assert p.s_allocation == "uniform"

    def test_explicit_and_budget_axis_resolved(self):
        assert _resolve(s_allocation="explicit").s_allocation == "explicit"
        assert _resolve(s_allocation="budget").s_allocation == "budget"

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            _resolve(s_allocation="per_depth")

    def test_uniform_map_is_global_s_for_every_depth(self):
        # The byte-identical lock: uniform returns the global S repeated.
        p = _resolve(simulation_steps=32)
        alloc = p.temporal_allocation(depth=9)
        assert alloc.per_depth_steps == tuple([32] * 9)
        assert alloc.is_uniform is True
        assert alloc.global_steps == 32
        assert alloc.derivation_deferred is None

    def test_explicit_map_validates_and_returns_list(self):
        p = _resolve(
            simulation_steps=32,
            s_allocation="explicit",
            s_allocation_explicit=[4, 8, 32],
        )
        alloc = p.temporal_allocation(depth=3)
        assert alloc.per_depth_steps == (4, 8, 32)
        assert alloc.is_uniform is False

    def test_budget_map_is_uniform_with_deferred_marker(self):
        from mimarsinan.tuning.orchestration.temporal_allocation import (
            BUDGET_DERIVATION_DEFERRED,
        )

        p = _resolve(
            simulation_steps=32,
            s_allocation="budget",
            s_allocation_budget={"target": 0.96},
        )
        alloc = p.temporal_allocation(depth=4)
        assert alloc.per_depth_steps == (32, 32, 32, 32)
        assert alloc.is_uniform is True
        assert alloc.derivation_deferred == BUDGET_DERIVATION_DEFERRED


class TestPipelineAccessor:
    def test_of_reads_pipeline_config(self):
        class _Stub:
            config = {"spiking_mode": "ttfs", "weight_quantization": True}

        p = DeploymentPlan.of(_Stub())
        assert p.spiking_mode == "ttfs"
        assert p.weight_quantization is True

    def test_spiking_contract_is_the_sub_part(self):
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        cfg = {"spiking_mode": "lif", "simulation_steps": 32}
        contract = DeploymentPlan.resolve(cfg).spiking_contract()
        assert isinstance(contract, SpikingDeploymentContract)
        assert contract.spiking_mode == "lif"
        assert contract.simulation_steps == 32


class TestCalibrationPipelineAxis:
    """E3: the conversion-health pipeline is a contract-keyed, pipeline-wide axis."""

    def test_lif_plan_gets_inert_pipeline_even_with_flags(self):
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        # A non-cascade cell ignores the ttfs_* step flags → inert (byte-identical).
        plan = _resolve(spiking_mode="lif", ttfs_gain_correction=True)
        assert plan.calibration_pipeline() == CalibrationPipeline.inert()

    def test_cascaded_cycle_plan_opts_in(self):
        plan = _resolve(
            spiking_mode="ttfs_cycle_based",
            ttfs_cycle_schedule="cascaded",
            ttfs_gain_correction=True,
        )
        cal = plan.calibration_pipeline()
        assert cal.gain_cold is True
        assert cal.gain_active is True

    def test_synchronized_cycle_plan_is_inert(self):
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        plan = _resolve(
            spiking_mode="ttfs_cycle_based",
            ttfs_cycle_schedule="synchronized",
            ttfs_gain_correction=True,
            ttfs_theta_cotrain=True,
        )
        assert plan.calibration_pipeline() == CalibrationPipeline.inert()

    def test_distmatch_driver_threads_through(self):
        plan = _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded",
        )
        assert plan.calibration_pipeline(distmatch_driven=True).distmatch is True
        assert plan.calibration_pipeline(distmatch_driven=False).distmatch is False


class TestConversionRecipe:
    """The plan exposes the ConversionPolicy SSOT recipe for its resolved mode."""

    def test_recipe_is_derived_for_the_plans_mode(self):
        plan = _resolve(spiking_mode="lif")
        recipe = plan.conversion_recipe
        assert recipe.driver == "fast"
        assert recipe.special_case == "bn_freeze"
        assert recipe.knobs["lif_blend_fast"] is True
        assert recipe.sim_enables["enable_loihi_simulation"] is True

    def test_cascaded_recipe_is_fast_never_controller(self):
        recipe = _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded"
        ).conversion_recipe
        assert recipe.driver == "fast"
        assert recipe.special_case == "fast_only_never_controller"

    def test_synchronized_recipe_disables_nevresim(self):
        recipe = _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized"
        ).conversion_recipe
        assert recipe.sim_enables["enable_nevresim_simulation"] is False


# ── V1 sole-reader guard (the deployment-decision flags) ───────────────────────

# Deployment-decision flags owned by ``DeploymentPlan.resolve`` — NOT the broad
# identity / sizing keys (model_type, seed, weight_bits, simulation_steps, …)
# that consumers legitimately read directly. ``ttfs_cycle_schedule`` is omitted
# here because it has its OWN src-wide sole-reader guard owned by
# ``SpikingDeploymentContract`` (tests/unit/chip_simulation/test_deployment_contract.py).
_FORBIDDEN_FLAGS = (
    "spiking_mode",
    "activation_quantization",
    "weight_quantization",
    "enable_training_noise",
    "cycle_accurate_lif_forward",
    "pruning",
    "pruning_fraction",
    "weight_source",
    "enable_nevresim_simulation",
    "enable_loihi_simulation",
    "enable_sanafe_simulation",
    "cuda_debug",
    "deployment_metric_full_eval",
)
_FLAGS_ALT = "|".join(_FORBIDDEN_FLAGS)

# A config-dict receiver: an identifier ending in ``config``/``cfg``/``params``
# (so ``config``, ``cfg``, ``pipeline_config``, ``simulation_config``,
# ``self.config``, ``pipeline.config`` all count) or the bare names ``dp``/
# ``flat`` the config-schema layer uses. Anchored so it never matches
# mid-identifier. The bracket arm has a write-guard ``(?!\s*=[^=])`` so an
# assignment TARGET (``dp["weight_quantization"] = …`` in the derivation layer)
# is not a READ offender. The guard has teeth: it matches BOTH ``.get("flag")``
# and ``["flag"]`` read forms regardless of the receiver's local name.
_RECEIVER = r"(?:[\w.]*?(?:config|cfg|params)|\bdp|\bflat)"
_DEPLOYMENT_FLAG_READ = re.compile(
    _RECEIVER
    + r"""\s*(?:\.get\(\s*['"](?P<g>%s)['"]"""
    r"""|\[\s*['"](?P<b>%s)['"]\s*\](?!\s*=[^=]))""" % (_FLAGS_ALT, _FLAGS_ALT)
)


def _src_root() -> Path:
    return Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


def _flag_read_offenders(root: Path) -> list[str]:
    offenders = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        for m in _DEPLOYMENT_FLAG_READ.finditer(text):
            flag = m.group("g") or m.group("b")
            offenders.append(f"{path.relative_to(root).as_posix()}: {flag}")
    return offenders


class TestPipelineStepsReadThePlan:
    """Tightest scope: pipeline_steps/** has ZERO allowlist.

    Pipeline steps must read deployment-decision flags from the resolved
    ``DeploymentPlan`` (``DeploymentPlan.of(self.pipeline).<field>``), never a
    raw ``config.get(<flag>)``. No carve-out is permitted inside this directory.
    """

    def test_pipeline_steps_have_no_raw_deployment_flag_reads(self):
        steps_root = _src_root() / "pipelining" / "pipeline_steps"
        offenders = _flag_read_offenders(steps_root)
        assert offenders == [], (
            "pipeline steps must read deployment flags from "
            "DeploymentPlan.of(self.pipeline), not raw config.get(...); "
            f"offenders: {offenders}"
        )


class TestNoStrayDeploymentFlagReadsAnywhere:
    """V1 sole-reader guard BROADENED to all of ``src/mimarsinan``.

    Deployment-DECISION flags are resolved once in ``DeploymentPlan`` and read
    as resolved fields; a raw ``config.get(<flag>)`` / ``config[<flag>]`` of a
    decision flag elsewhere bypasses the SSOT. The ALLOWLIST below is the tight,
    documented set of files that legitimately read the raw key because they are
    the RESOLVER / SSOT / derivation / presentation layers (not decision
    consumers), plus one documented byte-identity carve-out.
    """

    # Each entry is a relative path (file or ``dir/`` prefix) + WHY it is exempt.
    ALLOWLIST = (
        # ── the resolver itself + the spiking-semantics sub-contract SSOT ──
        # ``DeploymentPlan`` is the resolver; ``SpikingDeploymentContract`` /
        # ``NeuralBehaviorConfig`` / ``FiringStrategyFactory`` are the firing-
        # semantics SSOT layer (the C++-comparator factory + reset/compare
        # policy) that resolve their own sub-contract from the raw config.
        "pipelining/core/deployment_plan.py",
        "chip_simulation/deployment_contract.py",
        "chip_simulation/behavior_config.py",
        "chip_simulation/firing_strategy.py",
        # ``spiking_semantics`` is the mode-predicate SSOT (CLAUDE.md):
        # ``lif_per_hop_retiming_enabled`` is the ONE predicate both twins
        # (chip-aligned NF and the mapping) consult, so it resolves its own
        # (knob, spiking_mode) sub-contract from the raw config.
        "chip_simulation/spiking_semantics.py",
        # ── codegen: the C++ comparator/exec policy (FiringStrategyFactory SSOT
        #    side); reads its own ``simulation_config`` codegen param ──
        "code_generation/generate_main.py",
        # ── config_schema: DEFINES / DERIVES / DISPLAYS / VALIDATES the keys ──
        "config_schema/",
        # ── coverage/ledger projection: this is the AUDIT bridge that maps raw
        #    configs/rows and resolved plans into explicit hypervolume axes. It
        #    is a writer/normalizer, not a deployment-decision consumer.
        "chip_simulation/hypervolume_axis_encoder.py",
        # ── tuners select their own TRAINING-FORWARD family from spiking_mode /
        #    cycle_accurate_lif_forward (a V2 SpikingModePolicy concern, not a
        #    V1 decision-flag consumer). The read sits on the per-perceptron
        #    ``pipeline_config`` dict / a tuner's own knob (the LIF reads were
        #    collapsed into the ``LifAdaptationPlan`` resolver layer;
        #    ttfs_cycle's read defaults to "ttfs_cycle_based", not the plan's
        #    "lif"); threading a DeploymentPlan through these signatures removes
        #    no scattered branch and would not be byte-identical — pure churn.
        "tuning/orchestration/adaptation_manager.py",
        # ``lif_exact_qat`` was split out of ``adaptation_manager`` (module
        # budget); its predicate reads the same per-perceptron pipeline_config
        # keys (spiking_mode / cycle_accurate_lif_forward) under the same
        # training-forward-family rationale.
        "tuning/orchestration/lif_exact_qat.py",
        "tuning/tuners/ttfs_cycle_adaptation_tuner.py",
        # ── byte-identity carve-out (NOT a "this isn't a decision flag" claim) ──
        # ``simulation_runner/core.py`` reads ``weight_quantization`` with a
        # default of ``True`` (the legacy "quantized unless told otherwise"
        # runner default); ``DeploymentPlan.weight_quantization`` resolves it
        # with a default of ``False``. The key is genuinely absent for a
        # ``vanilla`` config (the ``phased`` preset is what materialises it), so
        # routing this read through the plan would FLIP the omitted-key value and
        # break byte-identity. Left verbatim + noted (see judgment in the V1
        # doc); ``spiking_mode`` in the same file WAS migrated to the plan.
        "chip_simulation/simulation_runner/core.py",
    )

    def test_only_the_allowlisted_layers_read_raw_decision_flags(self):
        root = _src_root()
        offenders = []
        for line in _flag_read_offenders(root):
            rel = line.split(":", 1)[0]
            if any(rel == a or rel.startswith(a) for a in self.ALLOWLIST):
                continue
            offenders.append(line)
        assert offenders == [], (
            "deployment-decision flags must be read from DeploymentPlan, not "
            "raw config.get(...)/config[...]; either route through the plan or "
            "add a documented entry to the ALLOWLIST; offenders: " + str(offenders)
        )

    # The resolver + the spiking sub-contract are structural anchors: the
    # resolver reads via a ``get = config.get`` alias and the contract reads only
    # the separately-guarded ``ttfs_cycle_schedule`` key, so the receiver-anchored
    # guard pattern does not match a forbidden-flag read inside them even though
    # they are the canonical exempt layers. They are checked for existence only.
    _STRUCTURAL_ANCHORS = (
        "pipelining/core/deployment_plan.py",
        "chip_simulation/deployment_contract.py",
    )

    def test_allowlist_has_no_dead_entries(self):
        """Keep the allowlist honest: no path may rot into a silent exemption.

        Every entry must exist; every NON-anchor entry must additionally still
        read a raw forbidden flag (so a carve-out whose read was migrated away,
        or a deleted file, fails instead of lingering as a blanket exemption).
        """
        root = _src_root()
        reading_files = {
            line.split(":", 1)[0] for line in _flag_read_offenders(root)
        }
        dead = []
        for entry in self.ALLOWLIST:
            is_dir = entry.endswith("/")
            if is_dir and not (root / entry).is_dir():
                dead.append(f"{entry} (missing dir)")
                continue
            if not is_dir and not (root / entry).exists():
                dead.append(f"{entry} (missing file)")
                continue
            if entry in self._STRUCTURAL_ANCHORS:
                continue
            reads = (
                any(f.startswith(entry) for f in reading_files)
                if is_dir
                else entry in reading_files
            )
            if not reads:
                dead.append(f"{entry} (no raw flag read — migrated away?)")
        assert dead == [], f"stale ALLOWLIST entries: {dead}"
