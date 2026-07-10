"""Workload-literal ratchet: framework-side dataset/vehicle strings and the
tracked inventory of tier-0-calibrated constants only ever SHRINK.

Two gates (the purity-audit enforcement, findings doc
purity_audit_workload_constants.md):

1. STRING SCAN — no dataset id (mnist/cifar/imagenet/svhn/kmnist/fashion/ecg)
   or registered model-type id may appear as a non-docstring string literal in
   framework code. Lawful homes are exempt (data providers, model builders);
   the chip_simulation research instruments are QUARANTINED (exempt but
   listed — campaign-definition data, not pipeline behavior); everything else
   rides a shrink-only allowlist.

2. NUMERIC INVENTORY — every kept workload-calibrated constant (the audit's
   (e*) class) is pinned with its value and validity domain. Moving or
   retuning one fails here, forcing a conscious, documented update; migrating
   one to a registry profile DELETES its row (the ratchet shrinks).
"""

import ast
import re
from operator import attrgetter
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

_DATASET_TOKEN_RE = re.compile(
    r"mnist|cifar|imagenet|svhn|kmnist|fashion|ecg", re.IGNORECASE
)

# Lawful workload homes: registrations DECLARE dataset/model facts here.
_EXEMPT_PREFIXES = (
    "data_handling/data_providers/",
    "models/builders/",
)

# Research-instrument quarantine (audit rows D8, D9, V3-V6): campaign
# definitions and ledger analytics over PAST runs — workload enumerations are
# their subject matter, not framework assumptions. Exempt, but named.
INSTRUMENT_QUARANTINE = frozenset({
    "chip_simulation/pareto.py",
    "chip_simulation/hypervolume_axes.py",
    "chip_simulation/hypervolume_axis_encoder.py",
    "chip_simulation/hypervolume_cells.py",
    "chip_simulation/coverage_ledger.py",
    "chip_simulation/weight_reuse_cost_model.py",
})

# file -> (max hit count, reason). SHRINK ONLY: lower a count when you remove
# a literal; never raise one or add a row without an audit-level rationale.
STRING_ALLOWLIST = {
    "common/env.py": (1, "IMAGENET_ROOT env accessor — the sanctioned env-var seam the ImageNet provider consumes"),
    "config_schema/registry/build.py": (1, "wizard document seed default (data_provider_name) — a deployment-config seed, not pipeline behavior"),
    "config_schema/registry/entries_model.py": (1, "preprocessing key doc text naming the provider-registered preset ids"),
    "tuning/orchestration/conversion_policy.py": (1, "recipe rationale citing the MNIST evidence run ids (provenance text)"),
}


def _docstring_ids(tree: ast.AST) -> set:
    docs = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docs.add(id(body[0].value))
    return docs


def _registered_model_ids() -> frozenset:
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

    return frozenset(ModelRegistry.builder_classes())


def _workload_string_hits() -> dict:
    model_ids = _registered_model_ids()
    hits: dict[str, list] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        if any(rel.startswith(p) for p in _EXEMPT_PREFIXES):
            continue
        if rel in INSTRUMENT_QUARANTINE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docs
            ):
                value = node.value
                if _DATASET_TOKEN_RE.search(value) or value.lower() in model_ids:
                    hits.setdefault(rel, []).append((node.lineno, value[:60]))
    return hits


class TestWorkloadStringRatchet:
    def test_quarantined_instruments_exist(self):
        missing = [rel for rel in INSTRUMENT_QUARANTINE if not (SRC_ROOT / rel).exists()]
        assert not missing, f"quarantine list is stale: {missing}"

    def test_no_new_framework_side_workload_strings(self):
        hits = _workload_string_hits()
        unlisted = {rel: v for rel, v in hits.items() if rel not in STRING_ALLOWLIST}
        assert not unlisted, (
            "workload strings (dataset ids / registered model ids) appeared in "
            f"framework code: {unlisted}. Inject them via the registry "
            "contracts (DataProvider / builder profiles) instead."
        )
        over = {
            rel: (len(v), STRING_ALLOWLIST[rel][0], v)
            for rel, v in hits.items()
            if len(v) > STRING_ALLOWLIST[rel][0]
        }
        assert not over, f"allowlisted files grew their workload-string count: {over}"

    def test_allowlist_only_shrinks(self):
        hits = _workload_string_hits()
        stale = [rel for rel in STRING_ALLOWLIST if rel not in hits]
        assert not stale, (
            f"allowlist rows with zero remaining hits — delete them (shrink!): {stale}"
        )


# The audit's (e*) inventory: kept constants whose VALUES were calibrated on
# the tier-0/0.1 MNIST corpus. Every row: (module, attribute path, value,
# audit row + validity domain). Deleting a row = the constant was migrated to
# a registry profile or re-expressed in invariant units. NEVER add a row
# without an audit-level rationale; never change a value silently.
CALIBRATED_CONSTANT_INVENTORY = (
    # -- TuningBudget clamps (B1/B2 generic defaults; profile-overridable) --
    ("mimarsinan.tuning.orchestration.tuning_budget", "_GENERIC_STEP_CAP", 4000,
     "B1: absolute per-tuner step cap (MNIST-epoch scale); override: tuning_step_cap_epochs"),
    ("mimarsinan.tuning.orchestration.tuning_budget", "_GENERIC_EVAL_SUBSAMPLE_TARGET", 5000,
     "B2: tuner eval-subset target; override: eval_subsample_target"),
    # -- TuningPolicy loop constants (T1/T2/T4/T5; tier-0 probe-validated) --
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.prefix_stage_lr", 1e-3,
     "T1: P4 stage LR ceiling (arm-B, x3b wave); override: ModelWorkloadProfile.prefix_stage_lr"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.endpoint_floor_lr", 2e-3,
     "T2: floor-chasing endpoint LR (t0_06 probe); override: ModelWorkloadProfile.endpoint_floor_lr"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.endpoint_floor_steps", 16000,
     "5u: run-total endpoint step budget (t01_23); per-cell key endpoint_floor_steps"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.endpoint_floor_min_cover_steps", 2000,
     "C1: absolute lr-dip cover before any armed-endpoint stop (t0_21 dip ~1.6k; Phase-1 probe recalibrates)"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.endpoint_floor_patience_fraction", 0.25,
     "C1: keep-best cover/patience fraction of the funded budget (budget-invariant units)"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.dfq_keepbest_patience", 5,
     "T4: DFQ keep-best patience (W-CAL-3), iteration units"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.prefix_stage_dfq_iters", 4,
     "T4: P4 stage re-affine iters (T4 arm B)"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.prefix_stage_keepbest_interval", 25,
     "T4: P4 keep-best cadence (arm B)"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.hop_stage_steps_per_rate", 40,
     "FAST respec: hop-frontier rung budget, measured budget-insensitive on tier-0"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.k_commit", 2.0,
     "T5: commit gate in SE multiples (workload-invariant units)"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "TUNING_POLICY.rollback_cumulative_bound", 0.05,
     "T5: cumulative rollback bound (absolute accuracy; tier-0 calibrated)"),
    ("mimarsinan.tuning.orchestration.tuning_policy", "FAST_LADDER_STEPS_PER_RATE", 120,
     "C5: the fast ladder's per-rung budget (was five scattered 120s); *_fast_steps_per_rate overrides"),
    # -- Tuner-base loop constants (T6) --
    ("mimarsinan.tuning.orchestration.tuner_base", "CATASTROPHIC_DROP_FACTOR", 0.8,
     "T6: catastrophic-drop detector, fraction of entry metric"),
    ("mimarsinan.tuning.orchestration.tuner_base", "_RECOVERY_PATIENCE", 5,
     "T6: recovery patience, check intervals"),
    ("mimarsinan.tuning.orchestration.tuner_base", "_STUCK_STREAK_REQUIRED", 3,
     "T6: stuck-streak declaration, check intervals"),
    # -- MBH gate / anneal (T7/T11) --
    ("mimarsinan.tuning.orchestration.mbh_gate", "ACCEPT_TOLERANCE", 0.01,
     "T7: gate slack, ABSOLUTE accuracy (k*SE re-expression is a filed follow-up)"),
    ("mimarsinan.tuning.orchestration.mbh_gate", "MAX_REFINEMENTS", 3,
     "T7: bisection refinements"),
    ("mimarsinan.tuning.orchestration.mbh_tanneal", "DEFAULT_START_T", 32,
     "T11: anneal start T; aliases the platform S=32 default (derive-from-target_tq follow-up)"),
    ("mimarsinan.tuning.orchestration.mbh_tanneal", "HIGH_TARGET_START_FACTOR", 4,
     "T11: anneal start multiple for high targets"),
    # -- Install-resolution gauges (G7/G8; tier-0/0.1 corpus laws) --
    ("mimarsinan.tuning.orchestration.install_resolution.gauges", "MIN_MEDIAN_EFFECTIVE_LEVELS", 2.0,
     "G7/A6: value-starvation bar, grid-level units (t0/t01 corpus)"),
    ("mimarsinan.tuning.orchestration.install_resolution.gauges", "NEAREST_MIN_MEDIAN_EFFECTIVE_LEVELS", 1.0,
     "G7/A6: nearest-kernel starvation bar"),
    ("mimarsinan.tuning.orchestration.install_resolution.gauges", "STARVED_MASS_WARN", 0.5,
     "G7/A6: starved-mass warn fraction"),
    ("mimarsinan.tuning.orchestration.install_resolution.gauges", "TEMPORAL_RECOVERY_HEADROOM", 2.0,
     "G7/A6(ii): delay/window recovery headroom (t01 ratios 1.3-1.7 recovered, >=3.3 failed)"),
    ("mimarsinan.tuning.orchestration.install_resolution.gauges", "PROVEN_RECOVERY_DEPTH", 6,
     "G8: chain-depth law (t0_22/t0_18/t0_03 vs t01_12); override: proven_recovery_depth"),
    # -- Hop staging (T12) --
    ("mimarsinan.tuning.orchestration.frontier.hop_staging", "HOP_STAGE_MIN_LEVELS", 6,
     "T12: stage only past the measured full-recovery depth; proven_recovery_depth overrides"),
    ("mimarsinan.tuning.orchestration.frontier.hop_staging", "_REAFFINE_ETA", 0.7,
     "T12: frontier DFQ step size (the distmatch default)"),
    # -- Calibration-set generic defaults (S3/S4; profile-overridable) --
    ("mimarsinan.tuning.orchestration.calibration_pipeline", "_GENERIC_DISTMATCH_BIAS_ITERS", 15,
     "S4: TTFS distmatch DFQ iters; override: calibration_set_policy.distmatch_bias_iters"),
    ("mimarsinan.tuning.orchestration.calibration_pipeline", "_GENERIC_DISTMATCH_CAL_BATCHES", 8,
     "S4: TTFS distmatch calibration batches; override: calibration_set_policy.distmatch_cal_batches"),
    ("mimarsinan.tuning.orchestration.lif_adaptation_plan", "_GENERIC_DISTMATCH_BIAS_ITERS", 10,
     "S4: LIF distmatch DFQ iters; same profile override"),
    ("mimarsinan.tuning.orchestration.lif_adaptation_plan", "_GENERIC_DISTMATCH_CAL_BATCHES", 8,
     "S4: LIF distmatch calibration batches; same profile override"),
    ("mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step", "MAX_ANALYSIS_BATCHES", 32,
     "S3: activation-analysis batch ceiling; override: calibration_set_policy.analysis_batches_max"),
    ("mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step", "ANALYSIS_BATCH_SIZE_CAP", 16,
     "S3: VRAM guard on the analysis batch (saved-tensor decorators pin outputs); override: analysis_batch_size_cap / activation_analysis_batch_size"),
    ("mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step", "MAX_SAMPLES_PER_BATCH", 8192,
     "S3: per-batch activation subsample size (aggregation cost bound)"),
    # -- Perf heuristics (S1) --
    ("mimarsinan.pipelining.core.simulation_factory", "_SIM_EVAL_BATCH_SIZE", 1024,
     "S1: host-GPU amortization batch (bit-equal per-sample; OOM retry honors simulation_batch_size); VRAM-probe derivation is a filed follow-up"),
    # -- Packer estimator (G5) --
    ("mimarsinan.mapping.verification.capacity.estimate", "PACKER_DIVERGENCE_MARGIN", 0.45,
     "G5: packer-estimator optimism band, measured on the current packer"),
    # -- ConversionPolicy recipe absolutes (C1-C4, kept VERBATIM as recipe data;
    #    invariant-unit re-expression is a filed follow-up, never bundled) --
    ("mimarsinan.tuning.orchestration.conversion_policy", "_WQ_RECIPE_KNOBS", {
        "wq_fast_rates": [0.5, 1.0], "wq_fast_steps_per_rate": 0,
        "wq_endpoint_recovery_steps": 600,
    }, "C1/C6: WQ demotion recipe (5g-v); step values are MNIST-scale budgets"),
    ("mimarsinan.tuning.orchestration.conversion_policy", "_BIT_PARITY_LOSSLESS_RECIPE_KNOBS", {
        "endpoint_target_floor": 0.98, "wq_endpoint_recovery_steps": 16000,
    }, "C4/5u: the tier-0 acceptance bar as the endpoint floor + probe-validated budget"),
    ("mimarsinan.tuning.orchestration.conversion_policy", "_WELL_CONDITIONED_ENDPOINT_FLOOR_KNOBS", {
        "wq_endpoint_target_floor": 0.98, "wq_endpoint_recovery_steps": 16000,
    }, "C4/5u generalized: final-WQ endpoint floor for near-lossless conversions "
       "incl. ttfs_quantized (sub-SE proxy→deployed transfer, 2026-07 grant)"),
)


class TestCalibratedConstantInventory:
    def test_inventory_count_is_pinned(self):
        # Shrink-only: lower this when a row is migrated/deleted; growing the
        # inventory needs an audit-level rationale in the same commit.
        # 38 -> 40 (C1): the two new TuningPolicy convergence-stop constants
        # are tier-0-calibrated and must be pinned like their siblings.
        assert len(CALIBRATED_CONSTANT_INVENTORY) == 40

    @pytest.mark.parametrize(
        "module,attr,expected",
        [(m, a, v) for m, a, v, _ in CALIBRATED_CONSTANT_INVENTORY],
        ids=[f"{m.split('.')[-1]}.{a}" for m, a, _, _ in CALIBRATED_CONSTANT_INVENTORY],
    )
    def test_constant_holds_its_documented_value(self, module, attr, expected):
        import importlib

        mod = importlib.import_module(module)
        actual = attrgetter(attr)(mod)
        assert actual == expected, (
            f"{module}.{attr} moved from its documented tier-0-calibrated value "
            f"({expected} -> {actual}); update the inventory row consciously "
            "(validity domain!) or migrate the constant to a registry profile."
        )
