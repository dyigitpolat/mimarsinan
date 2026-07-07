"""The tiered test_configs are schema-valid, legal, and cover the claimed pairs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TEST_CONFIGS = ROOT / "test_configs"

# "0_1" is the tier-0.1 diagnostic matrix (minimal pairs of tier-0 anchors).
TIERS = (0, "0_1", 1, 2)

TOP_LEVEL_KEYS = {
    "seed", "pipeline_mode", "experiment_name", "generated_files_path",
    "data_provider_name", "platform_constraints", "deployment_parameters",
    "target_metric_override", "start_step", "stop_step",
}
QUANT_REQUIRED_MODES = {"ttfs_quantized", "ttfs_cycle_based"}
SIM_ENABLE_KEYS = {
    "enable_nevresim_simulation", "enable_sanafe_simulation", "enable_loihi_simulation",
}


def _tier_configs(tier):
    tier_dir = TEST_CONFIGS / f"tier{tier}"
    return sorted(p for p in tier_dir.glob("t*.json"))


def _manifest(tier):
    return json.loads((TEST_CONFIGS / f"tier{tier}" / "manifest.json").read_text())


def _notes(tier):
    return {r["config"]: r.get("note", "") for r in _manifest(tier)["runs"]}


class TestGeneratorIsTheSSOT:
    def test_generator_reproduces_committed_files(self, tmp_path):
        """Regenerating must be a no-op: the JSONs never drift from generate.py."""
        before = {
            p.relative_to(TEST_CONFIGS): p.read_text()
            for p in TEST_CONFIGS.rglob("*.json")
        }
        subprocess.run(
            [sys.executable, str(TEST_CONFIGS / "generate.py")],
            check=True, capture_output=True,
        )
        after = {
            p.relative_to(TEST_CONFIGS): p.read_text()
            for p in TEST_CONFIGS.rglob("*.json")
        }
        assert before == after


class TestTierShapes:
    def test_run_counts(self):
        assert len(_tier_configs(0)) == 25
        assert len(_tier_configs("0_1")) == 25
        assert len(_tier_configs(1)) == 8
        assert len(_tier_configs(2)) == 3

    def test_manifest_matches_files(self):
        for tier in TIERS:
            manifest = _manifest(tier)
            names = {r["config"] for r in manifest["runs"]}
            files = {p.name for p in _tier_configs(tier)}
            assert names == files


class TestConfigValidity:
    @pytest.mark.parametrize("tier", TIERS)
    def test_keys_are_known(self, tier):
        from mimarsinan.config_schema.defaults import (
            CONFIG_KEYS_SET,
            DEFAULT_PLATFORM_CONSTRAINTS,
        )

        # max_axons/max_neurons/has_bias are user-facing platform keys
        # (see get_user_default_platform_constraints) without schema defaults.
        platform_keys = set(DEFAULT_PLATFORM_CONSTRAINTS) | {
            "cores", "max_axons", "max_neurons", "has_bias",
        }
        for path in _tier_configs(tier):
            cfg = json.loads(path.read_text())
            assert set(cfg) == TOP_LEVEL_KEYS, path.name
            unknown = set(cfg["deployment_parameters"]) - set(CONFIG_KEYS_SET)
            assert not unknown, f"{path.name}: unknown keys {unknown}"
            unknown_pc = set(cfg["platform_constraints"]) - platform_keys
            assert not unknown_pc, f"{path.name}: unknown platform keys {unknown_pc}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_sim_enables_left_to_derivation(self, tier):
        """Sim backends are ConversionPolicy-derived; configs must not pin them."""
        for path in _tier_configs(tier):
            cfg = json.loads(path.read_text())
            assert not (set(cfg["deployment_parameters"]) & SIM_ENABLE_KEYS), path.name

    @pytest.mark.parametrize("tier", TIERS)
    def test_legality_rules(self, tier):
        notes = _notes(tier)
        for path in _tier_configs(tier):
            dp = json.loads(path.read_text())["deployment_parameters"]
            if dp["spiking_mode"] in QUANT_REQUIRED_MODES:
                assert dp["weight_quantization"] is True, path.name
            # Sim-role respec (user-directed 2026-07-07): simulators are
            # PARITY probes — N=25 decision-parity sample everywhere; the
            # accuracy verdict is the SCM identity read (full test set).
            assert dp["max_simulation_samples"] == 25, path.name

    @pytest.mark.parametrize("tier", TIERS)
    def test_activation_quantization_left_to_derivation(self, tier):
        """AQ is a derived pipeline-assembly mode; configs must never pin it."""
        for path in _tier_configs(tier):
            dp = json.loads(path.read_text())["deployment_parameters"]
            assert "activation_quantization" not in dp, path.name

    @pytest.mark.parametrize("tier", TIERS)
    def test_quant_tags_are_runtime_truth(self, tier):
        """Names carry only wq/fp; no fictional aq/wqaq tags anywhere."""
        for path in _tier_configs(tier):
            assert "_aq_" not in path.name, path.name
            assert "_wqaq_" not in path.name, path.name
            cfg = json.loads(path.read_text())
            wq = cfg["deployment_parameters"]["weight_quantization"]
            expected_tag = "_wq_" if wq else "_fp_"
            assert expected_tag in path.name, path.name
            expected_mode = "phased" if wq else "vanilla"
            assert cfg["pipeline_mode"] == expected_mode, path.name

    @pytest.mark.parametrize("tier", TIERS)
    def test_configs_pass_the_assembly_contract(self, tier):
        from mimarsinan.config_schema.deployment_derivation import (
            enforce_quantization_assembly_contract,
        )

        for path in _tier_configs(tier):
            cfg = json.loads(path.read_text())
            enforce_quantization_assembly_contract(
                cfg["deployment_parameters"],
                cfg["platform_constraints"],
                pipeline_mode=cfg.get("pipeline_mode"),
            )

    @pytest.mark.parametrize("tier", TIERS)
    def test_configs_resolve_through_derivation(self, tier):
        """Every config must survive the real config derivation pipeline."""
        from mimarsinan.config_schema.defaults import (
            get_default_deployment_parameters,
            get_default_platform_constraints,
        )
        from mimarsinan.config_schema.deployment_derivation import (
            derive_deployment_parameters,
        )

        for path in _tier_configs(tier):
            cfg = json.loads(path.read_text())
            merged = get_default_deployment_parameters()
            merged.update(cfg["deployment_parameters"])
            merged.update(get_default_platform_constraints())
            merged.update(cfg["platform_constraints"])
            derive_deployment_parameters(merged)
            assert merged["spiking_mode"] == cfg["deployment_parameters"]["spiking_mode"]


class TestTier0PairwiseCoverage:
    def _cells(self):
        return [r["cell"] for r in _manifest(0)["runs"]]

    def test_firing_by_vehicle_full_grid(self):
        runs = _manifest(0)["runs"]
        pairs = {(r["cell"]["firing"], r["cell"]["sync"], r["model_type"]) for r in runs}
        assert len(pairs) == 25

    def test_firing_by_s_pairs(self):
        cells = self._cells()
        for firing, sync in [("lif", "none"), ("ttfs", "none"), ("ttfs_quantized", "none"),
                             ("ttfs_cycle_based", "cascaded"), ("ttfs_cycle_based", "synchronized")]:
            seen = {c["S"] for c in cells if (c["firing"], c["sync"]) == (firing, sync)}
            assert seen == {"4", "8", "16", "32"}, (firing, sync, seen)

    def test_every_mode_has_a_pruned_cell(self):
        cells = self._cells()
        for firing, sync in [("lif", "none"), ("ttfs", "none"), ("ttfs_quantized", "none"),
                             ("ttfs_cycle_based", "cascaded"), ("ttfs_cycle_based", "synchronized")]:
            assert any(
                c["pruning"] == "pruned"
                for c in cells if (c["firing"], c["sync"]) == (firing, sync)
            ), (firing, sync)

    def test_screened_axes_each_value_appears(self):
        cells = self._cells()
        assert {c["encoding_placement"] for c in cells} == {"subsume", "offload"}

    def test_cell_vocabulary_matches_coverage_ledger(self):
        from mimarsinan.chip_simulation.coverage_ledger import AXES

        domains = {a.name: set(a.values) for a in AXES}
        for cell in self._cells():
            for axis in ("firing", "sync", "quantization", "S", "vehicle", "dataset",
                         "regime", "pruning", "encoding_placement"):
                assert cell[axis] == "any" or cell[axis] in domains[axis], (axis, cell[axis])


class TestTier1Coverage:
    def test_all_five_modes_and_both_regimes(self):
        cells = [r["cell"] for r in _manifest(1)["runs"]]
        modes = {(c["firing"], c["sync"]) for c in cells}
        assert len(modes) == 5
        assert {c["regime"] for c in cells} == {"from_scratch", "pretrained"}


class TestW2Respecs:
    """W2 verdicts: t0_03 needs scheduling (packer-verified both ways) and
    plain deep_mlp d16 is recipe-unreachable (residual is the trainable
    backbone). USER DECISION 2026-07-06: residual respec, depth kept at 16."""

    def test_t0_03_is_scheduled(self):
        path = TEST_CONFIGS / "tier0" / "t0_03_lif_deepcnn_d8_wq_s16_sched.json"
        cfg = json.loads(path.read_text())
        assert cfg["deployment_parameters"]["allow_scheduling"] is True

    def test_t0_19_is_residual_d16(self):
        path = TEST_CONFIGS / "tier0" / "t0_19_casc_deepmlp_d16_wq_s16_residual.json"
        cfg = json.loads(path.read_text())
        model_config = cfg["deployment_parameters"]["model_config"]
        assert model_config["residual"] is True
        assert model_config["depth"] == 16

    def test_old_unscheduled_and_plain_specs_are_gone(self):
        tier0 = TEST_CONFIGS / "tier0"
        assert not (tier0 / "t0_03_lif_deepcnn_d8_wqaq_s16.json").exists()
        assert not (tier0 / "t0_19_casc_deepmlp_d16_wq_s16.json").exists()

    def test_manifest_tags_carry_the_respec(self):
        runs = {r["name"]: r for r in _manifest(0)["runs"]}
        assert "sched" in runs["t0_03_lif_deepcnn_d8_wq_s16_sched"]["tags"]
        assert "residual" in runs["t0_19_casc_deepmlp_d16_wq_s16_residual"]["tags"]


class TestW3cRespecs:
    """W3c matrix-truth respec (2026-07-06): the fictional aq axis is gone;
    t0_04/t0_07 became real WQ deployments; t0_15/t0_21 pruning 0.5 -> 0.10
    (user-directed); t0_02/t0_09/t0_18 keep the 0.5 heavy-pruning stressors."""

    def test_t0_04_and_t0_07_are_real_wq_deployments(self):
        for name in ("t0_04_lif_deepmlp_d8_wq_s32", "t0_07_ttfs_lenet5_wq_s16"):
            cfg = json.loads((TEST_CONFIGS / "tier0" / f"{name}.json").read_text())
            assert cfg["deployment_parameters"]["weight_quantization"] is True, name
            assert cfg["pipeline_mode"] == "phased", name

    def test_t0_15_and_t0_21_pruned_at_10_percent(self):
        for name in ("t0_15_ttfsq_simplemlp_wq_s8_pruned10",
                     "t0_21_sync_mmixcore_wq_s8_pruned10"):
            cfg = json.loads((TEST_CONFIGS / "tier0" / f"{name}.json").read_text())
            assert cfg["deployment_parameters"]["pruning_fraction"] == 0.10, name

    def test_heavy_pruning_stressors_kept_at_50_percent(self):
        runs = {r["name"]: r for r in _manifest(0)["runs"]}
        heavy = [n for n in runs if n.startswith(("t0_02_", "t0_09_", "t0_18_"))]
        assert len(heavy) == 3
        for name in heavy:
            cfg = json.loads((TEST_CONFIGS / "tier0" / f"{name}.json").read_text())
            assert cfg["deployment_parameters"]["pruning_fraction"] == 0.5, name

    def test_old_fictional_forms_are_gone(self):
        tier0 = TEST_CONFIGS / "tier0"
        assert not (tier0 / "t0_04_lif_deepmlp_d8_aq_s32.json").exists()
        assert not (tier0 / "t0_07_ttfs_lenet5_aq_s16.json").exists()
        assert not (tier0 / "t0_15_ttfsq_simplemlp_wqaq_s8_pruned.json").exists()
        assert not (tier0 / "t0_21_sync_mmixcore_wq_s8_pruned.json").exists()

    def test_manifest_notes_carry_the_respecs(self):
        manifest = _manifest(0)
        assert manifest.get("coverage_notes"), "tier0 manifest must carry coverage notes"
        runs = {r["name"]: r for r in manifest["runs"]}
        for name in ("t0_04_lif_deepmlp_d8_wq_s32", "t0_07_ttfs_lenet5_wq_s16",
                     "t0_15_ttfsq_simplemlp_wq_s8_pruned10",
                     "t0_21_sync_mmixcore_wq_s8_pruned10"):
            assert "respec" in runs[name].get("note", ""), name

    def test_manifest_quant_axis_is_runtime_truth(self):
        aq_on = {"lif", "ttfs_quantized", "ttfs_cycle_based"}
        for run in _manifest(0)["runs"]:
            cell = run["cell"]
            cfg = json.loads((TEST_CONFIGS / "tier0" / run["config"]).read_text())
            wq = cfg["deployment_parameters"]["weight_quantization"]
            if not wq:
                assert cell["quantization"] == "none", run["name"]
            elif cell["firing"] in aq_on:
                assert cell["quantization"] == "wq_aq", run["name"]
            else:
                assert cell["quantization"] == "wq", run["name"]


def _flatten(node, prefix=""):
    """Flatten a JSON tree into {dotted/indexed key-path: leaf value}."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(node, list):
        out = {}
        for i, value in enumerate(node):
            out.update(_flatten(value, f"{prefix}[{i}]"))
        return out
    return {prefix: node}


def _config_delta(a: dict, b: dict) -> set:
    """Key-paths whose leaf values differ between two configs (missing != equal)."""
    fa, fb = _flatten(a), _flatten(b)
    missing = object()
    return {
        path for path in set(fa) | set(fb)
        if fa.get(path, missing) != fb.get(path, missing)
    }


_S_MOVE = {"platform_constraints.target_tq", "platform_constraints.simulation_steps"}
_E4_MOVE = {"deployment_parameters.training_epochs"}
_WB_MOVE = {"platform_constraints.weight_bits"}
_FLOOR_MOVE = {"deployment_parameters.endpoint_floor_wall_s"}

# The tier-0.1 design table: cell -> (tier-0 anchor, exact config key-paths moved).
# Every cell is a minimal pair; experiment_name is excluded from the delta.
TIER01_EXPECTED_DELTAS = {
    # A - install-resolution law calibration (A6, section 5v)
    "t01_01_lif_mmixcore_wq_s8": ("t0_01_lif_mmixcore_wq_s4", _S_MOVE),
    "t01_02_lif_mmixcore_wq_s16": ("t0_01_lif_mmixcore_wq_s4", _S_MOVE),
    "t01_03_casc_mmixcore_wq_s4_offload_sched_nobias":
        ("t0_16_casc_mmixcore_wq_s8_offload_sched_nobias", _S_MOVE),
    "t01_04_sync_mmixcore_wq_s16_pruned10": ("t0_21_sync_mmixcore_wq_s8_pruned10", _S_MOVE),
    "t01_05_sync_mmixcore_wq_s4_pruned10": ("t0_21_sync_mmixcore_wq_s8_pruned10", _S_MOVE),
    "t01_06_ttfsq_mmixcore_wq_s8_offload": ("t0_11_ttfsq_mmixcore_wq_s16_offload", _S_MOVE),
    # B - pretrain envelope (training_epochs 2 -> 4)
    "t01_07_ttfs_mmixcore_wq_s8_e4": ("t0_06_ttfs_mmixcore_wq_s8", _E4_MOVE),
    "t01_08_lif_mmixcore_wq_s4_e4": ("t0_01_lif_mmixcore_wq_s4", _E4_MOVE),
    "t01_09_sync_mmixcore_wq_s8_pruned10_e4": ("t0_21_sync_mmixcore_wq_s8_pruned10", _E4_MOVE),
    "t01_10_casc_mmixcore_wq_s8_offload_sched_nobias_e4":
        ("t0_16_casc_mmixcore_wq_s8_offload_sched_nobias", _E4_MOVE),
    "t01_11_casc_deepmlp_d16_wq_s16_residual_e4":
        ("t0_19_casc_deepmlp_d16_wq_s16_residual", _E4_MOVE),
    # C - cascade structure isolation
    "t01_12_casc_mmixcore_wq_s8": (
        "t0_16_casc_mmixcore_wq_s8_offload_sched_nobias",
        {"deployment_parameters.encoding_layer_placement",
         "deployment_parameters.allow_scheduling",
         "platform_constraints.has_bias",
         "platform_constraints.cores[0].has_bias",
         "platform_constraints.cores[1].has_bias"},
    ),
    "t01_13_casc_mmixcore_wq_s8_nobias": (
        "t0_16_casc_mmixcore_wq_s8_offload_sched_nobias",
        {"deployment_parameters.encoding_layer_placement",
         "deployment_parameters.allow_scheduling"},
    ),
    # The endpoint wall budget rides the depth class (fix-round item 5):
    # deepmlp deep (40 s) -> shallow (100 s) is induced by the depth move.
    "t01_14_casc_deepmlp_d8_wq_s16_residual": (
        "t0_19_casc_deepmlp_d16_wq_s16_residual",
        {"deployment_parameters.model_config.depth",
         "deployment_parameters.endpoint_floor_wall_s"},
    ),
    # sched rides the depth axis: W2 proved platform C packs d8 only scheduled.
    "t01_15_casc_deepcnn_d8_wq_s4_sched": (
        "t0_18_casc_deepcnn_d4_wq_s4_pruned",
        {"deployment_parameters.model_config.depth",
         "deployment_parameters.allow_scheduling",
         "deployment_parameters.pruning",
         "deployment_parameters.pruning_fraction",
         "deployment_parameters.endpoint_floor_wall_s"},
    ),
    # D - wall / training-ceiling decomposition
    "t01_16_lif_deepcnn_d8_wq_s8_sched": ("t0_03_lif_deepcnn_d8_wq_s16_sched", _S_MOVE),
    "t01_17_ttfs_deepcnn_d8_fp_s16_sched": ("t0_08_ttfs_deepcnn_d8_fp_s32_sched", _S_MOVE),
    "t01_18_casc_lenet5_wq_s16": ("t0_17_casc_lenet5_wq_s32", _S_MOVE),
    "t01_19_lif_deepcnn_d6_wq_s16_sched": (
        "t0_03_lif_deepcnn_d8_wq_s16_sched",
        {"deployment_parameters.model_config.depth"},
    ),
    # E - quantization-resolution / WQ gap
    "t01_20_sync_mmixcore_wq_s8_pruned10_wb8": ("t0_21_sync_mmixcore_wq_s8_pruned10", _WB_MOVE),
    "t01_21_lif_mmixcore_wq_s4_wb8": ("t0_01_lif_mmixcore_wq_s4", _WB_MOVE),
    "t01_22_ttfsq_deepcnn_d4_wq_s4_wb4": ("t0_13_ttfsq_deepcnn_d4_wq_s4", _WB_MOVE),
    # F - floor mechanics + controls
    "t01_23_ttfs_mmixcore_wq_s8_floor": ("t0_06_ttfs_mmixcore_wq_s8", _FLOOR_MOVE),
    "t01_24_sync_mmixcore_wq_s8_pruned10_floor": ("t0_21_sync_mmixcore_wq_s8_pruned10", _FLOOR_MOVE),
    "t01_25_ttfsq_lenet5_wq_s32": ("t0_12_ttfsq_lenet5_wq_s32", set()),
}

TIER01_FAMILY_SIZES = {"A": 6, "B": 5, "C": 4, "D": 4, "E": 3, "F": 3}


class TestTier01DiagnosticMatrix:
    """Tier-0.1: 25 controlled minimal pairs probing tier-0's failure modes."""

    def _runs(self):
        return _manifest("0_1")["runs"]

    def _load(self, tier, name):
        return json.loads((TEST_CONFIGS / f"tier{tier}" / f"{name}.json").read_text())

    def test_every_cell_is_a_minimal_pair_of_its_anchor(self):
        tier0_names = {r["name"] for r in _manifest(0)["runs"]}
        runs = {r["name"]: r for r in self._runs()}
        assert set(runs) == set(TIER01_EXPECTED_DELTAS)
        for name, (anchor, expected_delta) in TIER01_EXPECTED_DELTAS.items():
            assert runs[name]["anchor"] == anchor, name
            assert anchor in tier0_names, name
            cfg = self._load("0_1", name)
            anchor_cfg = self._load(0, anchor)
            delta = _config_delta(cfg, anchor_cfg) - {"experiment_name"}
            assert delta == expected_delta, (name, sorted(delta))

    def test_manifest_carries_falsifiable_diagnostics(self):
        for run in self._runs():
            assert run["family"] in TIER01_FAMILY_SIZES, run["name"]
            assert isinstance(run["axes_moved"], list), run["name"]
            # Minimal pair: at most 2 axes; 0 only for the pure green control.
            assert len(run["axes_moved"]) <= 2, run["name"]
            if not run["axes_moved"]:
                assert run["name"] == "t01_25_ttfsq_lenet5_wq_s32"
            assert isinstance(run["hypothesis"], str), run["name"]
            assert len(run["hypothesis"]) >= 40, run["name"]

    def test_family_sizes(self):
        counts = {}
        for run in self._runs():
            counts[run["family"]] = counts.get(run["family"], 0) + 1
        assert counts == TIER01_FAMILY_SIZES

    def test_envelope_cells_train_four_epochs_and_others_two(self):
        for run in self._runs():
            dp = self._load("0_1", run["name"])["deployment_parameters"]
            expected = 4 if run["family"] == "B" else 2
            assert dp["training_epochs"] == expected, run["name"]

    def test_floor_cells_carry_the_overridable_wall_knob(self):
        # Fix-round item 5: every cell carries the sized RUN-total endpoint
        # wall budget (clamp(280 - measured class base, 40, 150)); the two
        # F-family diagnostics keep their explicit 600 s override.
        floor_cells = {"t01_23_ttfs_mmixcore_wq_s8_floor",
                       "t01_24_sync_mmixcore_wq_s8_pruned10_floor"}
        for run in self._runs():
            dp = self._load("0_1", run["name"])["deployment_parameters"]
            if run["name"] in floor_cells:
                assert dp["endpoint_floor_wall_s"] == 600, run["name"]
            else:
                assert 40 <= dp["endpoint_floor_wall_s"] <= 150, run["name"]

    def test_endpoint_wall_budgets_follow_the_measured_arithmetic(self):
        # Spot checks of the manifest-documented sizing on both matrices.
        assert self._load(
            0, "t0_03_lif_deepcnn_d8_wq_s16_sched",
        )["deployment_parameters"]["endpoint_floor_wall_s"] == 40
        assert self._load(
            0, "t0_12_ttfsq_lenet5_wq_s32",
        )["deployment_parameters"]["endpoint_floor_wall_s"] == 150
        assert self._load(
            0, "t0_21_sync_mmixcore_wq_s8_pruned10",
        )["deployment_parameters"]["endpoint_floor_wall_s"] == 100
        assert self._load(
            "0_1", "t01_14_casc_deepmlp_d8_wq_s16_residual",
        )["deployment_parameters"]["endpoint_floor_wall_s"] == 100

    def test_green_control_is_a_pure_t0_12_clone(self):
        cfg = self._load("0_1", "t01_25_ttfsq_lenet5_wq_s32")
        anchor = self._load(0, "t0_12_ttfsq_lenet5_wq_s32")
        assert _config_delta(cfg, anchor) == {"experiment_name"}

    def test_manifest_quant_axis_is_runtime_truth(self):
        aq_on = {"lif", "ttfs_quantized", "ttfs_cycle_based"}
        for run in self._runs():
            cell = run["cell"]
            wq = self._load("0_1", run["name"])["deployment_parameters"]["weight_quantization"]
            if not wq:
                assert cell["quantization"] == "none", run["name"]
            elif cell["firing"] in aq_on:
                assert cell["quantization"] == "wq_aq", run["name"]
            else:
                assert cell["quantization"] == "wq", run["name"]

    def test_sim_sample_respec_is_note_sanctioned(self):
        notes = _notes("0_1")
        assert "sim-sample respec" in notes["t01_17_ttfs_deepcnn_d8_fp_s16_sched.json"].lower()
