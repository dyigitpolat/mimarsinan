"""The tiered test_configs are schema-valid, legal, and cover the claimed pairs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TEST_CONFIGS = ROOT / "test_configs"

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
        assert len(_tier_configs(1)) == 8
        assert len(_tier_configs(2)) == 3

    def test_manifest_matches_files(self):
        for tier in (0, 1, 2):
            manifest = _manifest(tier)
            names = {r["config"] for r in manifest["runs"]}
            files = {p.name for p in _tier_configs(tier)}
            assert names == files


class TestConfigValidity:
    @pytest.mark.parametrize("tier", [0, 1, 2])
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

    @pytest.mark.parametrize("tier", [0, 1, 2])
    def test_sim_enables_left_to_derivation(self, tier):
        """Sim backends are ConversionPolicy-derived; configs must not pin them."""
        for path in _tier_configs(tier):
            cfg = json.loads(path.read_text())
            assert not (set(cfg["deployment_parameters"]) & SIM_ENABLE_KEYS), path.name

    @pytest.mark.parametrize("tier", [0, 1, 2])
    def test_legality_rules(self, tier):
        for path in _tier_configs(tier):
            dp = json.loads(path.read_text())["deployment_parameters"]
            if dp["spiking_mode"] in QUANT_REQUIRED_MODES:
                assert dp["weight_quantization"] is True, path.name
            assert dp["max_simulation_samples"] == 100, path.name

    @pytest.mark.parametrize("tier", [0, 1, 2])
    def test_activation_quantization_left_to_derivation(self, tier):
        """AQ is a derived pipeline-assembly mode; configs must never pin it."""
        for path in _tier_configs(tier):
            dp = json.loads(path.read_text())["deployment_parameters"]
            assert "activation_quantization" not in dp, path.name

    @pytest.mark.parametrize("tier", [0, 1, 2])
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

    @pytest.mark.parametrize("tier", [0, 1, 2])
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
