"""Validity tests for the generated deployment-mode template matrices.

These are NOT part of the mimarsinan unit suite (``testpaths = tests`` excludes
``scripts/``). The tier_0/1/2 matrices under ``templates/`` are integration-run
deployment-mode examples, not tests; this file checks that their generator is
the SSOT (regeneration is a no-op) and that every emitted config is schema-valid,
legal, and survives derivation. Run with: ``pytest scripts/template_tests``.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
GENERATOR = TEMPLATES / "generate.py"
TIERS = (0, 1, 2, 3)

TOP_LEVEL_KEYS = {
    "seed", "pipeline_mode", "experiment_name", "generated_files_path",
    "data_provider_name", "platform_constraints", "deployment_parameters",
    "target_metric_override", "start_step", "stop_step",
}
QUANT_REQUIRED_MODES = {"ttfs_quantized", "ttfs_cycle_based"}
SIM_ENABLE_KEYS = {
    "enable_nevresim_simulation", "enable_sanafe_simulation", "enable_loihi_simulation",
}


def _tier_dir(tier):
    return TEMPLATES / f"tier_{tier}"


def _tier_configs(tier):
    return sorted(p for p in _tier_dir(tier).glob("t*.json"))


def _manifest(tier):
    return json.loads((_tier_dir(tier) / "manifest.json").read_text())


class TestGeneratorIsTheSSOT:
    def test_generator_reproduces_committed_files(self):
        """Regenerating must be a no-op: the JSONs never drift from generate.py."""
        before = {
            p.relative_to(TEMPLATES): p.read_text()
            for tier in TIERS for p in _tier_dir(tier).rglob("*.json")
        }
        subprocess.run(
            [sys.executable, str(GENERATOR)], check=True, capture_output=True,
        )
        after = {
            p.relative_to(TEMPLATES): p.read_text()
            for tier in TIERS for p in _tier_dir(tier).rglob("*.json")
        }
        assert before == after

    def test_manifest_matches_files(self):
        for tier in TIERS:
            manifest = _manifest(tier)
            listed = {r["config"] for r in manifest["runs"]}
            on_disk = {p.name for p in _tier_configs(tier)}
            assert listed == on_disk, tier


class TestConfigValidity:
    @pytest.mark.parametrize("tier", TIERS)
    def test_keys_are_known(self, tier):
        from mimarsinan.config_schema.defaults import (
            CONFIG_KEYS_SET,
            DEFAULT_PLATFORM_CONSTRAINTS,
        )

        platform_keys = set(DEFAULT_PLATFORM_CONSTRAINTS) | {
            # Defaultless platform keys: absence is meaningful (activation_bits
            # absent = float boundary I/O on value-domain platforms).
            "cores", "max_axons", "max_neurons", "has_bias", "activation_bits",
        }
        for path in _tier_configs(tier):
            cfg = json.loads(path.read_text())
            assert set(cfg) == TOP_LEVEL_KEYS, path.name
            unknown = set(cfg["deployment_parameters"]) - set(CONFIG_KEYS_SET)
            assert not unknown, f"{path.name}: unknown keys {unknown}"
            unknown_pc = set(cfg["platform_constraints"]) - platform_keys
            assert not unknown_pc, f"{path.name}: unknown platform keys {unknown_pc}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_sim_enables_and_aq_left_to_derivation(self, tier):
        for path in _tier_configs(tier):
            dp = json.loads(path.read_text())["deployment_parameters"]
            assert not (set(dp) & SIM_ENABLE_KEYS), path.name
            assert "activation_quantization" not in dp, path.name

    @pytest.mark.parametrize("tier", TIERS)
    def test_legality_rules(self, tier):
        for path in _tier_configs(tier):
            dp = json.loads(path.read_text())["deployment_parameters"]
            if dp.get("spiking_mode") in QUANT_REQUIRED_MODES:
                assert dp["weight_quantization"] is True, path.name
            assert dp["max_simulation_samples"] == 25, path.name

    @pytest.mark.parametrize("tier", TIERS)
    def test_quant_tags_are_runtime_truth(self, tier):
        for path in _tier_configs(tier):
            assert "_aq_" not in path.name and "_wqaq_" not in path.name, path.name
            cfg = json.loads(path.read_text())
            wq = cfg["deployment_parameters"]["weight_quantization"]
            tag = "_wq" if wq else "_fp"
            # Gridless (mvm) rows have no trailing S part, so the quant tag
            # may terminate the stem.
            assert f"{tag}_" in path.name or path.name.endswith(f"{tag}.json"), path.name
            assert cfg["pipeline_mode"] == ("phased" if wq else "vanilla"), path.name

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
            declared = cfg["deployment_parameters"]
            if "spiking_mode" in declared:
                assert merged["spiking_mode"] == declared["spiking_mode"]
            else:
                assert declared.get("core_semantics") == "mvm", path.name
                # [mvm AQ] platform activation_bits is the sole armer.
                assert merged["activation_quantization"] is bool(
                    cfg["platform_constraints"].get("activation_bits")
                ), path.name


class TestCascDescopedFromTier0:
    """casc is de-scoped from the tier_0 MNIST family (user directive 2026-07-12);
    it survives in tier_1/tier_2 coverage."""

    def _is_casc(self, dp):
        return (
            dp.get("spiking_mode") == "ttfs_cycle_based"
            and dp.get("ttfs_cycle_schedule") == "cascaded"
        )

    def test_no_tier0_cell_is_cascaded(self):
        for path in _tier_configs(0):
            dp = json.loads(path.read_text())["deployment_parameters"]
            assert not self._is_casc(dp), path.name

    def test_casc_survives_in_higher_tiers(self):
        higher = [
            p for tier in (1, 2) for p in _tier_configs(tier)
            if self._is_casc(json.loads(p.read_text())["deployment_parameters"])
        ]
        assert higher, "casc coverage must survive in tier_1/tier_2"


class TestModeCoverage:
    def test_tier0_covers_every_deployment_mode(self):
        modes = {
            json.loads(p.read_text())["deployment_parameters"].get("spiking_mode")
            for p in _tier_configs(0)
        }
        assert {"lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"} <= modes

    def test_tier0_covers_the_value_domain_family(self):
        semantics = {
            json.loads(p.read_text())["deployment_parameters"].get("core_semantics", "spiking")
            for p in _tier_configs(0)
        }
        assert "mvm" in semantics
