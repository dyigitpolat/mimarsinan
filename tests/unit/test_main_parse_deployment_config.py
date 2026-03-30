"""Tests for ``main._parse_deployment_config`` — persisted config must keep ``search_space``."""

import json

from main import _parse_deployment_config


def _minimal_deployment_config(tmp_path, hw_search: bool) -> dict:
    """Minimal valid deployment dict for parse; writes under tmp_path."""
    base = {
        "experiment_name": "t_parse",
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": str(tmp_path),
        "seed": 0,
        "pipeline_mode": "phased",
        "deployment_parameters": {
            "hw_config_mode": "search" if hw_search else "fixed",
            "arch_search": {
                "optimizer": "nsga2",
                "pop_size": 4,
                "generations": 2,
            },
        },
        "platform_constraints": {
            "target_tq": 16,
            "simulation_steps": 16,
            "weight_bits": 8,
            "has_bias": True,
        },
        "_working_directory": str(tmp_path / "run_wd"),
    }
    if hw_search:
        base["platform_constraints"]["search_space"] = {
            "num_core_types": 2,
            "core_type_counts": [100, 100],
            "core_axons_bounds": [64, 1024],
            "core_neurons_bounds": [64, 1024],
            "max_threshold_groups": 3,
        }
    return base


class TestParseDeploymentConfigSearchSpacePreserved:
    def test_hw_search_does_not_mutate_input_platform_constraints(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        ss_before = cfg["platform_constraints"]["search_space"].copy()

        parsed = _parse_deployment_config(cfg)

        assert "search_space" in cfg["platform_constraints"]
        assert cfg["platform_constraints"]["search_space"] == ss_before

    def test_hw_search_runtime_platform_constraints_has_no_search_space(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        parsed = _parse_deployment_config(cfg)

        assert "search_space" not in parsed["platform_constraints"]

    def test_written_config_json_retains_search_space(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        _parse_deployment_config(cfg)

        written = json.loads((tmp_path / "run_wd" / "_RUN_CONFIG" / "config.json").read_text(encoding="utf-8"))
        assert written["platform_constraints"].get("search_space", {}).get("num_core_types") == 2

    def test_merge_into_arch_search(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        parsed = _parse_deployment_config(cfg)

        arch = parsed["deployment_parameters"]["arch_search"]
        assert arch.get("num_core_types") == 2
        assert arch.get("max_threshold_groups") == 3


class TestParseDeploymentConfigFixedHw:
    def test_fixed_hw_unchanged(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=False)
        pc_before = dict(cfg["platform_constraints"])

        _parse_deployment_config(cfg)

        assert cfg["platform_constraints"] == pc_before
