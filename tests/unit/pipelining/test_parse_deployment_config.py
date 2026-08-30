"""``parse_deployment_config`` — persisted config must keep ``search_space``."""

import json

from mimarsinan.pipelining.session import parse_deployment_config


def _minimal_deployment_config(tmp_path, hw_search: bool) -> dict:
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
            "max_residency_classes": 3,
        }
    return base


class TestParseDeploymentConfigSearchSpacePreserved:
    def test_hw_search_does_not_mutate_input_platform_constraints(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        ss_before = cfg["platform_constraints"]["search_space"].copy()

        parse_deployment_config(cfg)

        assert "search_space" in cfg["platform_constraints"]
        assert cfg["platform_constraints"]["search_space"] == ss_before

    def test_hw_search_runtime_platform_constraints_has_no_search_space(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        parsed = parse_deployment_config(cfg)

        assert "search_space" not in parsed.platform_constraints

    def test_written_config_json_retains_search_space(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        parse_deployment_config(cfg)

        written = json.loads(
            (tmp_path / "run_wd" / "_RUN_CONFIG" / "config.json").read_text(encoding="utf-8")
        )
        assert written["platform_constraints"].get("search_space", {}).get("num_core_types") == 2

    def test_merge_into_arch_search(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=True)
        parsed = parse_deployment_config(cfg)

        arch = parsed.deployment_parameters["arch_search"]
        assert arch.get("num_core_types") == 2
        assert arch.get("max_residency_classes") == 3


class TestParseDeploymentConfigFixedHw:
    def test_fixed_hw_unchanged(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=False)
        pc_before = dict(cfg["platform_constraints"])

        parse_deployment_config(cfg)

        assert cfg["platform_constraints"] == pc_before


class TestParseDeploymentConfigRunName:
    """The run's NAME must reach the merged pipeline config, like its seed.

    Every artifact a step signs -- the ODIN deployment bundle's provenance and
    its default bundle name are the ones that made this visible -- names the run
    through ``config["experiment_name"]``. It is a TOP-LEVEL document key, so a
    step reading the merged config saw an empty string on every real run while
    unit harnesses (which build the flat config themselves) saw the real name.
    """

    def test_experiment_name_reaches_the_merged_parameters(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=False)

        parsed = parse_deployment_config(cfg)

        assert parsed.deployment_parameters["experiment_name"] == "t_parse"
        assert parsed.deployment_name == "t_parse"

    def test_an_explicit_parameter_wins_over_the_document_key(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=False)
        cfg["deployment_parameters"]["experiment_name"] = "explicit"

        parsed = parse_deployment_config(cfg)

        assert parsed.deployment_parameters["experiment_name"] == "explicit"

    def test_the_input_document_is_not_mutated(self, tmp_path):
        cfg = _minimal_deployment_config(tmp_path, hw_search=False)
        parameters_before = dict(cfg["deployment_parameters"])

        parse_deployment_config(cfg)

        assert cfg["deployment_parameters"] == parameters_before
