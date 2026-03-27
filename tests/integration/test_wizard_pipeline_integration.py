"""
Integration test: wizard-built config is accepted by the pipeline.

Builds a deployment config via the wizard config builder, resolves platform_constraints
as main.py does, and constructs DeploymentPipeline to ensure protocol compatibility.
Uses MockDataProviderFactory (tiny in-memory data) to avoid network/dataset access.
"""

import pytest

from mimarsinan.gui.wizard import build_deployment_config_from_state, validate_wizard_state
from mimarsinan.config_schema import validate_deployment_config


def _noop_reporter():
    class R:
        def report(self, *args, **kwargs): pass
        def console_log(self, *args, **kwargs): pass
        def finish(self): pass
    return R()


def test_wizard_built_config_validates_and_pipeline_accepts(tmp_path):
    """Wizard-built config passes validation and DeploymentPipeline can be constructed."""
    from conftest import MockDataProviderFactory

    # Build config from minimal wizard state (same shape the UI would produce).
    # Use patch 4x4 and 8x8 input so patch divides input (MockDataProviderFactory uses (1,8,8)).
    state = {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "wizard_integration_test",
        "generated_files_path": str(tmp_path / "out"),
        "seed": 0,
        "pipeline_mode": "phased",
        "start_step": None,
        "stop_step": None,
        "target_metric_override": None,
        "deployment_parameters": {
            "configuration_mode": "user",
            "model_type": "mlp_mixer",
            "model_config": {
                "patch_n_1": 4,
                "patch_m_1": 4,
                "patch_c_1": 16,
                "fc_w_1": 32,
                "fc_w_2": 32,
            },
        },
        "platform_constraints": {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 1000}], "target_tq": 16, "simulation_steps": 16, "weight_bits": 8},
    }
    cfg = build_deployment_config_from_state(state)
    assert validate_deployment_config(cfg) == []
    assert validate_wizard_state(state) == []

    # Resolve platform_constraints as main.py does (flat user mode here)
    platform_constraints_raw = cfg["platform_constraints"]
    if isinstance(platform_constraints_raw, dict) and "mode" in platform_constraints_raw:
        mode = platform_constraints_raw.get("mode", "user")
        if mode == "user":
            platform_constraints = platform_constraints_raw.get(
                "user",
                {k: v for k, v in platform_constraints_raw.items() if k != "mode"},
            )
        else:
            auto = platform_constraints_raw.get("auto", {}) or {}
            platform_constraints = auto.get("fixed", {}) or {}
    else:
        platform_constraints = platform_constraints_raw

    deployment_parameters = dict(cfg["deployment_parameters"])
    from mimarsinan.pipelining.pipelines.deployment_pipeline import DeploymentPipeline
    DeploymentPipeline.apply_preset(cfg["pipeline_mode"], deployment_parameters)

    # Use mock data provider (tiny in-memory) so test does not need real dataset
    data_provider_factory = MockDataProviderFactory(input_shape=(1, 8, 8), num_classes=4)
    reporter = _noop_reporter()
    working_directory = str(tmp_path / "run")

    pipeline = DeploymentPipeline(
        data_provider_factory=data_provider_factory,
        deployment_parameters=deployment_parameters,
        platform_constraints=platform_constraints,
        reporter=reporter,
        working_directory=working_directory,
    )
    assert pipeline.config is not None
    assert pipeline.config.get("model_type") == "mlp_mixer"
    assert pipeline.config.get("configuration_mode") == "user"
