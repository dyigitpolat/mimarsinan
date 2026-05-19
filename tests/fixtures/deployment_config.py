"""Minimal deployment configs for tests."""

from __future__ import annotations

from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state


def minimal_deployment_config(**overrides) -> dict:
    """Canonical minimal deployment JSON (Python builder, LIF phased preset)."""
    state = {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "test",
        "deployment_parameters": {
            "spiking_mode": "lif",
            "weight_quantization": True,
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
        },
        "platform_constraints": {
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 100}],
        },
        "model_type": "simple_mlp",
        "model_config": {"hidden_dims": [32]},
    }
    state.update(overrides)
    return build_deployment_config_from_state(state)
