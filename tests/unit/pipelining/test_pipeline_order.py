"""Regression guard: Pruning Adaptation must precede Activation Analysis.

This was the ordering the code already used (see ``deployment_pipeline.py``),
but ``ARCHITECTURE.md`` documented the old ordering. The test locks the code
behaviour in place so doc/code never drift again.
"""

import pytest

from mimarsinan.pipelining.pipelines.deployment_pipeline import get_pipeline_step_specs


def _names(config):
    return [name for name, _ in get_pipeline_step_specs(config)]


def _config(**overrides):
    cfg = {
        "configuration_mode": "user",
        "spiking_mode": "ttfs",
        "activation_quantization": True,
        "weight_quantization": True,
        "pruning": True,
        "pruning_fraction": 0.10,
        "model_type": "mlp_mixer",
    }
    cfg.update(overrides)
    return cfg


class TestPruningBeforeActivationAnalysis:
    def test_pruning_precedes_activation_analysis(self):
        names = _names(_config())
        assert "Pruning Adaptation" in names, f"steps were: {names}"
        assert "Activation Analysis" in names
        assert names.index("Pruning Adaptation") < names.index("Activation Analysis"), (
            "Pruning Adaptation must run before Activation Analysis (matches "
            "deployment_pipeline.py's step specs)."
        )

    def test_pruning_precedes_clamp_adaptation(self):
        names = _names(_config())
        assert "Pruning Adaptation" in names
        assert "Clamp Adaptation" in names
        assert names.index("Pruning Adaptation") < names.index("Clamp Adaptation")

    def test_pruning_precedes_weight_quantization(self):
        names = _names(_config())
        assert "Pruning Adaptation" in names
        assert "Weight Quantization" in names
        assert names.index("Pruning Adaptation") < names.index("Weight Quantization")
