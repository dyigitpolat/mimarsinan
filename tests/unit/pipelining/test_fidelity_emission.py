"""The C4 fidelity emission wiring: searched runs write, others write nothing."""

from types import SimpleNamespace

from mimarsinan.pipelining.pipeline_steps.verification.fidelity_emission import (
    _fidelity_axis_names,
    emit_run_fidelity,
)


def _pipeline(config):
    return SimpleNamespace(config=config, working_directory="/nonexistent")


class TestTheEmissionGate:
    def test_a_fixed_run_writes_nothing(self):
        """No search, no prediction, no correlation — None, not an empty file."""
        config = {"model_config_mode": "user", "hw_config_mode": "user"}
        assert emit_run_fidelity(_pipeline(config), record=None) is None

    def test_a_searched_run_without_declared_objectives_writes_nothing(self):
        config = {
            "model_config_mode": "user", "hw_config_mode": "search",
            "arch_search": {},
        }
        assert emit_run_fidelity(_pipeline(config), record=None) is None


class TestTheThroughPath:
    def test_a_searched_run_emits_the_report(self, tmp_path):
        """END TO END: the rebuild resolves the run's own config and the
        report lands beside the record — the gate tests alone missed a
        malformed candidate_layout call (the study's first launch)."""
        from unit.deployment_record.test_fidelity_from_record import _record

        config = {
            "device": "cpu",
            "input_shape": (1, 8, 8),
            "num_classes": 4,
            "target_tq": 4,
            "weight_bits": 4,
            "lr": 0.001,
            "seed": 0,
            "model_type": "simple_mlp",
            "model_config_mode": "user",
            "hw_config_mode": "search",
            "model_config": {
                "mlp_width_1": 16, "mlp_width_2": 16,
                "base_activation": "ReLU",
            },
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
            "platform_physics_profile": "truenorth",
            "activity_factor": 0.05,
            "arch_search": {
                "objectives": ["param_utilization_pct", "chip_area_mm2"],
            },
        }
        pipeline = SimpleNamespace(
            config=config, working_directory=str(tmp_path),
        )
        path = emit_run_fidelity(pipeline, _record())
        assert path is not None
        assert (tmp_path / "fidelity.json").exists()

    def test_the_twin_is_the_sealed_winner_not_the_declaration(self, tmp_path):
        """The raw config keeps the PRE-SEARCH declaration; the rebuild must
        resolve the record's own deployed platform, or fidelity compares the
        measurement against a chip the run never deployed (the LeNet5
        stretch's live catch: 31.9M declared vs 10.9M deployed cells)."""
        import json

        from unit.deployment_record.test_fidelity_from_record import _record

        config = {
            "device": "cpu", "input_shape": (1, 8, 8), "num_classes": 4,
            "target_tq": 4, "weight_bits": 4, "lr": 0.001, "seed": 0,
            "model_type": "simple_mlp",
            "model_config_mode": "user", "hw_config_mode": "search",
            "model_config": {
                "mlp_width_1": 16, "mlp_width_2": 16,
                "base_activation": "ReLU",
            },
            # DELIBERATELY not the record's platform (20 cores of 256x256).
            "cores": [{"max_axons": 128, "max_neurons": 128, "count": 999}],
            "platform_physics_profile": "truenorth",
            "activity_factor": 0.05,
            "arch_search": {
                "objectives": ["param_utilization_pct", "chip_area_mm2"],
            },
        }
        pipeline = SimpleNamespace(
            config=config, working_directory=str(tmp_path),
        )
        emit_run_fidelity(pipeline, _record())
        report = json.load(open(tmp_path / "fidelity.json"))
        capacity = [
            a for a in report["axes"] if a["key"] == "total_param_capacity"
        ][0]
        assert capacity["predicted"] == 20 * 256 * 256.0
        assert capacity["predicted"] == capacity["measured"]


class TestTheAxisFilter:
    def test_the_training_proxy_is_dropped(self):
        """The rebuild never trains; hardware completeness is what a deployed
        config can answer."""
        names = _fidelity_axis_names({
            "arch_search": {
                "objectives": ["estimated_accuracy", "chip_area_mm2"],
            },
        })
        assert names == ["chip_area_mm2"]

    def test_a_proxy_only_search_yields_no_axes(self):
        names = _fidelity_axis_names({
            "arch_search": {"objectives": ["estimated_accuracy"]},
        })
        assert names is None
