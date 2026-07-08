"""Explicit-keys-only emission: verbatim, canonical order, loud unknown report."""

from mimarsinan.gui.wizard.emit import emit_deployment_config, emit_with_unknown_report


class TestEmit:
    def test_unknown_keys_are_preserved_verbatim(self):
        out = emit_deployment_config({
            "deployment_parameters": {"endpoint_floor_wall_s": 60, "lr": 0.01},
        })
        assert out["deployment_parameters"] == {"endpoint_floor_wall_s": 60, "lr": 0.01}

    def test_owned_derived_and_runtime_keys_are_removed(self):
        out = emit_deployment_config({
            "device": "cuda",
            "deployment_parameters": {
                "activation_quantization": True,
                "weight_quantization": True,
                "firing_mode": "TTFS",
            },
            "platform_constraints": {"weight_bits": 4},
        })
        assert "device" not in out
        dp = out["deployment_parameters"]
        assert "activation_quantization" not in dp
        assert dp["weight_quantization"] is True
        assert dp["firing_mode"] == "TTFS"

    def test_canonical_top_level_order(self):
        out = emit_deployment_config({
            "stop_step": None,
            "experiment_name": "x",
            "seed": 3,
            "pipeline_mode": "phased",
        })
        keys = list(out)
        assert keys.index("seed") < keys.index("pipeline_mode") < keys.index("experiment_name")
        assert keys.index("platform_constraints") < keys.index("deployment_parameters")

    def test_required_run_identity_defaults_fill_in(self):
        out = emit_deployment_config({})
        assert out["data_provider_name"] == "MNIST_DataProvider"
        assert out["experiment_name"] == "experiment"
        assert out["generated_files_path"] == "./generated"
        assert out["seed"] == 0
        assert out["start_step"] is None
        assert "stop_step" not in out
        assert "pipeline_mode" not in out

    def test_declared_vehicle_off_survives_emission(self):
        # Round-3 defect 6: user-off on a supported simulator is a legitimate
        # declarable override — the emitted document must carry it.
        out = emit_deployment_config({
            "deployment_parameters": {
                "spiking_mode": "lif",
                "enable_sanafe_simulation": False,
            },
        })
        assert out["deployment_parameters"]["enable_sanafe_simulation"] is False
        # The non-declarable derived key still never survives.
        assert "activation_quantization" not in out["deployment_parameters"]

    def test_meta_keys_survive(self):
        out = emit_deployment_config({"_continue_from_run_id": "run_9"})
        assert out["_continue_from_run_id"] == "run_9"

    def test_unknown_report(self):
        out, unknown = emit_with_unknown_report({
            "deployment_parameters": {"endpoint_floor_wall_s": 60},
        })
        assert unknown == ["deployment_parameters.endpoint_floor_wall_s"]
        assert out["deployment_parameters"]["endpoint_floor_wall_s"] == 60
