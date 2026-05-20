from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state


def test_lif_disables_activation_quantization():
    dp = {"spiking_mode": "lif", "weight_quantization": True, "activation_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["activation_quantization"] is False


def test_ttfs_quantized_enables_activation_quant():
    dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["activation_quantization"] is True


def test_float_weights_vanilla():
    dp = {"weight_quantization": False, "spiking_mode": "ttfs_quantized"}
    derive_deployment_parameters(dp)
    assert dp["pipeline_mode"] == "vanilla"
    assert dp["weight_quantization"] is False
    assert dp["activation_quantization"] is False


def test_config_builder_pipeline_mode_sync():
    cfg = build_deployment_config_from_state({
        "pipeline_mode": "phased",
        "deployment_parameters": {"spiking_mode": "lif", "weight_quantization": True},
    })
    assert cfg["pipeline_mode"] == cfg["deployment_parameters"]["pipeline_mode"]


def test_config_builder_lif_derives_quant_flags():
    cfg = build_deployment_config_from_state({
        "deployment_parameters": {
            "spiking_mode": "lif",
            "activation_quantization": True,
            "weight_quantization": True,
        },
    })
    dp = cfg["deployment_parameters"]
    assert dp["activation_quantization"] is False
