from mimarsinan.config_schema.defaults import get_default_deployment_parameters
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


def test_ttfs_cycle_based_finetune_disables_activation_quant():
    # LIF-style: TTFSCycleActivation subsumes the quant chain, so activation
    # quantization is forced OFF when fine-tuning is on (the default).
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["activation_quantization"] is False


def test_ttfs_cycle_based_without_finetune_enables_activation_quant():
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "enable_ttfs_finetuning": False}
    derive_deployment_parameters(dp)
    assert dp["activation_quantization"] is True


def test_ttfs_cycle_based_synchronized_disables_nevresim():
    # No genuine synchronized-window nevresim backend yet → forced off.
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "synchronized", "enable_nevresim_simulation": True}
    derive_deployment_parameters(dp)
    assert dp["enable_nevresim_simulation"] is False


def test_ttfs_cycle_based_cascaded_keeps_nevresim():
    # Cascaded greedy TTFS runs genuinely on nevresim (fire-once-latch policy).
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "cascaded", "enable_nevresim_simulation": True}
    derive_deployment_parameters(dp)
    assert dp["enable_nevresim_simulation"] is True


def test_ttfs_cycle_based_default_schedule_keeps_nevresim():
    # Default schedule is cascaded → nevresim stays enabled.
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "enable_nevresim_simulation": True}
    derive_deployment_parameters(dp)
    assert dp["enable_nevresim_simulation"] is True


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


def test_default_cycle_accurate_lif_forward_is_true():
    dp = get_default_deployment_parameters()
    assert dp["cycle_accurate_lif_forward"] is True


def test_default_enable_nevresim_simulation_is_true():
    dp = get_default_deployment_parameters()
    assert dp["enable_nevresim_simulation"] is True


def test_config_builder_cycle_accurate_default_for_lif():
    cfg = build_deployment_config_from_state({})
    assert cfg["deployment_parameters"]["cycle_accurate_lif_forward"] is True
