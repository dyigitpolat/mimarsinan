from mimarsinan.mapping.platform_constraints import resolve_scalar_mapping_params


def test_scalar_legacy_bias_reserves_axon():
    p = resolve_scalar_mapping_params(
        max_axons=256, max_neurons=128, hardware_bias=False
    )
    assert p.effective_max_axons == 255


def test_scalar_hardware_bias():
    p = resolve_scalar_mapping_params(
        max_axons=256, max_neurons=128, hardware_bias=True
    )
    assert p.effective_max_axons == 256
