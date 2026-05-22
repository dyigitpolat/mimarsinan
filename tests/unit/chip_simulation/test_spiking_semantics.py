import pytest

from mimarsinan.chip_simulation.spiking_semantics import (
    require_spiking_mode_supported,
    supports_spiking_mode,
)


def test_sanafe_supports_ttfs_modes():
    assert supports_spiking_mode("sanafe", "ttfs")
    assert supports_spiking_mode("sanafe", "ttfs_quantized")


def test_loihi_does_not_support_ttfs():
    assert not supports_spiking_mode("loihi", "ttfs")


def test_require_raises_for_loihi_ttfs():
    with pytest.raises(ValueError, match="loihi"):
        require_spiking_mode_supported(
            "ttfs_quantized", backend="loihi", context="LoihiSimulationStep",
        )
