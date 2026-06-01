import pytest

from mimarsinan.chip_simulation.spiking_semantics import (
    ANALYTICAL_TTFS_MODES,
    CYCLE_BASED_MODES,
    TTFS_FAMILY_MODES,
    forces_activation_quantization,
    is_analytical_ttfs,
    is_cycle_based,
    is_ttfs_cycle_based,
    require_spiking_mode_supported,
    requires_ttfs_firing,
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


# ── ttfs_cycle_based taxonomy ────────────────────────────────────────────────

def test_ttfs_cycle_based_is_ttfs_family_but_not_analytical():
    """It uses TTFS firing/encoding but is NOT the closed-form analytical path."""
    assert "ttfs_cycle_based" in TTFS_FAMILY_MODES
    assert requires_ttfs_firing("ttfs_cycle_based")
    assert "ttfs_cycle_based" not in ANALYTICAL_TTFS_MODES
    assert not is_analytical_ttfs("ttfs_cycle_based")


def test_ttfs_cycle_based_is_cycle_based_like_lif():
    assert is_ttfs_cycle_based("ttfs_cycle_based")
    assert is_cycle_based("ttfs_cycle_based")
    assert is_cycle_based("lif")
    assert "ttfs_cycle_based" in CYCLE_BASED_MODES
    assert "lif" in CYCLE_BASED_MODES


def test_analytical_ttfs_modes_are_not_cycle_based():
    for mode in ("ttfs", "ttfs_quantized"):
        assert is_analytical_ttfs(mode)
        assert not is_cycle_based(mode)
        assert not is_ttfs_cycle_based(mode)


def test_forces_activation_quantization():
    assert forces_activation_quantization("ttfs_quantized")
    assert forces_activation_quantization("ttfs_cycle_based")
    assert not forces_activation_quantization("ttfs")
    assert not forces_activation_quantization("lif")


def test_backends_support_ttfs_cycle_based():
    for backend in ("hcm", "nevresim", "unified", "hybrid", "sanafe"):
        assert supports_spiking_mode(backend, "ttfs_cycle_based"), backend


def test_loihi_lava_training_reject_ttfs_cycle_based():
    for backend in ("loihi", "lava", "training"):
        assert not supports_spiking_mode(backend, "ttfs_cycle_based"), backend
