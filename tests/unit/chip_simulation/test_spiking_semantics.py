import pytest

from mimarsinan.chip_simulation.spiking_semantics import (
    ANALYTICAL_TTFS_MODES,
    CYCLE_BASED_MODES,
    TTFS_FAMILY_MODES,
    forces_activation_quantization,
    is_analytical_ttfs,
    is_cascaded_ttfs,
    is_cycle_based,
    is_default_firing_mode,
    is_explicit_ttfs_cycle_schedule,
    is_novena_firing_mode,
    is_synchronized_ttfs,
    is_ttfs_cycle_based,
    require_spiking_mode_supported,
    requires_ttfs_firing,
    supports_spiking_mode,
    ttfs_cycle_schedule,
    uses_ttfs_floor_ceil_convention,
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


class TestCycleSchedule:
    def test_default_is_cascaded(self):
        assert ttfs_cycle_schedule(None) == "cascaded"
        assert ttfs_cycle_schedule("") == "cascaded"
        assert ttfs_cycle_schedule("bogus") == "cascaded"
        assert ttfs_cycle_schedule("synchronized") == "synchronized"

    def test_cascaded_predicate(self):
        assert is_cascaded_ttfs("ttfs_cycle_based", "cascaded")
        assert is_cascaded_ttfs("ttfs_cycle_based", None)  # default cascaded
        assert not is_cascaded_ttfs("ttfs_cycle_based", "synchronized")
        assert not is_cascaded_ttfs("lif", "cascaded")
        assert not is_cascaded_ttfs("ttfs_quantized", "cascaded")

    def test_synchronized_predicate(self):
        assert is_synchronized_ttfs("ttfs_cycle_based", "synchronized")
        assert not is_synchronized_ttfs("ttfs_cycle_based", "cascaded")
        assert not is_synchronized_ttfs("ttfs_cycle_based", None)  # default cascaded
        assert not is_synchronized_ttfs("lif", "synchronized")

    def test_explicit_schedule_predicate(self):
        # Explicit schedules are exactly the values ttfs_cycle_schedule leaves unchanged.
        assert is_explicit_ttfs_cycle_schedule("cascaded")
        assert is_explicit_ttfs_cycle_schedule("synchronized")
        for explicit in ("cascaded", "synchronized"):
            assert ttfs_cycle_schedule(explicit) == explicit
        for placeholder in (None, "", "analytical", "none", "bogus"):
            assert not is_explicit_ttfs_cycle_schedule(placeholder), placeholder

    def test_floor_ceil_convention_predicate(self):
        # ttfs_quantized AND the synchronized floor-collapse train the floor +
        # half-step-bias NF and deploy the mode-derived ceil kernel.
        assert uses_ttfs_floor_ceil_convention("ttfs_quantized")
        assert uses_ttfs_floor_ceil_convention("ttfs_quantized", "cascaded")
        assert uses_ttfs_floor_ceil_convention("ttfs_cycle_based", "synchronized")
        # cascaded (segment-spike), continuous ttfs, and lif keep their own kernels.
        assert not uses_ttfs_floor_ceil_convention("ttfs_cycle_based", "cascaded")
        assert not uses_ttfs_floor_ceil_convention("ttfs_cycle_based", None)
        assert not uses_ttfs_floor_ceil_convention("ttfs")
        assert not uses_ttfs_floor_ceil_convention("lif")


# ── firing-mode predicates ───────────────────────────────────────────────────

class TestFiringModePredicates:
    def test_default_firing_mode(self):
        assert is_default_firing_mode("Default")
        for other in ("Novena", "TTFS", "", "default", None):
            assert not is_default_firing_mode(other), other

    def test_novena_firing_mode(self):
        assert is_novena_firing_mode("Novena")
        for other in ("Default", "TTFS", "", "novena", None):
            assert not is_novena_firing_mode(other), other

    def test_predicates_accept_firing_mode_enum_members(self):
        from mimarsinan.chip_simulation.firing_strategy import FiringMode

        assert is_default_firing_mode(FiringMode.DEFAULT)
        assert is_novena_firing_mode(FiringMode.NOVENA)
        assert not is_novena_firing_mode(FiringMode.DEFAULT)
        assert not is_default_firing_mode(FiringMode.TTFS)
