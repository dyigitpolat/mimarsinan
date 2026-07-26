"""MvmCorePolicy: the value-domain policy implements exactly the consumed seams."""

import pytest

from mimarsinan.chip_simulation.mvm_core_policy import MvmCorePolicy
from mimarsinan.chip_simulation.spiking_mode_policy import SpikingModePolicy


@pytest.fixture
def policy():
    return MvmCorePolicy()


class TestTypedSeams:
    def test_certification_observable_is_values(self, policy):
        assert policy.certification_observable() == ("values", None)

    def test_training_forward_kind(self, policy):
        assert policy.training_forward_kind() == "value"

    def test_no_dedicated_activation_replacement_step(self, policy):
        assert policy.single_step_activation_replacement is False

    def test_no_conversion_health_calibration(self, policy):
        assert policy.does_conversion_health_calibration is False

    def test_requires_ttfs_firing_is_false(self, policy):
        assert policy.requires_ttfs_firing is False


class TestBackendCapability:
    @pytest.mark.parametrize(
        "backend", ["nevresim", "sanafe", "loihi", "lava", "hcm", "training"]
    )
    def test_no_spiking_backend_supports_mvm(self, policy, backend):
        assert policy.supports_backend(backend) is False

    def test_valid_backends_is_empty(self, policy):
        assert policy.valid_backends(("nevresim", "sanafe", "loihi")) == ()

    def test_require_backend_supported_raises(self, policy):
        with pytest.raises(ValueError, match="value-domain"):
            policy.require_backend_supported(backend="nevresim", context="test")


class TestUnreachedSeamsStayLoud:
    def test_decode_mode_raises(self, policy):
        with pytest.raises(NotImplementedError):
            policy.decode_mode()

    def test_calibration_forward_raises(self, policy):
        with pytest.raises(NotImplementedError):
            policy.calibration_forward()

    def test_is_a_mode_policy(self, policy):
        # Consumers type against the SpikingModePolicy duck surface.
        assert isinstance(policy, SpikingModePolicy)
        assert policy.spiking_mode == "mvm"
