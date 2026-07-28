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


class TestActivationAlignmentCapability:
    """[D2] steps ask the POLICY whether the alignment ladder applies."""

    def test_value_cores_need_no_alignment(self):
        from mimarsinan.chip_simulation.mvm_core_policy import MvmCorePolicy
        assert MvmCorePolicy().requires_activation_alignment() is False

    def test_event_families_do(self):
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )
        for mode in ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert policy_for_spiking_mode(mode).requires_activation_alignment()

    def test_the_alignment_steps_consult_it_not_the_domain(self):
        import inspect

        from mimarsinan.pipelining.pipeline_steps.adaptation import (
            activation_adaptation_step,
            activation_analysis_step,
        )
        for module in (activation_adaptation_step, activation_analysis_step):
            source = inspect.getsource(module)
            assert "requires_activation_alignment" in source
            assert "is_mvm" not in source, (
                f"{module.__name__} still tests the domain directly"
            )


class TestObservesValuesCapability:
    """[D2] the certificate and the deployed-metric read share ONE axis."""

    def test_value_family_observes_values(self):
        from mimarsinan.chip_simulation.mvm_core_policy import MvmCorePolicy
        assert MvmCorePolicy().observes_values() is True

    def test_event_families_do_not(self):
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )
        for mode in ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert policy_for_spiking_mode(mode).observes_values() is False

    def test_it_is_derived_from_the_observable_not_duplicated(self):
        # A domain cannot answer the two questions inconsistently.
        from mimarsinan.chip_simulation.mvm_core_policy import MvmCorePolicy
        policy = MvmCorePolicy()
        observable, _ = policy.certification_observable()
        assert policy.observes_values() == (observable == "values")

    def test_hcm_metric_dispatch_asks_the_policy(self):
        import inspect

        from mimarsinan.pipelining.pipeline_steps.mapping import hard_core_mapping_step
        source = inspect.getsource(hard_core_mapping_step)
        assert "observes_values()" in source
        assert "is_mvm" not in source, "HCM still tests the domain directly"
