"""SpikingModePolicy: behavior-carrying policy per (firing × sync) (Vector V2).

The dispatch rules that callers used to re-derive — training-forward kind,
calibration NF forward, SANA-FE soma name + attrs, log-potential, decode mode —
are now method overrides on one policy per (firing × sync). These tests pin the
truth table and lock byte-identity against the pre-refactor behavior.
"""

import pytest

from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
from mimarsinan.chip_simulation.spiking_mode_policy import (
    LifModePolicy,
    SpikingModePolicy,
    TtfsAnalyticalModePolicy,
    TtfsCascadeModePolicy,
    TtfsSyncCycleModePolicy,
    policy_for_spiking_mode,
)


def _contract(mode, schedule="cascaded"):
    cfg = {
        "spiking_mode": mode,
        "firing_mode": "TTFS" if mode.startswith("ttfs") else "Default",
        "thresholding_mode": "<=",
        "spike_generation_mode": "TTFS" if mode.startswith("ttfs") else "Uniform",
        "simulation_steps": 4,
        "ttfs_cycle_schedule": schedule,
        "encoding_layer_placement": "subsume",
    }
    return SpikingDeploymentContract.from_pipeline_config(cfg)


# ── selection ────────────────────────────────────────────────────────────────

class TestSelection:
    CASES = [
        ("lif", None, LifModePolicy),
        ("rate", None, LifModePolicy),
        ("ttfs", "cascaded", TtfsAnalyticalModePolicy),
        ("ttfs", "synchronized", TtfsAnalyticalModePolicy),
        ("ttfs_quantized", "cascaded", TtfsAnalyticalModePolicy),
        ("ttfs_quantized", "synchronized", TtfsAnalyticalModePolicy),
        ("ttfs_cycle_based", "cascaded", TtfsCascadeModePolicy),
        ("ttfs_cycle_based", "synchronized", TtfsSyncCycleModePolicy),
        ("ttfs_cycle_based", None, TtfsCascadeModePolicy),  # default cascaded
    ]

    @pytest.mark.parametrize("mode,schedule,cls", CASES)
    def test_policy_for_spiking_mode(self, mode, schedule, cls):
        assert isinstance(policy_for_spiking_mode(mode, schedule), cls)

    def test_unknown_mode_falls_back_to_lif(self):
        assert isinstance(policy_for_spiking_mode("frobnicate"), LifModePolicy)

    def test_none_mode_is_lif(self):
        assert isinstance(policy_for_spiking_mode(None), LifModePolicy)

    def test_from_contract_matches_factory(self):
        for mode, schedule, cls in self.CASES:
            contract = _contract(mode, schedule or "cascaded")
            policy = SpikingModePolicy.from_contract(contract)
            assert isinstance(policy, cls), (mode, schedule)


# ── training_forward_kind (was deployment_contract.py:89-102) ─────────────────

class TestTrainingForwardKind:
    CASES = [
        ("ttfs_cycle_based", "cascaded", "segment_spike"),
        ("ttfs_cycle_based", "synchronized", "analytical_staircase"),
        ("ttfs", "cascaded", "analytical_staircase"),
        ("ttfs", "synchronized", "analytical_staircase"),
        ("ttfs_quantized", "cascaded", "analytical_staircase"),
        ("ttfs_quantized", "synchronized", "analytical_staircase"),
        ("lif", "cascaded", "lif_cycle"),
        ("lif", "synchronized", "lif_cycle"),
        ("rate", "cascaded", "rate"),
        ("rate", "synchronized", "rate"),
    ]

    @pytest.mark.parametrize("mode,schedule,kind", CASES)
    def test_policy(self, mode, schedule, kind):
        assert policy_for_spiking_mode(mode, schedule).training_forward_kind() == kind

    @pytest.mark.parametrize("mode,schedule,kind", CASES)
    def test_contract_delegates_to_policy(self, mode, schedule, kind):
        assert _contract(mode, schedule).training_forward_kind() == kind


# ── decode mode ──────────────────────────────────────────────────────────────

class TestDecodeMode:
    CASES = [
        ("lif", None, "count"),
        ("rate", None, "count"),
        ("ttfs", "cascaded", "timing"),
        ("ttfs_quantized", "cascaded", "timing"),
        ("ttfs_cycle_based", "synchronized", "timing"),
        ("ttfs_cycle_based", "cascaded", "count"),  # count-based decode, LIF-like
    ]

    @pytest.mark.parametrize("mode,schedule,decode", CASES)
    def test_decode_mode(self, mode, schedule, decode):
        assert policy_for_spiking_mode(mode, schedule).decode_mode() == decode


# ── SANA-FE soma name + log_potential (was build.py:104-113 / 184-218) ────────

class TestSomaName:
    CASES = [
        ("lif", None, "lif"),
        ("rate", None, "lif"),
        ("ttfs", "cascaded", "ttfs_continuous"),
        ("ttfs", "synchronized", "ttfs_continuous"),
        ("ttfs_quantized", "cascaded", "ttfs_quantized"),
        ("ttfs_cycle_based", "synchronized", "ttfs_cycle"),
        ("ttfs_cycle_based", "cascaded", "ttfs_cascade"),
    ]

    @pytest.mark.parametrize("mode,schedule,name", CASES)
    def test_soma_hw_name(self, mode, schedule, name):
        assert policy_for_spiking_mode(mode, schedule).soma_hw_name() == name

    @pytest.mark.parametrize("mode,schedule,name", CASES)
    def test_matches_legacy_helper(self, mode, schedule, name):
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            soma_hw_name_for_spiking_mode,
        )

        assert soma_hw_name_for_spiking_mode(mode, schedule) == name


class TestLogPotential:
    CASES = [
        ("lif", None, False),
        ("rate", None, False),
        ("ttfs", "cascaded", True),
        ("ttfs_quantized", "synchronized", True),
        ("ttfs_cycle_based", "synchronized", True),
        ("ttfs_cycle_based", "cascaded", False),  # count decode, no V trace
    ]

    @pytest.mark.parametrize("mode,schedule,expected", CASES)
    def test_log_potential(self, mode, schedule, expected):
        assert policy_for_spiking_mode(mode, schedule).log_potential is expected


# ── SANA-FE soma model attributes (was the 5-way build chain) ─────────────────

class TestSomaModelAttributes:
    """Each policy builds the same attrs dict the legacy 5-way chain produced."""

    def test_lif_routes_to_lif_attrs(self):
        from mimarsinan.chip_simulation.sanafe.neuron_model import lif_model_attributes

        got = policy_for_spiking_mode("lif").soma_model_attributes(
            threshold=1.5, hardware_bias=0.25, active_start=2, active_length=4,
            firing_mode="Novena",
        )
        want = lif_model_attributes(
            threshold=1.5, hardware_bias=0.25, active_start=2, active_length=4,
            firing_mode="Novena",
        )
        assert got == want
        assert got["reset_mode"] == "hard"  # firing_mode threaded through

    def test_ttfs_continuous_routes_to_continuous_attrs(self):
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_continuous_model_attributes,
        )

        got = policy_for_spiking_mode("ttfs").soma_model_attributes(
            threshold=2.0, hardware_bias=None, active_start=1, active_length=1,
        )
        want = ttfs_continuous_model_attributes(
            threshold=2.0, hardware_bias=None, active_start=1, active_length=1,
        )
        assert got == want

    def test_ttfs_quantized_routes_to_quantized_attrs(self):
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_quantized_model_attributes,
        )

        got = policy_for_spiking_mode("ttfs_quantized").soma_model_attributes(
            threshold=2.0, hardware_bias=0.1, active_start=0, active_length=8,
        )
        want = ttfs_quantized_model_attributes(
            threshold=2.0, hardware_bias=0.1, active_start=0, active_length=8,
        )
        assert got == want

    def test_ttfs_sync_cycle_routes_to_cycle_attrs(self):
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_cycle_model_attributes,
        )

        got = policy_for_spiking_mode(
            "ttfs_cycle_based", "synchronized"
        ).soma_model_attributes(
            threshold=1.0, hardware_bias=0.5, active_start=8, active_length=4,
        )
        want = ttfs_cycle_model_attributes(
            threshold=1.0, hardware_bias=0.5, active_start=8, active_length=4,
        )
        assert got == want

    def test_ttfs_cascade_routes_to_cascade_attrs(self):
        from mimarsinan.chip_simulation.sanafe.neuron_model import (
            ttfs_cascade_model_attributes,
        )

        got = policy_for_spiking_mode(
            "ttfs_cycle_based", "cascaded"
        ).soma_model_attributes(
            threshold=1.0, hardware_bias=0.5, active_start=3, active_length=4,
        )
        want = ttfs_cascade_model_attributes(
            threshold=1.0, hardware_bias=0.5, active_start=3, active_length=4,
        )
        assert got == want


# ── calibration_forward (was neg_shift_bias.py:125-140) ───────────────────────

class TestCalibrationForward:
    def test_lif_returns_chip_aligned(self):
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

        assert (
            policy_for_spiking_mode("lif").calibration_forward()
            is chip_aligned_segment_forward
        )

    def test_analytical_modes_share_one_forward(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            _analytical_segment_calibration_forward,
        )

        for mode in ("ttfs", "ttfs_quantized"):
            assert (
                policy_for_spiking_mode(mode).calibration_forward()
                is _analytical_segment_calibration_forward
            )

    def test_ttfs_cycle_schedules_share_one_forward(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            _ttfs_segment_calibration_forward,
        )

        for schedule in ("cascaded", "synchronized"):
            assert (
                policy_for_spiking_mode("ttfs_cycle_based", schedule).calibration_forward()
                is _ttfs_segment_calibration_forward
            )

    def test_legacy_entry_point_matches_policy(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            calibration_forward_for_mode,
        )

        for mode in ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert (
                calibration_forward_for_mode(mode)
                is policy_for_spiking_mode(mode).calibration_forward()
            )

    def test_unsupported_mode_still_fails_loud(self):
        from mimarsinan.mapping.support.neg_shift_bias import (
            calibration_forward_for_mode,
        )

        with pytest.raises(NotImplementedError, match="rate"):
            calibration_forward_for_mode("rate")


# ── valid_backends ───────────────────────────────────────────────────────────

class TestValidBackends:
    ALL = ("hcm", "nevresim", "unified", "hybrid", "sanafe", "lava", "loihi", "training")

    def test_lif_valid_everywhere(self):
        assert policy_for_spiking_mode("lif").valid_backends(self.ALL) == self.ALL

    def test_ttfs_excludes_lif_only_backends(self):
        got = policy_for_spiking_mode("ttfs").valid_backends(self.ALL)
        assert "loihi" not in got and "lava" not in got and "training" not in got
        assert "sanafe" in got and "nevresim" in got

    def test_ttfs_cycle_excludes_lif_only_backends(self):
        got = policy_for_spiking_mode("ttfs_cycle_based", "cascaded").valid_backends(
            self.ALL
        )
        assert set(got) == {"hcm", "nevresim", "unified", "hybrid", "sanafe"}


# ── does_conversion_health_calibration (E3 CalibrationPipeline key) ───────────

class TestConversionHealthCalibration:
    """The (firing × sync) cell that opts into conversion-health calibration.

    Only the cascaded fire-once-latch cycle does today (its deployed decode has the
    depth-attenuation / distribution gap the steps correct); every other cell is
    inert, so the E3 ``CalibrationPipeline`` stays byte-identically off for them.
    """

    CASES = [
        ("lif", None, False),
        ("rate", None, False),
        ("ttfs", "cascaded", False),
        ("ttfs", "synchronized", False),
        ("ttfs_quantized", "cascaded", False),
        ("ttfs_cycle_based", "synchronized", False),
        ("ttfs_cycle_based", "cascaded", True),
        ("ttfs_cycle_based", None, True),  # default cascaded
    ]

    @pytest.mark.parametrize("mode,schedule,expected", CASES)
    def test_does_conversion_health_calibration(self, mode, schedule, expected):
        policy = policy_for_spiking_mode(mode, schedule)
        assert policy.does_conversion_health_calibration is expected

    def test_only_cascade_policy_opts_in(self):
        # The cascade policy is the sole opt-in; everything else inherits the
        # inert base default.
        assert TtfsCascadeModePolicy("ttfs_cycle_based", "cascaded") \
            .does_conversion_health_calibration is True
        for policy in (
            LifModePolicy("lif"),
            TtfsAnalyticalModePolicy("ttfs"),
            TtfsSyncCycleModePolicy("ttfs_cycle_based", "synchronized"),
        ):
            assert policy.does_conversion_health_calibration is False


# ── requires_ttfs_firing ──────────────────────────────────────────────────────

class TestRequiresTtfsFiring:
    def test_lif_does_not_require_ttfs(self):
        assert policy_for_spiking_mode("lif").requires_ttfs_firing is False

    def test_ttfs_family_requires_ttfs(self):
        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert policy_for_spiking_mode(mode).requires_ttfs_firing is True


# ── backend capability: the policy is the SSOT (spiking_semantics is internals) ─

class TestBackendCapability:
    ALL = ("hcm", "nevresim", "unified", "hybrid", "sanafe", "lava", "loihi", "training")
    MODES = ("lif", "rate", "ttfs", "ttfs_quantized", "ttfs_cycle_based")

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("backend", ALL)
    def test_supports_backend_matches_legacy_predicate(self, mode, backend):
        from mimarsinan.chip_simulation.spiking_semantics import supports_spiking_mode

        assert (
            policy_for_spiking_mode(mode).supports_backend(backend)
            is supports_spiking_mode(backend, mode)
        )

    def test_supports_backend_consistent_with_valid_backends(self):
        for mode in self.MODES:
            policy = policy_for_spiking_mode(mode)
            assert policy.valid_backends(self.ALL) == tuple(
                b for b in self.ALL if policy.supports_backend(b)
            )

    def test_require_backend_supported_passes_for_supported(self):
        # lif is valid on every backend → no raise.
        for backend in self.ALL:
            policy_for_spiking_mode("lif").require_backend_supported(
                backend=backend, context="ctx"
            )

    def test_require_backend_supported_raises_with_legacy_message(self):
        from mimarsinan.chip_simulation.spiking_semantics import (
            require_spiking_mode_supported,
        )

        # Byte-identical error to the function the call-sites used to call.
        with pytest.raises(ValueError) as policy_exc:
            policy_for_spiking_mode("ttfs").require_backend_supported(
                backend="loihi", context="ctx"
            )
        with pytest.raises(ValueError) as legacy_exc:
            require_spiking_mode_supported("ttfs", backend="loihi", context="ctx")
        assert str(policy_exc.value) == str(legacy_exc.value)

    def test_require_backend_supported_schedule_irrelevant(self):
        # Schedule never changes backend capability (only the mode does).
        for schedule in ("cascaded", "synchronized"):
            policy = policy_for_spiking_mode("ttfs_cycle_based", schedule)
            assert policy.supports_backend("lava") is False
            assert policy.supports_backend("nevresim") is True
