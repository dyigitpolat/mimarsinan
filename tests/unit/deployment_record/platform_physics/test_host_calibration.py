"""[H3b] The calibration block is honest: measured where measured, absent where not."""

from __future__ import annotations

from mimarsinan.deployment_record.platform_physics import apply_overrides
from mimarsinan.deployment_record.platform_physics.host_calibration import (
    calibration_overrides,
    macs_of,
)
from mimarsinan.deployment_record.platform_physics.registry import (
    get_platform_physics,
)


class TestTheBlock:
    def test_the_rate_is_declared_in_the_vocabularys_unit(self):
        block = calibration_overrides(
            macs_per_s=12.5e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=None, identity="testhost (x86_64)")
        rate = block["host_macs_per_s"]
        assert rate["nominal"] == 12.5 and rate["unit"] == "G/s"
        assert rate["evidence_kind"] == "measured"
        assert "testhost" in rate["note"]

    def test_no_rapl_means_no_p_host_claim(self):
        """A power figure nobody measured is exactly the silent default this
        program exists to kill — absence, with the basis saying why."""
        block = calibration_overrides(
            macs_per_s=1e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=None, identity="h")
        assert "p_host" not in block

    def test_a_measured_p_host_rides_in_as_measured(self):
        block = calibration_overrides(
            macs_per_s=1e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=17.5, identity="h")
        assert block["p_host"]["nominal"] == 17.5
        assert block["p_host"]["evidence_kind"] == "measured"

    def test_the_block_applies_onto_a_real_profile(self):
        """End of the chain: the override machinery accepts the block and the
        resulting profile prices host terms from measured evidence."""
        block = calibration_overrides(
            macs_per_s=40e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=20.0, identity="h")
        physics = apply_overrides(get_platform_physics("loihi"), block)
        assert physics.has("host_macs_per_s") and physics.has("p_host")
        assert "measured" in physics.band("host_macs_per_s").basis

    def test_macs_of_is_the_matmul_count(self):
        assert macs_of((2, 784, 500)) == 2 * 784 * 500
