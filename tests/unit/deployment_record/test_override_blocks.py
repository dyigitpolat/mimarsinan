"""The one override value: named blocks merged constant by constant, a collision refused by name."""

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import UNJUDGED, price_absolute
from mimarsinan.deployment_record.platform_physics.host_calibration import calibration_overrides, host_block
from mimarsinan.deployment_record.platform_physics.resolve import (
    OverrideBlockConflictError,
    merge_override_blocks,
    resolve_platform_physics,
)

from unit.deployment_record.g3_fixtures import constant, quantities

_BACKING_STORE = {"e_dma_per_byte": constant(31.25, unit="pJ", note="HBM2 I/O, jouppi2021ten Table 2 p.3 (test)")}
_HOST = {
    "host_macs_per_s": {"nominal": 12.0, "unit": "G/s", "evidence_kind": "measured",
                        "note": "calibrate_host.py on this machine (test)"},
    "t_host_op_overhead": {"nominal": 40.0, "unit": "us", "evidence_kind": "measured",
                           "note": "calibrate_host.py on this machine (test)"},
    "p_host": constant(280.0, unit="W", low=15.0, note="package TDP high end, bound (test)"),
}


class TestTheMerge:
    def test_two_blocks_merge_into_one_value(self):
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": _HOST})
        assert set(merged) == {"e_dma_per_byte", "host_macs_per_s", "t_host_op_overhead", "p_host"}
        assert merged["e_dma_per_byte"]["note"].startswith("[backing_store] ")
        assert merged["p_host"]["note"].startswith("[host] ")
        assert merged["host_macs_per_s"]["evidence_kind"] == "measured"

    def test_the_22nm_point_resolves_with_both_sets_of_constants(self):
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": _HOST})
        physics = resolve_platform_physics("generic_estimated_22nm", merged)
        assert physics.band("e_dma_per_byte").nominal == pytest.approx(31.25e-12)
        assert physics.band("t_host_op_overhead").nominal == pytest.approx(40e-6)
        assert set(physics.overridden_keys()) == set(merged)

    def test_a_constant_in_two_blocks_refuses_by_name(self):
        with pytest.raises(OverrideBlockConflictError) as info:
            merge_override_blocks({"backing_store": _BACKING_STORE, "variant": _BACKING_STORE})
        assert isinstance(info.value, ToolchainRefusal)
        assert "e_dma_per_byte" in str(info.value) and "'variant'" in str(info.value)

    def test_the_merge_leaves_its_inputs_untouched(self):
        merge_override_blocks({"host": _HOST})
        assert _HOST["p_host"]["note"] == "package TDP high end, bound (test)"


class TestTheHostBlockOfCl0:
    """G3 acceptance 16 with CL0's host block as the calibration builds it: measured rate and dispatch, the
    declared p_host band (low end published SECONDARY, high end a datasheet bound)."""

    _P_HOST = {"low": 4.98, "nominal": 37.34, "high": 280.0, "unit": "W", "evidence_kind": "published",
               "citation": "edge-host running power, Table 1 p.2 (low end); CPU default TDP (high end) (test)",
               "note": "low end SECONDARY; high end a datasheet bound (test)"}

    def _host(self):
        return host_block({"platform_physics_overrides": calibration_overrides(
            macs_per_s=2.0e9, op_overhead_s=(1e-5, 2e-5, 4e-5), p_host_w=None, identity="h",
            declared={"p_host": self._P_HOST})})

    def test_the_22nm_point_resolves_with_the_host_block_and_a_backing_store(self):
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": self._host()})
        physics = resolve_platform_physics("generic_estimated_22nm", merged)
        assert set(physics.overridden_keys()) == {
            "e_dma_per_byte", "host_macs_per_s", "t_host_op_overhead", "p_host"}
        assert physics.band("e_dma_per_byte").nominal == pytest.approx(31.25e-12)
        assert physics.band("host_macs_per_s").nominal == pytest.approx(2.0e9)
        assert physics.band("t_host_op_overhead").high == pytest.approx(4e-5)
        p_host = physics.band("p_host")
        assert (p_host.low, p_host.nominal, p_host.high) == (4.98, 37.34, 280.0)
        assert physics.constants["p_host"].evidence_kind == "published"
        assert physics.constants["host_macs_per_s"].evidence_kind == "measured"

    def test_a_point_priced_with_both_blocks_reads_both(self):
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": self._host()})
        physics = resolve_platform_physics("generic_estimated_22nm", merged)
        pricing = price_absolute(quantities(
            synaptic_events=1e6, host_macs=0, compute_op_count=10, cores_physical=4, latency_steps=4,
            timesteps=1, carry_in_bytes=1000, reprogrammed_bytes=4096), physics, geometry=UNJUDGED)
        terms = {t.name: t for t in pricing.terms}
        assert "energy_per_inference_mj" in terms, [r.reason for r in pricing.refusals]
        host = terms["energy_host_mj"].band
        assert host.low == pytest.approx(4.98 * 10 * 1e-5 * 1e3, rel=1e-12)
        assert host.high == pytest.approx(280.0 * 10 * 4e-5 * 1e3, rel=1e-12)
        assert "[backing_store]" in terms["energy_per_inference_mj"].source
        assert "[backing_store]" in terms["energy_programming_mj"].source
