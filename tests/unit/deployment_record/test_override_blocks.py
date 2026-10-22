"""The one override value: named blocks merged constant by constant, a collision refused by name."""

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.resolve import (
    OverrideBlockConflictError,
    merge_override_blocks,
    resolve_platform_physics,
)

from unit.deployment_record.g3_fixtures import constant

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
