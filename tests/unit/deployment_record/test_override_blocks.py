"""The one override value: named blocks merged constant by constant, a collision refused by name."""

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import UNJUDGED, price_absolute
from mimarsinan.deployment_record.platform_physics.host_calibration import calibration_diagnostic
from mimarsinan.deployment_record.platform_physics.hosts.admission import HostRowAdmissionError
from mimarsinan.deployment_record.platform_physics.resolve import (
    OverrideBlockConflictError,
    merge_override_blocks,
    resolve_platform_physics,
)

from unit.deployment_record.g3_fixtures import constant, quantities
from unit.deployment_record.registered_hosts import declared_rows, registered_fixture
from unit.deployment_record.load_census_fixtures import law_energy, written_program

_BACKING_STORE = {"e_dma_per_byte": constant(31.25, unit="pJ", note="HBM2 I/O, jouppi2021ten Table 2 p.3 (test)")}
#: [U-HOST2] A test host's own description (test numbers), registered through the registry's contract around each
#: test; a run's host block declares its rows (the admission rule admits a value only as a description's row).
_HOST_NAME = "test_override_blocks_host"
_HOST_ROWS = {
    "host_macs_per_s": {"nominal": 12.0, "unit": "G/s", "evidence_kind": "estimated",
                        "note": "a test host's rate (test)"},
    "t_host_op_overhead": {"nominal": 40.0, "unit": "us", "evidence_kind": "estimated",
                           "note": "a test host's dispatch (test)"},
    "p_host": constant(280.0, unit="W", low=15.0, note="package TDP high end, bound (test)"),
}
_HOST = declared_rows(_HOST_NAME, _HOST_ROWS)
_host_description = registered_fixture(_HOST_NAME, _HOST_ROWS)
_CL0_P_HOST = {"low": 4.98, "nominal": 37.34, "high": 280.0, "unit": "W", "evidence_kind": "published",
               "citation": "edge-host running power, Table 1 p.2 (low end); CPU default TDP (high end) (test)",
               "note": "low end SECONDARY; high end a datasheet bound (test)"}
#: [U-HOST2] CL0's numbers as a test host's own description (test numbers, never the calibration's rows).
_CL0_NAME = "test_cl0_shaped_host"
_CL0_ROWS = {"host_macs_per_s": {"nominal": 2.0, "unit": "G/s", "evidence_kind": "estimated", "note": "test rate"},
             "t_host_op_overhead": {"low": 10.0, "nominal": 20.0, "high": 40.0, "unit": "us",
                                    "evidence_kind": "estimated", "note": "test dispatch"},
             "p_host": _CL0_P_HOST}
_cl0_description = registered_fixture(_CL0_NAME, _CL0_ROWS)


class TestTheMerge:
    def test_two_blocks_merge_into_one_value(self):
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": _HOST})
        assert set(merged) == {"e_dma_per_byte", "host_macs_per_s", "t_host_op_overhead", "p_host"}
        assert merged["e_dma_per_byte"]["note"].startswith("[backing_store] ")
        assert merged["p_host"]["note"].startswith("[host] ")
        assert merged["host_macs_per_s"]["evidence_kind"] == "estimated"

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
    declared p_host band (low end published SECONDARY, high end a datasheet bound). [U-HOST] That block is the
    development machine's (a labelled diagnostic): refused by name where it is loaded; the same rows declared as an
    identified host's price beside the backing store as before."""

    _P_HOST = _CL0_P_HOST
    _NAME, _ROWS = _CL0_NAME, _CL0_ROWS

    def _cl0(self):
        return calibration_diagnostic(macs_per_s=2.0e9, op_overhead_s=(1e-5, 2e-5, 4e-5), p_host_w=None,
                                      identity="h", declared={"p_host": self._P_HOST})

    def _host(self):
        return declared_rows(self._NAME, self._ROWS)

    def test_cl0s_block_refuses_by_name_beside_a_backing_store(self):
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": self._cl0()})
        with pytest.raises(HostRowAdmissionError, match="identified host description"):
            resolve_platform_physics("generic_estimated_22nm", merged)

    def test_cl0s_block_under_a_described_hosts_name_refuses_by_name(self):
        """[U-HOST2] The calibration's rows stamped with a read description's identity (measured declared
        estimated): the rule judges the value, and no description declares these rows."""
        from mimarsinan.deployment_record.platform_physics.hosts.description import (
            host_description_block, host_descriptions)

        shipped = [name for name in host_descriptions() if name not in (self._NAME, _HOST_NAME)][0]
        identity = host_description_block(shipped)["p_host"]["host"]
        stamped = {key: {**row, "host": identity, "evidence_kind": "estimated", "note": row.get("note") or "test"}
                   for key, row in self._cl0().items()}
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": stamped})
        with pytest.raises(HostRowAdmissionError, match="not admitted"):
            resolve_platform_physics("generic_estimated_22nm", merged)

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
        assert physics.constants["host_macs_per_s"].evidence_kind == "estimated"

    def test_a_point_priced_with_both_blocks_reads_both(self):
        merged = merge_override_blocks({"backing_store": _BACKING_STORE, "host": self._host()})
        physics = resolve_platform_physics("generic_estimated_22nm", merged)
        loads = written_program((2048,), (2048,))  # [U-ENERGY] the program load's census: 4096 B per round
        pricing = price_absolute(quantities(
            synaptic_events=1e6, host_macs=0, compute_op_count=10, cores_physical=4, latency_steps=4,
            timesteps=1, carry_in_bytes=1000), physics, geometry=UNJUDGED, loads=loads)
        terms = {t.name: t for t in pricing.terms}
        assert "energy_per_inference_mj" in terms, [r.reason for r in pricing.refusals]
        host = terms["energy_host_mj"].band
        assert host.low == pytest.approx(4.98 * 10 * 1e-5 * 1e3, rel=1e-12)
        assert host.high == pytest.approx(280.0 * 10 * 4e-5 * 1e3, rel=1e-12)
        assert "[backing_store]" in terms["energy_per_inference_mj"].source
        # [U-ENERGY] the one load law reads the block's e_dma_per_byte for the program's 4096 B
        commit = law_energy(loads.period, physics, without=("e_dma_per_byte",))
        for corner in ("low", "nominal", "high"):
            dynamic = getattr(terms["energy_programming_mj"].band, corner) - getattr(
                terms["energy_static_programming_mj"].band, corner)
            assert dynamic - getattr(commit, corner) == pytest.approx(4096 * 31.25e-12 * 1e3, rel=1e-9), corner
