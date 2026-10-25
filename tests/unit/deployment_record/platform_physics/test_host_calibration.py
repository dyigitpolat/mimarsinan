"""[H3b] The calibration block is honest: measured where measured, absent where not."""

from __future__ import annotations

import json

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
import mimarsinan.deployment_record.platform_physics.host_calibration as host_calibration
from mimarsinan.deployment_record.platform_physics import apply_overrides, profile_from_dict
from mimarsinan.deployment_record.platform_physics.host_calibration import (
    HOST_BLOCK_KEYS,
    HostBlockIncompleteError,
    HostRowConflictError,
    calibration_overrides,
    host_block,
    load_host_block,
    machine_identity,
    macs_of,
    run_calibration,
)
from mimarsinan.deployment_record.platform_physics.registry import (
    get_platform_physics,
)

from unit.deployment_record.g3_fixtures import quantities


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


# --- CL0: the named host block (measured rate and dispatch, a declared p_host band) -----------------

#: The shape of CL0's declaration: one band over two hosts, each end with its own evidence.
_P_HOST_BAND = {"p_host": {
    "low": 5.0, "nominal": 37.4, "high": 280.0, "unit": "W", "evidence_kind": "published",
    "citation": "an edge-host benchmark, Table 1 p.2 (low end); a CPU datasheet, default TDP (high end) (test)",
    "note": "low end SECONDARY (another host's active power); high end a datasheet bound (test)",
}}
_MEASURED = dict(macs_per_s=1e9, op_overhead_s=(1e-5, 2e-5, 4e-5), identity="h")


class TestDeclaredRows:
    def test_a_declared_p_host_band_rides_in_verbatim(self):
        block = calibration_overrides(**_MEASURED, p_host_w=None, declared=_P_HOST_BAND)
        assert block["p_host"] == _P_HOST_BAND["p_host"]
        assert set(block) == set(HOST_BLOCK_KEYS)

    def test_the_merge_leaves_the_declaration_untouched(self):
        block = calibration_overrides(**_MEASURED, p_host_w=None, declared=_P_HOST_BAND)
        block["p_host"]["note"] = "changed"
        assert _P_HOST_BAND["p_host"]["note"].startswith("low end SECONDARY")

    def test_a_declared_row_the_calibration_measured_refuses_by_name(self):
        declared = {"host_macs_per_s": {"nominal": 3.0, "unit": "G/s", "evidence_kind": "estimated", "note": "n"}}
        with pytest.raises(HostRowConflictError) as info:
            calibration_overrides(**_MEASURED, p_host_w=None, declared=declared)
        assert isinstance(info.value, ToolchainRefusal)
        assert "host_macs_per_s" in str(info.value)

    def test_a_declared_p_host_beside_a_rapl_measurement_refuses_by_name(self):
        """Two powers for one host, one measured and one declared: the run names both instead of picking."""
        with pytest.raises(HostRowConflictError, match="p_host"):
            calibration_overrides(**_MEASURED, p_host_w=17.5, declared=_P_HOST_BAND)

    def test_a_declared_row_without_its_evidence_refuses(self):
        bad = {"p_host": dict(_P_HOST_BAND["p_host"], citation="")}
        with pytest.raises(ValueError, match="citation"):
            calibration_overrides(**_MEASURED, p_host_w=None, declared=bad)

    def test_a_declared_band_out_of_order_refuses(self):
        bad = {"p_host": dict(_P_HOST_BAND["p_host"], nominal=300.0)}
        with pytest.raises(ValueError, match="low <= nominal <= high"):
            calibration_overrides(**_MEASURED, p_host_w=None, declared=bad)

    def test_a_declared_key_outside_the_vocabulary_refuses(self):
        with pytest.raises(KeyError, match="not a declarable physics constant"):
            calibration_overrides(**_MEASURED, p_host_w=None, declared={"p_hots": _P_HOST_BAND["p_host"]})


class TestTheHostBlock:
    def _artifact(self, declared=_P_HOST_BAND):
        return {"platform_physics_overrides": calibration_overrides(**_MEASURED, p_host_w=None,
                                                                    declared=declared)}

    def test_the_block_of_an_artifact_carries_every_host_row(self):
        block = host_block(self._artifact())
        assert set(block) == {"host_macs_per_s", "t_host_op_overhead", "p_host"}
        assert block["host_macs_per_s"]["evidence_kind"] == "measured"
        assert block["t_host_op_overhead"]["evidence_kind"] == "measured"
        assert block["p_host"]["high"] == 280.0

    def test_a_block_without_p_host_refuses_by_name(self):
        """Without p_host every host-carrying energy headline refuses; the block says so where it is read."""
        with pytest.raises(HostBlockIncompleteError) as info:
            host_block(self._artifact(declared=None))
        assert isinstance(info.value, ToolchainRefusal)
        assert "p_host" in str(info.value)

    def test_a_block_with_a_row_beyond_the_host_rows_refuses_by_name(self):
        """The host block is the host's: a chip constant in it would override the chip by the host's name."""
        artifact = self._artifact()
        artifact["platform_physics_overrides"]["e_mac"] = {"nominal": 1.0, "unit": "pJ",
                                                           "evidence_kind": "estimated", "note": "n"}
        with pytest.raises(HostBlockIncompleteError, match="e_mac"):
            host_block(artifact)

    def test_the_block_loads_from_its_artifact_file(self, tmp_path):
        path = tmp_path / "host_calibration.json"
        path.write_text(json.dumps(self._artifact()))
        assert load_host_block(str(path)) == host_block(self._artifact())

    def test_a_malformed_row_in_a_loaded_artifact_refuses(self, tmp_path):
        artifact = self._artifact()
        artifact["platform_physics_overrides"]["p_host"]["low"] = 400.0
        path = tmp_path / "host_calibration.json"
        path.write_text(json.dumps(artifact))
        with pytest.raises(ValueError, match="low <= nominal <= high"):
            load_host_block(str(path))

    def test_the_block_prices_host_energy_at_both_band_ends(self):
        """The p_host band reaches the priced host energy end by end (low power x low time, high x high)."""
        chip = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T", "description_file": "t.md",
            "validity": {"measurement_kind": "mixed"},
            "constants": {"e_synaptic_event_total": {"nominal": 10.0, "unit": "pJ",
                                                     "evidence_kind": "estimated", "note": "n"}}})
        physics = apply_overrides(chip, host_block(self._artifact()))
        pricing = price_absolute(quantities(host_macs=0, compute_op_count=10, synaptic_events=1e6), physics)
        host = {t.name: t for t in pricing.terms}["energy_host_mj"].band
        assert host.low == pytest.approx(5.0 * 10 * 1e-5 * 1e3, rel=1e-12)
        assert host.high == pytest.approx(280.0 * 10 * 4e-5 * 1e3, rel=1e-12)


class TestTheRun:
    def _measured(self, monkeypatch):
        monkeypatch.setattr(host_calibration, "measure_host_mac_rates",
                            lambda **kw: {(1, 784, 500): 2.0e9, (1, 500, 100): 1.5e9})
        monkeypatch.setattr(host_calibration, "measure_host_op_overhead_s", lambda: (1e-5, 2e-5, 3e-5))
        monkeypatch.setattr(host_calibration, "measure_p_host_w", lambda: None)

    def test_the_artifact_carries_the_declared_band_and_the_machine(self, monkeypatch):
        self._measured(monkeypatch)
        artifact = run_calibration(declared=_P_HOST_BAND)
        assert artifact["host_macs_per_s"] == 2.0e9
        assert artifact["host_macs_per_s_by_shape"] == {"1x784x500": 2.0e9, "1x500x100": 1.5e9}
        assert artifact["threads"] >= 1 and artifact["cpu_affinity"]
        assert artifact["p_host_basis"].startswith("declared")
        block = host_block(artifact)
        assert block["p_host"] == _P_HOST_BAND["p_host"]
        assert "thread" in block["host_macs_per_s"]["note"]

    def test_without_rapl_or_a_declaration_the_artifact_says_why_p_host_is_absent(self, monkeypatch):
        self._measured(monkeypatch)
        artifact = run_calibration()
        assert artifact["p_host_w"] is None and "p_host" not in artifact["platform_physics_overrides"]
        assert artifact["p_host_basis"].startswith("unavailable")

    def test_the_machine_identity_names_the_cpu_model(self, monkeypatch):
        monkeypatch.setattr(host_calibration, "cpu_model", lambda: "Test CPU 9000")
        assert "Test CPU 9000" in machine_identity()
