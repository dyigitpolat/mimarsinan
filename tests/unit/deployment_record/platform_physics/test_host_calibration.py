"""[H3b] The calibration is honest: measured where measured, absent where not. [U-HOST] It is the development
machine's labelled diagnostic, never a host block: host rows are admitted only from an identified host description."""

from __future__ import annotations

import json

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
import mimarsinan.deployment_record.platform_physics.host_calibration as host_calibration
from mimarsinan.deployment_record.platform_physics import apply_overrides, profile_from_dict
from mimarsinan.deployment_record.platform_physics.host_calibration import (
    DIAGNOSTIC_LABEL,
    HostRowConflictError,
    calibration_diagnostic,
    host_block,
    load_host_block,
    machine_identity,
    macs_of,
    run_calibration,
)
from mimarsinan.deployment_record.platform_physics.hosts.admission import HostRowAdmissionError
from mimarsinan.deployment_record.platform_physics.registry import (
    get_platform_physics,
)

from unit.deployment_record.g3_fixtures import quantities
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block, host_descriptions

_ROWS = {"host_macs_per_s", "t_host_op_overhead", "p_host"}


class TestTheDiagnostic:
    def test_the_rate_is_declared_in_the_vocabularys_unit(self):
        rows = calibration_diagnostic(
            macs_per_s=12.5e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=None, identity="testhost (x86_64)")
        rate = rows["host_macs_per_s"]
        assert rate["nominal"] == 12.5 and rate["unit"] == "G/s"
        assert rate["evidence_kind"] == "measured"
        assert "testhost" in rate["note"] and "diagnostic" in rate["note"]

    def test_no_rapl_means_no_p_host_claim(self):
        """A power figure nobody measured is exactly the silent default this
        program exists to kill — absence, with the basis saying why."""
        rows = calibration_diagnostic(
            macs_per_s=1e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=None, identity="h")
        assert "p_host" not in rows

    def test_a_measured_p_host_rides_in_as_measured(self):
        rows = calibration_diagnostic(
            macs_per_s=1e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=17.5, identity="h")
        assert rows["p_host"]["nominal"] == 17.5
        assert rows["p_host"]["evidence_kind"] == "measured"

    def test_the_diagnostic_never_applies_onto_a_profile(self):
        """[U-HOST] The machine that ran the toolchain is not the deployment host: its rows refuse by name."""
        rows = calibration_diagnostic(
            macs_per_s=40e9, op_overhead_s=(1e-5, 2e-5, 4e-5),
            p_host_w=20.0, identity="h")
        with pytest.raises(HostRowAdmissionError, match="identified host description") as refused:
            apply_overrides(get_platform_physics("loihi"), rows)
        assert isinstance(refused.value, ToolchainRefusal)
        assert all(key in str(refused.value) for key in _ROWS)

    def test_macs_of_is_the_matmul_count(self):
        assert macs_of((2, 784, 500)) == 2 * 784 * 500


# --- CL0: the calibration with a declared p_host band ------------------------------------------------

#: The shape of CL0's declaration: one band over two hosts, each end with its own evidence.
_P_HOST_BAND = {"p_host": {
    "low": 5.0, "nominal": 37.4, "high": 280.0, "unit": "W", "evidence_kind": "published",
    "citation": "an edge-host benchmark, Table 1 p.2 (low end); a CPU datasheet, default TDP (high end) (test)",
    "note": "low end SECONDARY (another host's active power); high end a datasheet bound (test)",
}}
_MEASURED = dict(macs_per_s=1e9, op_overhead_s=(1e-5, 2e-5, 4e-5), identity="h")


class TestDeclaredRows:
    def test_a_declared_p_host_band_rides_in_verbatim(self):
        rows = calibration_diagnostic(**_MEASURED, p_host_w=None, declared=_P_HOST_BAND)
        assert rows["p_host"] == _P_HOST_BAND["p_host"]
        assert set(rows) == _ROWS

    def test_the_merge_leaves_the_declaration_untouched(self):
        rows = calibration_diagnostic(**_MEASURED, p_host_w=None, declared=_P_HOST_BAND)
        rows["p_host"]["note"] = "changed"
        assert _P_HOST_BAND["p_host"]["note"].startswith("low end SECONDARY")

    def test_a_declared_row_the_calibration_measured_refuses_by_name(self):
        declared = {"host_macs_per_s": {"nominal": 3.0, "unit": "G/s", "evidence_kind": "estimated", "note": "n"}}
        with pytest.raises(HostRowConflictError) as info:
            calibration_diagnostic(**_MEASURED, p_host_w=None, declared=declared)
        assert isinstance(info.value, ToolchainRefusal)
        assert "host_macs_per_s" in str(info.value)

    def test_a_declared_p_host_beside_a_rapl_measurement_refuses_by_name(self):
        """Two powers for one machine, one measured and one declared: the run names both instead of picking."""
        with pytest.raises(HostRowConflictError, match="p_host"):
            calibration_diagnostic(**_MEASURED, p_host_w=17.5, declared=_P_HOST_BAND)

    def test_a_declared_row_without_its_evidence_refuses(self):
        bad = {"p_host": dict(_P_HOST_BAND["p_host"], citation="")}
        with pytest.raises(ValueError, match="citation"):
            calibration_diagnostic(**_MEASURED, p_host_w=None, declared=bad)

    def test_a_declared_band_out_of_order_refuses(self):
        bad = {"p_host": dict(_P_HOST_BAND["p_host"], nominal=300.0)}
        with pytest.raises(ValueError, match="low <= nominal <= high"):
            calibration_diagnostic(**_MEASURED, p_host_w=None, declared=bad)

    def test_a_declared_key_outside_the_vocabulary_refuses(self):
        with pytest.raises(KeyError, match="not a declarable physics constant"):
            calibration_diagnostic(**_MEASURED, p_host_w=None, declared={"p_hots": _P_HOST_BAND["p_host"]})


class TestNoHostBlock:
    """[U-HOST] A calibration artifact is never a host block: ``host_block`` and ``load_host_block`` refuse it by
    the admission rule's name (they stay, so a caller gets the refusal rather than an import error)."""

    def _artifact(self, declared=_P_HOST_BAND):
        return {"machine": "h", "diagnostic": {"label": DIAGNOSTIC_LABEL, "rows": calibration_diagnostic(
            **_MEASURED, p_host_w=None, declared=declared)}}

    def test_the_artifact_refuses_as_a_host_block_by_name(self):
        with pytest.raises(HostRowAdmissionError, match="labelled diagnostic") as info:
            host_block(self._artifact())
        assert isinstance(info.value, ToolchainRefusal)
        assert "identified host description" in str(info.value)

    def test_an_artifact_file_refuses_as_a_host_block_by_name(self, tmp_path):
        path = tmp_path / "host_calibration.json"
        path.write_text(json.dumps(self._artifact()))
        with pytest.raises(HostRowAdmissionError, match="labelled diagnostic"):
            load_host_block(str(path))

    def test_the_rows_under_a_read_hosts_name_refuse_by_name(self):
        """[U-HOST2] The same rows declared under a read description's name (measured declared estimated, as a
        relabel would): the rule judges the value, and no description declares these rows, so they refuse by name
        and price nothing (the calibration is the development machine's, never a host price)."""
        chip = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T", "description_file": "t.md",
            "validity": {"measurement_kind": "mixed"},
            "constants": {"e_synaptic_event_total": {"nominal": 10.0, "unit": "pJ",
                                                     "evidence_kind": "estimated", "note": "n"}}})
        identity = host_description_block(host_descriptions()[0])["p_host"]["host"]
        rows = {key: {**row, "host": identity, "evidence_kind": "estimated", "note": row.get("note") or "a row"}
                for key, row in self._artifact()["diagnostic"]["rows"].items()}
        with pytest.raises(HostRowAdmissionError, match="not admitted") as info:
            apply_overrides(chip, rows)
        assert all(key in str(info.value) for key in rows)


class TestTheRun:
    def _measured(self, monkeypatch):
        monkeypatch.setattr(host_calibration, "measure_host_mac_rates",
                            lambda **kw: {(1, 784, 500): 2.0e9, (1, 500, 100): 1.5e9})
        monkeypatch.setattr(host_calibration, "measure_host_op_overhead_s", lambda: (1e-5, 2e-5, 3e-5))
        monkeypatch.setattr(host_calibration, "measure_p_host_w", lambda: None)

    def test_the_artifact_is_a_labelled_diagnostic_with_the_declared_band_and_the_machine(self, monkeypatch):
        self._measured(monkeypatch)
        artifact = run_calibration(declared=_P_HOST_BAND)
        assert artifact["host_macs_per_s"] == 2.0e9
        assert artifact["host_macs_per_s_by_shape"] == {"1x784x500": 2.0e9, "1x500x100": 1.5e9}
        assert artifact["threads"] >= 1 and artifact["cpu_affinity"]
        assert artifact["p_host_basis"].startswith("declared")
        assert "platform_physics_overrides" not in artifact
        assert artifact["diagnostic"]["label"] == DIAGNOSTIC_LABEL and "never" in DIAGNOSTIC_LABEL
        rows = artifact["diagnostic"]["rows"]
        assert rows["p_host"] == _P_HOST_BAND["p_host"]
        assert "thread" in rows["host_macs_per_s"]["note"]

    def test_without_rapl_or_a_declaration_the_artifact_says_why_p_host_is_absent(self, monkeypatch):
        self._measured(monkeypatch)
        artifact = run_calibration()
        assert artifact["p_host_w"] is None and "p_host" not in artifact["diagnostic"]["rows"]
        assert artifact["p_host_basis"].startswith("unavailable")

    def test_the_machine_identity_names_the_cpu_model(self, monkeypatch):
        monkeypatch.setattr(host_calibration, "cpu_model", lambda: "Test CPU 9000")
        assert "Test CPU 9000" in machine_identity()
