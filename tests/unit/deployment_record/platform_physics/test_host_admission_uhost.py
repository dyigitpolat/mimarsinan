"""[U-HOST] (a) One admission rule at the platform loader for the HOST group.

Every host row (``hosts.rows``) is admitted only from an identified host description: its host named (a shipped
description and one of its modes), its evidence published, datasheet, derived or estimated, never measured; host rows
of one physics name one host; anything else is refused by name, at each of the loader's entries (a profile's own rows,
a run's overrides, a sealed physics). No shipped profile declares a host row (the bare profiles' ``host_compute_rate``
rows are withdrawn). A calibration of the development machine (``scripts/calibrate_host.py``) is a labelled
diagnostic, never a host block. Tests T-A1 to T-A13 (the check's M-2: per host row an unidentified override refuses by
name; t-3 with server blocks on both planes; t-4 widened to the calibration's rows).
"""

from __future__ import annotations

import copy
import json

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import (
    apply_overrides, available_profiles, get_platform_physics, profile_from_dict,
)
from mimarsinan.deployment_record.platform_physics.constants import keys_in_group, spec_for
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HOST_IDENTITY_FIELD, host_description, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

from unit.deployment_record.g3_fixtures import quantities

HOST_ROWS = keys_in_group(HOST)
NAME = host_descriptions()[0]
IDENTITY = f"{NAME} {host_description(NAME).default_mode}"
#: A chip profile with no host rows of its own at any head (the rule judges the host rows a run adds to it).
CHIP = "neurram"
#: CL0's form of a server block: rate and dispatch measured on the development machine, a declared power band over
#: two hosts, no host named (``explore/cl0/host_block.json``'s shape, test numbers).
SERVER = {
    "t_host_op_overhead": {"low": 13.37, "nominal": 13.48, "high": 13.64, "unit": "us", "evidence_kind": "measured",
                           "note": "dispatch wall measured on the development machine (test)"},
    "host_macs_per_s": {"nominal": 22.0, "unit": "G/s", "evidence_kind": "measured",
                        "note": "rate measured on the development machine (test)"},
    "p_host": {"low": 4.98, "nominal": 37.34, "high": 280.0, "unit": "W", "evidence_kind": "published",
               "citation": "a band over two hosts (test)"},
}


def _row(key, **extra):
    """One host row of ``key`` in its display unit (estimated, a test note), without a host unless ``extra`` says."""
    return {"nominal": 1.0, "unit": spec_for(key).display_unit, "evidence_kind": "estimated", "note": "a test row",
            **extra}


def _mode_rows(mode):
    """[U-HOST2] The rows one mode of the description declares itself, as override rows naming that mode."""
    facts = host_description(NAME).modes[mode]
    return {key: {**dict(row), HOST_IDENTITY_FIELD: f"{NAME} {mode}"}
            for key, row in (facts.get("constants") or {}).items()}


def _refusal(call):
    with pytest.raises(ToolchainRefusal) as refused:
        call()
    return str(refused.value)


class TestEveryHostRow:
    @pytest.mark.parametrize("key", HOST_ROWS)
    def test_an_unidentified_override_refuses_by_name(self, key):
        """T-A1: the check's M-2 test, per HOST row (red at R5 for every row)."""
        why = _refusal(lambda: apply_overrides(get_platform_physics(CHIP), {key: _row(key)}))
        assert key in why and "identified host description" in why

    @pytest.mark.parametrize("key", HOST_ROWS)
    def test_an_identified_row_is_admitted_and_carries_its_host(self, key):
        """T-A2 [U-HOST2]: the named mode's own row (as its block declares it) is admitted and the resolved value
        carries the host; a row the description does not declare refuses by name (another value under the host's
        name was admitted at revision 1, the verify's M-1)."""
        block = host_description_block(NAME)
        if key not in block:
            why = _refusal(lambda: apply_overrides(get_platform_physics(CHIP), {
                key: _row(key, **{HOST_IDENTITY_FIELD: IDENTITY})}))
            assert key in why and "declares no" in why
            return
        physics = apply_overrides(get_platform_physics(CHIP), {key: block[key]})
        assert physics.constants[key].host == IDENTITY
        other = {**block[key], **{end: block[key][end] * 2.0 for end in ("low", "nominal", "high") if end in block[key]}}
        assert key in _refusal(lambda: apply_overrides(get_platform_physics(CHIP), {key: other}))

    @pytest.mark.parametrize("key", HOST_ROWS)
    def test_a_measured_row_refuses_by_name_even_when_it_names_a_host(self, key):
        """T-A3: never a measurement of the machine running the toolchain: the description's own value declared
        measured refuses by its class (a row the description does not declare refuses as undeclared)."""
        block = host_description_block(NAME)
        row = {**block[key], "evidence_kind": "measured", "note": "a measurement (test)"} if key in block else _row(
            key, **{HOST_IDENTITY_FIELD: IDENTITY, "evidence_kind": "measured"})
        why = _refusal(lambda: apply_overrides(get_platform_physics(CHIP), {key: row}))
        assert key in why and ("measured" in why if key in block else "declares no" in why)

    def test_a_host_no_description_ships_refuses_by_name(self):
        """T-A4: the identity names a shipped description and one of its modes."""
        for identity in ("no_such_host 15W", f"{NAME} no_such_mode", NAME):
            why = _refusal(lambda: apply_overrides(get_platform_physics(CHIP), {
                "p_host": _row("p_host", **{HOST_IDENTITY_FIELD: identity})}))
            assert "p_host" in why and repr(identity) in why

    def test_rows_of_two_hosts_refuse_by_name(self):
        """T-A5 (a guard): one physics, one host."""
        block = host_description_block(NAME)
        other = {key: {**row, HOST_IDENTITY_FIELD: "another_host 1W"} for key, row in block.items()}
        mixed = {**{k: v for k, v in block.items() if k.startswith("t_")},
                 **{k: v for k, v in other.items() if not k.startswith("t_")}}
        assert "another_host" in _refusal(lambda: resolve_platform_physics(CHIP, mixed))
        modes = sorted(host_description(NAME).modes)  # two modes of one shipped description are two hosts
        other = next(m for m in modes if f"{NAME} {m}" != IDENTITY)
        own = _mode_rows(other)  # [U-HOST2] each mode's own rows, so only the two hosts are refused
        two = {**{k: v for k, v in block.items() if k not in own}, **own}
        assert len(modes) > 1 and own and "one host" in _refusal(lambda: resolve_platform_physics(CHIP, two))


    def test_an_identified_block_replaces_another_hosts_rows_whole(self):
        """T-A14: a physics holding one host's rows takes another identified host's block whole (require_one_host),
        so the result names one host, never a mix the rule would refuse."""
        block = host_description_block(NAME)
        first = apply_overrides(get_platform_physics(CHIP), block)
        other_mode = next(m for m in sorted(host_description(NAME).modes) if f"{NAME} {m}" != IDENTITY)
        some = _mode_rows(other_mode)  # [U-HOST2] the other mode's own rows (its description's values)
        second = apply_overrides(first, some)
        hosts = {v.host for k, v in second.constants.items() if spec_for(k).group == HOST}
        assert hosts == {f"{NAME} {other_mode}"} and set(some) == {k for k in second.constants
                                                                   if spec_for(k).group == HOST}


class TestTheProfiles:
    @pytest.mark.parametrize("name", available_profiles())
    def test_no_shipped_profile_declares_a_host_row(self, name):
        """T-A6: the bare profiles' host_compute_rate rows withdrawn (red at R5 on the seven that carried it)."""
        physics = get_platform_physics(name)
        assert not [key for key in physics.constants if spec_for(key).group == HOST]

    def test_a_profile_declaring_a_host_row_refuses_by_name(self):
        """T-A7: a profile is no host description; its host row refuses at load."""
        data = {"format_version": 1, "name": "t", "display_name": "T", "description_file": "t.md",
                "validity": {"measurement_kind": "mixed"},
                "constants": {"e_mac": {"nominal": 1.0, "unit": "pJ", "evidence_kind": "estimated", "note": "n"},
                              "host_compute_rate": {"nominal": 1.0, "unit": "fraction", "evidence_kind": "derived",
                                                    "derivation": "identity (test)"}}}
        why = _refusal(lambda: profile_from_dict(data))
        assert "host_compute_rate" in why and "'t'" in why


class TestASealedPhysics:
    def _sealed(self):
        return resolve_platform_physics(CHIP, merge_override_blocks({"host": host_description_block(NAME)})).to_dict()

    def test_an_identified_physics_round_trips_with_its_host(self):
        """T-A9: the identity is sealed with each host row (and only there), so the sealed physics reads back."""
        payload = self._sealed()
        hosts = {key: row.get("host") for key, row in payload["constants"].items()}
        assert {k for k, h in hosts.items() if h} == {k for k in payload["constants"] if spec_for(k).group == HOST}
        assert set(h for h in hosts.values() if h) == {IDENTITY}
        assert PlatformPhysics.from_dict(payload).to_dict() == payload

    def test_a_sealed_unidentified_host_row_refuses_by_name(self):
        """T-A8: a sealed physics whose host row names no host (a pre-rule seal, a server row) refuses where it is
        read back (red at R5: R5 read any sealed row)."""
        payload = self._sealed()
        payload["constants"]["p_host"].pop("host", None)
        assert "p_host" in _refusal(lambda: PlatformPhysics.from_dict(payload))
        server = get_platform_physics(CHIP).to_dict()
        server["constants"]["host_macs_per_s"] = {"key": "host_macs_per_s", "low": 22.0, "nominal": 22.0,
                                                  "high": 22.0, "unit": "G/s", "evidence_kind": "measured",
                                                  "citation": "", "derivation": "", "note": "a server's rate",
                                                  "overridden": True}
        assert "host_macs_per_s" in _refusal(lambda: PlatformPhysics.from_dict(server))

    @pytest.mark.parametrize("name", available_profiles())
    def test_a_physics_without_host_rows_keeps_its_bytes(self, name):
        """T-A10 (a guard): no constant of a host-free physics gains a field."""
        payload = get_platform_physics(name).to_dict()
        assert not any("host" in row for row in payload["constants"].values())


class TestTheCalibration:
    def _artifact(self, monkeypatch):
        import mimarsinan.deployment_record.platform_physics.host_calibration as calibration

        monkeypatch.setattr(calibration, "measure_host_mac_rates", lambda **kw: {(1, 784, 500): 2.0e9})
        monkeypatch.setattr(calibration, "measure_host_op_overhead_s", lambda: (1e-5, 2e-5, 3e-5))
        monkeypatch.setattr(calibration, "measure_p_host_w", lambda: 20.0)
        return calibration.run_calibration()

    def test_the_artifact_is_a_labelled_diagnostic(self, monkeypatch):
        """T-A11: no override block; a label saying what the rows are and that they price nothing."""
        artifact = self._artifact(monkeypatch)
        assert "platform_physics_overrides" not in artifact
        assert "diagnostic" in artifact["diagnostic"]["label"] and "never" in artifact["diagnostic"]["label"]
        assert all(row["evidence_kind"] == "measured" for row in artifact["diagnostic"]["rows"].values())

    def test_the_artifact_refuses_as_a_host_block_by_name(self, monkeypatch, tmp_path):
        """T-A11: host_block and load_host_block (CL4's and CL5's readers) refuse it by the rule's name."""
        from mimarsinan.deployment_record.platform_physics.host_calibration import host_block, load_host_block

        artifact = self._artifact(monkeypatch)
        assert "identified host description" in _refusal(lambda: host_block(artifact))
        path = tmp_path / "host_calibration.json"
        path.write_text(json.dumps(artifact))
        assert "identified host description" in _refusal(lambda: load_host_block(str(path)))


class TestServerBlocksOnBothPlanes:
    @pytest.mark.parametrize("name", available_profiles())
    def test_t3_a_server_block_refuses_by_name_on_the_record_and_the_candidate_plane(self, name):
        """T-A12 (the check's t-3 with server blocks on both planes): both planes take their physics through the
        loader (a candidate's resolved platform; a record's sealed physics), and both refuse the server block."""
        why = _refusal(lambda: resolve_platform_physics(name, merge_override_blocks({"host": SERVER})))
        assert all(key in why for key in SERVER)
        sealed = get_platform_physics(name).to_dict()
        for key, row in SERVER.items():  # the rows as a toolchain that admitted them sealed them
            sealed["constants"][key] = {"key": key, "low": row.get("low", row["nominal"]), "nominal": row["nominal"],
                                        "high": row.get("high", row["nominal"]), "unit": row["unit"],
                                        "evidence_kind": row["evidence_kind"], "citation": row.get("citation", ""),
                                        "derivation": "", "note": row.get("note", ""), "overridden": True}
        why = _refusal(lambda: PlatformPhysics.from_dict(sealed))
        assert all(key in why for key in SERVER)

    @pytest.mark.parametrize("factor", [10.0])
    def test_t4_scaling_a_calibration_row_changes_no_priced_term(self, factor):
        """T-A13 (t-4 widened): a development-machine row scaled by 10 prices nothing either way: the scaled and the
        unscaled server block refuse alike, so no priced term moves (red at R5: both priced, and differently)."""
        scaled = copy.deepcopy(SERVER)
        for key in ("t_host_op_overhead", "host_macs_per_s"):
            for end in ("low", "nominal", "high"):
                if end in scaled[key]:
                    scaled[key][end] *= factor

        def priced(block):
            try:
                physics = resolve_platform_physics(CHIP, merge_override_blocks({"host": block}))
            except ToolchainRefusal as refused:
                return ("refused", type(refused).__name__, sorted(k for k in block if k in str(refused)))
            pricing = price_absolute(quantities(synaptic_events=1e6, host_macs=1e6, compute_op_count=3,
                                                cores_physical=8, latency_steps=32, timesteps=4), physics)
            return ("priced", sorted((t.name, t.value) for t in pricing.terms))

        assert priced(SERVER) == priced(scaled)
        assert priced(SERVER)[0] == "refused"
