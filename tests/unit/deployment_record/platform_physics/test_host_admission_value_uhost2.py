"""[U-HOST2] The admission rule judges the value: the verify's cases a1 to a6, a8 and a13 (FU/uhost_pricing/verify.md
section 6), written before the rule and red on revision 1's tree (each was admitted there by its label alone).

A row naming a host (``'<description> <mode>'``) is admitted only as that mode's own row as the description resolves
it (band, unit and class equal) or as a declared derivation of its rows; anything else under the host's name refuses
by name: a development machine's measured rate or dispatch, another host's power band, a row for a mode the
description refuses, a row the description does not declare, the relabel a test helper once applied, and a sealed
row whose value was changed. No description, mode or chip is named here: the first read description, its documented
default mode, and the first registry profile that takes its block.
"""

from __future__ import annotations

import copy
import importlib.util

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import available_profiles
from mimarsinan.deployment_record.platform_physics.constants import keys_in_group, spec_for
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HOST_IDENTITY_FIELD, host_description, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

from unit.deployment_record.g3_fixtures import quantities

NAME = host_descriptions()[0]
#: A development machine's calibration block in the form a host calibration wrote it: a measured dispatch wall and a
#: measured MAC rate of the machine that ran the toolchain, and a power band spanning two other machines (test
#: numbers in the shape of such a block; no machine is named).
DEVELOPMENT_MACHINE = {
    "t_host_op_overhead": {"low": 13.37006688117981, "nominal": 13.479962944984436, "high": 13.640150427818298,
                           "unit": "us", "evidence_kind": "measured",
                           "note": "per-invocation dispatch wall measured on the development machine (test)"},
    "host_macs_per_s": {"nominal": 21.986041466365474, "unit": "G/s", "evidence_kind": "measured",
                        "note": "MAC rate measured on the development machine (test)"},
    "p_host": {"low": 4.98, "nominal": 37.34, "high": 280.0, "unit": "W", "evidence_kind": "published",
               "citation": "a band over two other machines (test)", "note": "a band over two hosts (test)"},
}
VIEW = dict(synaptic_events=1e6, host_macs=1e8, compute_op_count=3, cores_physical=8, latency_steps=32, timesteps=4,
            neurons_used=100, segment_cores=1)


def _block(mode="default"):
    return host_description_block(NAME, mode)


def _identity():
    return _block()["p_host"][HOST_IDENTITY_FIELD]


def _chip():
    """The first registry profile that takes the identified block (the rule judges the host rows a run adds)."""
    for name in available_profiles():
        try:
            resolve_platform_physics(name, merge_override_blocks({"host": _block()}))
        except ToolchainRefusal:
            continue
        return name
    raise AssertionError("no registry profile takes the identified host block")


def _resolve(rows):
    return resolve_platform_physics(_chip(), merge_override_blocks({"host": rows}))


def _refused(call):
    with pytest.raises(ToolchainRefusal) as refused:
        call()
    return str(refused.value)


class TestTheVerifysCases:
    def test_a1_a_partial_block_of_a_foreign_rate_named_published_refuses(self):
        """a1: one row, the development machine's measured rate, declared published under the description's name."""
        row = {**DEVELOPMENT_MACHINE["host_macs_per_s"], "evidence_kind": "published",
               "citation": "a citation string (test)", HOST_IDENTITY_FIELD: _identity()}
        assert "host_macs_per_s" in _refused(lambda: _resolve({"host_macs_per_s": row}))

    def test_a2_the_same_row_with_no_evidence_class_refuses(self):
        """a2: the loader's default class (estimated) is no description's row either."""
        row = {"nominal": DEVELOPMENT_MACHINE["host_macs_per_s"]["nominal"], "unit": "G/s",
               HOST_IDENTITY_FIELD: _identity()}
        assert "host_macs_per_s" in _refused(lambda: _resolve({"host_macs_per_s": row}))

    def test_a3_the_whole_block_with_one_row_replaced_by_another_hosts_band_refuses(self):
        """a3: the identified block with its power row replaced by a band over two other machines, identity kept."""
        block = copy.deepcopy(_block())
        block["p_host"].update({end: DEVELOPMENT_MACHINE["p_host"][end] for end in ("low", "nominal", "high")})
        assert "p_host" in _refused(lambda: _resolve(block))

    def test_a4_a_row_for_a_mode_the_description_refuses_refuses(self):
        """a4: a power row naming a mode whose description declares no power row (it states why)."""
        desc = host_description(NAME)
        modes = [m for m in sorted(desc.modes) if "p_host" not in (desc.modes[m].get("constants") or {})
                 and "p_host" not in desc.constants]
        assert modes, "the description declares p_host in every mode"
        row = {"nominal": 280.0, "unit": "W", "evidence_kind": "estimated", "note": "a test row",
               HOST_IDENTITY_FIELD: f"{NAME} {modes[0]}"}
        assert "p_host" in _refused(lambda: _resolve({"p_host": row}))

    def test_a5_a_row_the_description_does_not_declare_refuses(self):
        """a5: a host row no mode of the description declares, named under its identity."""
        declared = set(host_description(NAME).constants).union(
            *((facts.get("constants") or {}) for facts in host_description(NAME).modes.values()))
        undeclared = [key for key in keys_in_group(HOST) if key not in declared]
        assert undeclared, "the description declares every host row"
        key = undeclared[0]
        row = {"nominal": 1.0, "unit": spec_for(key).display_unit, "evidence_kind": "derived",
               "derivation": "identity (test)", HOST_IDENTITY_FIELD: _identity()}
        assert key in _refused(lambda: _resolve({key: row}))

    def test_a6_the_development_machines_block_stamped_and_relabelled_refuses(self):
        """a6: the development machine's block with the identity stamped and 'measured' relabelled 'estimated'."""
        stamped = {key: {**row, HOST_IDENTITY_FIELD: _identity(),
                         "evidence_kind": "estimated" if row["evidence_kind"] == "measured" else row["evidence_kind"]}
                   for key, row in DEVELOPMENT_MACHINE.items()}
        why = _refused(lambda: _resolve(stamped))
        assert all(key in why for key in DEVELOPMENT_MACHINE)

    def test_a8_a_sealed_identified_row_with_a_foreign_value_refuses(self):
        """a8: a sealed identified physics whose rate was changed to the development machine's."""
        payload = _resolve(_block()).to_dict()
        rate = DEVELOPMENT_MACHINE["host_macs_per_s"]["nominal"]
        payload["constants"]["host_macs_per_s"].update(low=rate, nominal=rate, high=rate)
        assert "host_macs_per_s" in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_a13_the_relabel_a_test_helper_applied_refuses_and_the_helper_is_gone(self):
        """a13: the transformation revision 1's test helper applied (every host row stamped with the first
        description's default identity, measured rows declared estimated) refuses on the development machine's block;
        the helper module is removed."""
        relabelled = {}
        for key, row in DEVELOPMENT_MACHINE.items():
            entry = dict(row)
            if spec_for(key).group == HOST:
                entry[HOST_IDENTITY_FIELD] = f"{NAME} {host_description(NAME).default_mode}"
                if entry.get("evidence_kind") == "measured":
                    entry["evidence_kind"] = "estimated"
                    entry.setdefault("note", "a test row")
            relabelled[key] = entry
        why = _refused(lambda: _resolve(relabelled))
        assert all(key in why for key in DEVELOPMENT_MACHINE)
        assert importlib.util.find_spec("unit.deployment_record.host_identity") is None


class TestTheRuleStillAdmitsTheDescription:
    def test_the_identified_block_resolves_and_reads_back(self):
        """Control: the description's own block is admitted and its sealed physics reads back byte for byte."""
        physics = _resolve(_block())
        assert PlatformPhysics.from_dict(physics.to_dict()).to_dict() == physics.to_dict()

    def test_a12_a_foreign_rate_never_prices(self):
        """a12 (a guard): were a foreign rate admitted it would price; the rule refuses it, so the priced e2e is
        the description's alone."""
        good = _resolve(_block())
        block = copy.deepcopy(_block())
        rate = DEVELOPMENT_MACHINE["host_macs_per_s"]["nominal"]
        block["host_macs_per_s"].update(low=rate, nominal=rate, high=rate)
        _refused(lambda: _resolve(block))
        terms = {t.name: t for t in price_absolute(quantities(**VIEW), good).terms}
        assert "e2e_latency_s" in terms
