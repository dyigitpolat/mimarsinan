"""[U-HOST2] The record names its raw host wall apart and states its version (verify m-3).

Revision 1 kept one name for two normalizations: the sealed ``timing.latency.host_ops_s`` was the raw whole-run total
while the ``host_ops_s`` quantity and the ODIN fragment's ``host_ops_s`` are per inference, and the sealed per-pass
wall changed from per call to per inference, all under format version 1. So: the sealed raw field is renamed
(``host_ops_s_raw_total``, the ODIN fragment's name for the same raw total), the record's format version is raised to
2, the schema document states both fields and the version, and a version 1 record is read by one stated rule (its raw
total renamed with its value, its per-pass wall withheld because version 1's was per call, every other field as
sealed); any other version refuses.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest

from mimarsinan.deployment_record.schema import record as R
from mimarsinan.deployment_record.schema.physics import LatencyDecomposition

from unit.deployment_record.record_fixtures import make_full_record

DOC = Path(__file__).resolve().parents[3] / "docs" / "deployment_record_schema.md"


def _fields():
    return {f.name for f in dataclasses.fields(LatencyDecomposition)}


def _timed_record():
    record = make_full_record()
    latency = dataclasses.replace(record.timing.latency, host_ops_s_raw_total=0.75, host_ops_s_per_pass=0.03)
    return dataclasses.replace(record, timing=dataclasses.replace(record.timing, latency=latency))


def _as_version_1(payload):
    """A record dict as a version 1 toolchain sealed it: version 1, the raw total under host_ops_s."""
    old = copy.deepcopy(payload)
    old["format_version"] = 1
    old["identity"]["format_version"] = 1
    old["timing"]["latency"]["host_ops_s"] = old["timing"]["latency"].pop("host_ops_s_raw_total")
    return old


class TestTheSealedFieldsAndTheVersion:
    def test_the_raw_total_is_named_apart_from_the_per_inference_quantity(self):
        assert {"host_ops_s_raw_total", "host_ops_s_per_pass"} <= _fields() and "host_ops_s" not in _fields()

    def test_the_format_version_is_raised(self):
        assert R.DEPLOYMENT_RECORD_FORMAT_VERSION == 3  # [composed head] format 3 (U-HOST2 and U-LIF2 W7)
        payload = make_full_record().to_dict()
        assert payload["format_version"] == 3 and payload["identity"]["format_version"] == 3
        assert "host_ops_s_raw_total" in payload["timing"]["latency"]

    def test_a_current_record_round_trips(self):
        payload = _timed_record().to_dict()
        assert R.DeploymentRecord.from_dict(payload).to_dict() == payload


class TestAVersion1Record:
    def test_it_is_read_by_the_one_stated_rule(self):
        """The raw total renamed with its value; the per-pass wall (per call in version 1) withheld; the rest as
        sealed; the result is a version 3 record."""
        current = _timed_record().to_dict()
        read = R.DeploymentRecord.from_dict(_as_version_1(current)).to_dict()
        assert read["timing"]["latency"]["host_ops_s_raw_total"] == 0.75
        assert read["timing"]["latency"]["host_ops_s_per_pass"] is None
        expected = copy.deepcopy(current)
        expected["timing"]["latency"]["host_ops_s_per_pass"] = None
        assert read == expected

    def test_the_rule_is_stated_once_in_the_schema_and_in_the_document(self):
        assert "version 1" in R.VERSION_1_RULE and "host_ops_s_raw_total" in R.VERSION_1_RULE
        text = DOC.read_text(encoding="utf-8")
        assert "DEPLOYMENT_RECORD_FORMAT_VERSION = 3" in text and "# == 3" in text
        assert "host_ops_s_raw_total" in text and "host_ops_s_per_pass" in text
        assert "samples_per_call" in text and "invocations" in text
        assert R.VERSION_1_RULE in " ".join(text.split())

    @pytest.mark.parametrize("version", [0, 2, 4])  # [composed head] 2 refused by name: two unit forms
    def test_any_other_version_refuses(self, version):
        payload = make_full_record().to_dict()
        payload["format_version"] = version
        with pytest.raises(ValueError, match="format_version"):
            R.DeploymentRecord.from_dict(payload)
