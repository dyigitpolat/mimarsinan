"""[U-HOST3] m-6: the version rule reads exact integer versions only, and a version 1 record without a field its rule
reads refuses by name.

At revision 2 ``DeploymentRecord.from_dict`` chose the version 1 rule by ``int(format_version) == 1``, so 1.5 and True
were read as version 1 and converted (the second verify's w4), and ``_require_version``'s ``int()`` read 2.5 and "2"
as version 2; a version 1 record without ``timing.latency.host_ops_s`` raised KeyError (w2). Now a format_version is
an int (never a bool, a float or a string) or the record refuses by name (``RecordVersionError``, a ToolchainRefusal),
and the version 1 rule names the fields it reads (``VERSION_1_FIELDS``), each required: a version 1 record without one,
with an identity of another version, or already naming the version 2 field, refuses by name.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.schema import record as R

from unit.deployment_record.record_fixtures import make_full_record


def _current():
    record = make_full_record()
    latency = dataclasses.replace(record.timing.latency, host_ops_s_raw_total=0.75, host_ops_s_per_pass=0.03)
    return dataclasses.replace(record, timing=dataclasses.replace(record.timing, latency=latency)).to_dict()


def _as_version_1(payload):
    old = copy.deepcopy(payload)
    old["format_version"] = 1
    old["identity"]["format_version"] = 1
    old["timing"]["latency"]["host_ops_s"] = old["timing"]["latency"].pop("host_ops_s_raw_total")
    return old


def _refused(call):
    with pytest.raises(ToolchainRefusal) as why:
        call()
    assert isinstance(why.value, ValueError)  # the record's refusals stay ValueErrors (callers catch them so)
    return str(why.value)


class TestExactIntegerVersions:
    @pytest.mark.parametrize("version", [True, 1.5, 1.0, "1", None], ids=["true", "one_and_a_half", "one_float",
                                                                         "one_string", "none"])
    def test_no_other_spelling_of_one_is_read_by_the_version_1_rule(self, version):
        """w4: True and 1.5 were read as version 1 and converted."""
        old = _as_version_1(_current())
        old["format_version"] = version
        assert "format_version" in _refused(lambda: R.DeploymentRecord.from_dict(old))

    @pytest.mark.parametrize("version", [2.0, 2.5, "2", False], ids=["two_float", "two_and_a_half", "two_string",
                                                                     "false"])
    def test_no_other_spelling_of_two_is_read_as_version_2(self, version):
        payload = _current()
        payload["format_version"] = version
        assert "format_version" in _refused(lambda: R.DeploymentRecord.from_dict(payload))

    @pytest.mark.parametrize("version", [2.0, "2", True], ids=["two_float", "two_string", "true"])
    def test_the_identitys_version_is_exact_too(self, version):
        payload = _current()
        payload["identity"]["format_version"] = version
        assert "format_version" in _refused(lambda: R.DeploymentRecord.from_dict(payload))

    @pytest.mark.parametrize("version", [0, 2, 4, -1])  # [composed head] format 3; 2 refused (two unit forms)
    def test_another_integer_version_refuses_by_name(self, version):
        payload = _current()
        payload["format_version"] = version
        assert "format_version" in _refused(lambda: R.DeploymentRecord.from_dict(payload))

    def test_the_current_and_the_version_1_record_read(self):
        payload = _current()
        assert R.DeploymentRecord.from_dict(payload).to_dict() == payload
        assert R.DeploymentRecord.from_dict(_as_version_1(payload)).timing.latency.host_ops_s_raw_total == 0.75


class TestTheFieldsTheVersion1RuleReads:
    def test_the_rule_names_its_fields(self):
        assert R.VERSION_1_FIELDS == (("identity", "format_version"), ("timing", "latency", "host_ops_s"),
                                      ("timing", "latency", "host_ops_s_per_pass"))

    @pytest.mark.parametrize("path", [("timing", "latency", "host_ops_s"), ("timing", "latency", "host_ops_s_per_pass"),
                                      ("identity", "format_version"), ("timing", "latency"), ("timing",),
                                      ("identity",)], ids=lambda p: ".".join(p))
    def test_a_version_1_record_without_one_refuses_by_name(self, path):
        """w2: a KeyError at revision 2."""
        old = _as_version_1(_current())
        holder = old
        for part in path[:-1]:
            holder = holder[part]
        holder.pop(path[-1])
        assert ".".join(path) in _refused(lambda: R.DeploymentRecord.from_dict(old))

    @pytest.mark.parametrize("version", [2, True, "1"], ids=["two", "true", "one_string"])
    def test_a_version_1_record_whose_identity_states_another_version_refuses_by_name(self, version):
        old = _as_version_1(_current())
        old["identity"]["format_version"] = version
        assert "identity" in _refused(lambda: R.DeploymentRecord.from_dict(old))

    def test_a_version_1_record_already_naming_the_version_2_field_refuses_by_name(self):
        old = _as_version_1(_current())
        old["timing"]["latency"]["host_ops_s_raw_total"] = 0.5
        assert "host_ops_s_raw_total" in _refused(lambda: R.DeploymentRecord.from_dict(old))
