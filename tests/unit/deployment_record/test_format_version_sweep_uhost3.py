"""[U-HOST3] m-6 swept as its class, "a version read inexactly": every reader of a format_version reads it through the
version rule's one reading (``common.refusal.format_version``: an int, never a bool, a float or a string).

The record's rule is in ``test_record_version_uhost3``. The same reading was ``int(version)`` or ``version ==
expected`` at every other reader, so 1.0, 1.5, "1" or True read as version 1. One case per reader here: the sealed
physics, a physics profile, a host description, a cost record, a certification floor book, a cost report, an area
component and a lane block; and a census over src that no reader compares or converts a version outside the rule.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.certification import CertificationFloorBook
from mimarsinan.chip_simulation.cost_extraction import CostRecord
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.common.refusal.format_version import FormatVersionError, exact_version, is_version
from mimarsinan.deployment_record.cost.terms import DeploymentCostReport
from mimarsinan.deployment_record.platform_physics import available_profiles
from mimarsinan.deployment_record.platform_physics import area_components as AC
from mimarsinan.deployment_record.platform_physics import lane_blocks as LB
from mimarsinan.deployment_record.platform_physics.hosts.description import host_descriptions
from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics

from unit.deployment_record.registered_hosts import registered_copy

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
PROFILES = SRC / "deployment_record" / "platform_physics" / "profiles"
#: Spellings of version 1 that are no integer version.
NOT_ONE = [(True, "true"), (1.0, "float"), ("1", "string"), (1.5, "one_and_a_half")]


def _refused(call):
    with pytest.raises(ToolchainRefusal) as why:
        call()
    return str(why.value)


class TestTheOneReading:
    @pytest.mark.parametrize("version", [v for v, _ in NOT_ONE] + [None, [1]], ids=[i for _, i in NOT_ONE] + ["none",
                                                                                                         "list"])
    def test_anything_but_an_int_refuses_by_name(self, version):
        with pytest.raises(FormatVersionError, match="format_version"):
            exact_version(version, "a test format")
        assert not is_version(version, 1)

    def test_an_int_is_read(self):
        assert exact_version(3, "a test format") == 3 and is_version(1, 1) and not is_version(2, 1)


@pytest.mark.parametrize("version", [v for v, _ in NOT_ONE], ids=[i for _, i in NOT_ONE])
class TestEachReader:
    def test_the_sealed_physics(self, version):
        payload = resolve_platform_physics(available_profiles()[0], None).to_dict()
        payload["format_version"] = version
        _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_a_physics_profile(self, version):
        data = json.loads((PROFILES / f"{available_profiles()[0]}.json").read_text())
        data["format_version"] = version
        _refused(lambda: profile_from_dict(data))

    def test_a_host_description(self, version, tmp_path):
        def edit(data):
            data["format_version"] = version

        with pytest.raises(ToolchainRefusal):
            with registered_copy(tmp_path, host_descriptions()[0], edit, as_name="test_version_host"):
                pass

    def test_a_cost_record(self, version):
        assert "format_version" in _refused(lambda: CostRecord.from_dict({"format_version": version}))

    def test_a_certification_floor_book(self, version):
        assert "format_version" in _refused(
            lambda: CertificationFloorBook.from_dict({"format_version": version, "floors": {}}))

    def test_a_cost_report(self, version):
        assert "format_version" in _refused(lambda: DeploymentCostReport(
            segments=(), energy=(), latency=(), area=(), throughput=(), notes=(), format_version=version))

    def test_an_area_component(self, version, tmp_path, monkeypatch):
        name = AC.component_names()[0]
        for stem in (name, "node_scaling"):
            shutil.copyfile(AC._here() / f"{stem}.json", tmp_path / f"{stem}.json")
        data = json.loads((tmp_path / f"{name}.json").read_text())
        data["format_version"] = version
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
        monkeypatch.setattr(AC, "_here", lambda: tmp_path)
        AC.load_component.cache_clear()
        AC.scaling_rules.cache_clear()
        try:
            assert name in _refused(lambda: AC.load_component(name))
        finally:
            AC.load_component.cache_clear()
            AC.scaling_rules.cache_clear()

    def test_a_lane_block(self, version, tmp_path, monkeypatch):
        name = LB.lane_block_names()[0]
        data = json.loads(Path(LB.__file__).with_name(f"{name}.json").read_text())
        data["format_version"] = version
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
        monkeypatch.setattr(LB, "__file__", str(tmp_path / "__init__.py"))
        assert name in _refused(lambda: LB.load_lane_block(name))


# ---------------------------------------------------------------------------------------------------- the census
def _is_version_read(node):
    """An expression reading a format_version: ``x.format_version``, ``x["format_version"]``, ``x.get("format_version",
    ...)``."""
    if isinstance(node, ast.Attribute) and node.attr == "format_version":
        return True
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "format_version":
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == "format_version")


def _outside_the_rule(tree):
    """Each place a version read is compared or converted with ``int`` directly (never through the rule)."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(_is_version_read(n) for n in [node.left, *node.comparators]):
            hits.append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int" and \
                any(_is_version_read(a) for a in node.args):
            hits.append(node.lineno)
    return hits


def test_no_reader_compares_or_converts_a_version_outside_the_rule():
    hits = {}
    for path in sorted(SRC.rglob("*.py")):
        if found := _outside_the_rule(ast.parse(path.read_text(encoding="utf-8"))):
            hits[path.relative_to(SRC).as_posix()] = found
    assert hits == {}
