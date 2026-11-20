"""[U-HOST4] The third verify's m-2 at the version rule's readers: the clause that refuses another integer version.

The verify's v0578 (profile.py:195, the raise removed) survived because the unit's sweep tried versions that are no
integer only (``test_format_version_sweep_uhost3``: True, 1.0, "1", 1.5), and ``exact_version`` refuses those before
the comparison. The same clause stands at every reader the version rule sweeps, so each reader gets its case at two
other integer versions (the record's own is ``test_record_version_uhost3``): the sealed physics, a physics profile, a
host description, a cost record, a certification floor book, a cost report, an area component and a lane block. Each
case is red on its reader's raise removed (the harness's RAISE). Beside them, the cases the harness's survivors asked
for at the same readers: each reader reads its own version, and each operand of the compound refusal an area
component's and a lane block's version reading sits in (a field outside the form, a name not the file's) refuses by
name. No description, mode, host, chip, component or block is named here: the first of each kind is read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.certification import FLOOR_BOOK_FORMAT_VERSION, CertificationFloorBook
from mimarsinan.chip_simulation.cost_extraction import COST_RECORD_FORMAT_VERSION, CostRecord
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.terms import COST_REPORT_FORMAT_VERSION, DeploymentCostReport
from mimarsinan.deployment_record.platform_physics import available_profiles
from mimarsinan.deployment_record.platform_physics import area_components as AC
from mimarsinan.deployment_record.platform_physics import lane_blocks as LB
from mimarsinan.deployment_record.platform_physics.hosts.description import FORMAT_VERSION, host_descriptions
from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict
from mimarsinan.deployment_record.platform_physics.profile import PHYSICS_FORMAT_VERSION, PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics

from unit.deployment_record.registered_hosts import registered_copy

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
PROFILES = SRC / "deployment_record" / "platform_physics" / "profiles"
OTHER = [(-1, "below"), (1, "above")]


def _refused(call, kind=(ToolchainRefusal, ValueError)):
    with pytest.raises(kind) as why:
        call()
    return str(why.value)


@pytest.mark.parametrize("step", [s for s, _ in OTHER], ids=[i for _, i in OTHER])
class TestEachReaderAtAnotherIntegerVersion:
    def test_the_sealed_physics(self, step):
        payload = resolve_platform_physics(available_profiles()[0], None).to_dict()
        payload["format_version"] = PHYSICS_FORMAT_VERSION + step
        assert "format_version" in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_a_physics_profile(self, step):
        data = json.loads((PROFILES / f"{available_profiles()[0]}.json").read_text())
        data["format_version"] = PHYSICS_FORMAT_VERSION + step
        assert "format_version" in _refused(lambda: profile_from_dict(data))

    def test_a_host_description(self, step, tmp_path):
        def edit(data):
            data["format_version"] = FORMAT_VERSION + step

        with pytest.raises(ToolchainRefusal) as why:
            with registered_copy(tmp_path, host_descriptions()[0], edit, as_name="test_integer_version_host"):
                pass
        assert "format_version" in str(why.value)

    def test_a_cost_record(self, step):
        assert "format_version" in _refused(
            lambda: CostRecord.from_dict({"format_version": COST_RECORD_FORMAT_VERSION + step}))

    def test_a_certification_floor_book(self, step):
        assert "format_version" in _refused(
            lambda: CertificationFloorBook.from_dict({"format_version": FLOOR_BOOK_FORMAT_VERSION + step,
                                                      "floors": {}}))

    def test_a_cost_report(self, step):
        assert "format_version" in _refused(lambda: DeploymentCostReport(
            segments=(), energy=(), latency=(), area=(), throughput=(), notes=(),
            format_version=COST_REPORT_FORMAT_VERSION + step))

    def test_an_area_component(self, step, tmp_path, monkeypatch):
        name = AC.component_names()[0]
        for stem in (name, "node_scaling"):
            shutil.copyfile(AC._here() / f"{stem}.json", tmp_path / f"{stem}.json")
        data = json.loads((tmp_path / f"{name}.json").read_text())
        data["format_version"] = AC.AREA_COMPONENT_FORMAT_VERSION + step
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
        monkeypatch.setattr(AC, "_here", lambda: tmp_path)
        AC.load_component.cache_clear()
        AC.scaling_rules.cache_clear()
        try:
            assert "format_version" in _refused(lambda: AC.load_component(name))
        finally:
            AC.load_component.cache_clear()
            AC.scaling_rules.cache_clear()

    def test_a_lane_block(self, step, tmp_path, monkeypatch):
        name = LB.lane_block_names()[0]
        data = json.loads(Path(LB.__file__).with_name(f"{name}.json").read_text())
        data["format_version"] = LB.LANE_BLOCK_FORMAT_VERSION + step
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
        monkeypatch.setattr(LB, "__file__", str(tmp_path / "__init__.py"))
        assert "format_version" in _refused(lambda: LB.load_lane_block(name))


class TestEachReaderReadsItsOwnVersion:
    """The other half of each reader's clause (the harness's h0887 and h0888 at the floor book: the comparison taken
    always or negated): the current version of each kind reads."""

    def test_the_sealed_physics(self):
        payload = resolve_platform_physics(available_profiles()[0], None).to_dict()
        assert PlatformPhysics.from_dict(payload).to_dict() == payload

    def test_a_physics_profile(self):
        data = json.loads((PROFILES / f"{available_profiles()[0]}.json").read_text())
        assert profile_from_dict(data).format_version == PHYSICS_FORMAT_VERSION

    def test_a_certification_floor_book(self):
        book = CertificationFloorBook.from_dict({"format_version": FLOOR_BOOK_FORMAT_VERSION, "floors": {}})
        assert book.format_version == FLOOR_BOOK_FORMAT_VERSION and not book.floors

    def test_a_cost_report(self):
        report = DeploymentCostReport(segments=(), energy=(), latency=(), area=(), throughput=(), notes=())
        assert report.format_version == COST_REPORT_FORMAT_VERSION

    def test_an_area_component_and_a_lane_block(self):
        assert AC.load_component(AC.component_names()[0]).name == AC.component_names()[0]
        assert LB.load_lane_block(LB.lane_block_names()[0]).name == LB.lane_block_names()[0]


def _area_copy(tmp_path, monkeypatch, edit):
    name = AC.component_names()[0]
    for stem in (name, "node_scaling"):
        shutil.copyfile(AC._here() / f"{stem}.json", tmp_path / f"{stem}.json")
    data = json.loads((tmp_path / f"{name}.json").read_text())
    edit(data)
    (tmp_path / f"{name}.json").write_text(json.dumps(data))
    monkeypatch.setattr(AC, "_here", lambda: tmp_path)
    AC.load_component.cache_clear()
    AC.scaling_rules.cache_clear()
    return name


def _lane_copy(tmp_path, monkeypatch, edit):
    name = LB.lane_block_names()[0]
    data = json.loads(Path(LB.__file__).with_name(f"{name}.json").read_text())
    edit(data)
    (tmp_path / f"{name}.json").write_text(json.dumps(data))
    monkeypatch.setattr(LB, "__file__", str(tmp_path / "__init__.py"))
    return name


@pytest.mark.parametrize("edit", [lambda d: d.update(stray_field_test="x"), lambda d: d.update(name="test_other_name")],
                         ids=["a_field_outside_its_form", "a_name_not_its_files"])
class TestTheCompoundRefusalBesideTheVersion:
    """area_components/__init__.py:138-139 and lane_blocks/__init__.py:61-62 (the harness's h0900, h0904, h0911 and
    h0915, each operand of the compound refusal the unit's version reading sits in replaced by False): a field outside
    the form and a name that is not the file's each refuse by name."""

    def test_an_area_component(self, edit, tmp_path, monkeypatch):
        name = _area_copy(tmp_path, monkeypatch, edit)
        try:
            assert name in _refused(lambda: AC.load_component(name))
        finally:
            AC.load_component.cache_clear()
            AC.scaling_rules.cache_clear()

    def test_a_lane_block(self, edit, tmp_path, monkeypatch):
        name = _lane_copy(tmp_path, monkeypatch, edit)
        assert name in _refused(lambda: LB.load_lane_block(name))
