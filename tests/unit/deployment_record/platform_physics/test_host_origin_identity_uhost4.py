"""[U-HOST4] The third verify's m-1 as a class (rulings 2026-11-11, U-HOST3 item 1): a description is identified by its
digest, never by its path. One digest held at two paths is one description wherever descriptions or origins are
compared, collected or counted; the stated path stays in the record as stated text.

The sites (FU/uhost_pricing4/report.md, the sweep), one class each:
  the origin's identity (``HostOrigin``: kind, name and digest compared, the path a statement);
  the one-description rule (``admission.admit_host_rows``: origins collected and counted), with a re-pricing with the
  attach row (``loader.apply_overrides``) and the origin it seals;
  the judgment a read returns (``admission.described``: one judgment per description, the origin stating where this
  read found it);
  the read-back of a seal (``registry.held_description``: held by name and digest);
  the registration's name rule (one name, one registration, whatever path is offered);
  and the census of where an origin's path is read and where an origin is built.
Two descriptions with one name and different digests still refuse by name. No description, mode, host or chip is
named here: the first read description, its default mode, and the first registry profile with a backing store.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.hosts import admission as A
from mimarsinan.deployment_record.platform_physics.hosts import registry as REG
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    host_attach_block, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.loader import apply_overrides
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.held_hosts import correct, digest, first_row, shipped_copy  # noqa: F401  (a fixture)
from unit.deployment_record.registered_hosts import registered_copy

NAME = host_descriptions()[0]
ORIGIN = A.ORIGIN_FIELD
AS_NAME = "test_two_paths_host"
STORE_CHIP = next(p for p in available_profiles() if get_platform_physics(p).backing_store is not None)
VIEW = dict(synaptic_events=1e6, host_macs=1e8, compute_op_count=3, cores_physical=8, latency_steps=32, timesteps=4,
            neurons_used=100, segment_cores=1)
PACKAGE = Path(REG.__file__).resolve().parents[3]


def _same(data):
    """The description's bytes as read (registered_copy writes the same bytes in every folder)."""


def _scaled(data):
    """Other bytes under the same name: the first row the default mode resolves, each corner times 1.01."""
    rows, key = first_row(data)
    for end in ("low", "nominal", "high"):
        if end in rows[key]:
            rows[key][end] *= 1.01


def _sealed_with_store(name):
    """A physics priced on the description ``name`` with a host-owned store (so the attach row composes), sealed."""
    blocks = {"host": host_description_block(name), "store": ST.store_block(ST.HOST, "a variant (test)")}
    return resolve_platform_physics(STORE_CHIP, merge_override_blocks(blocks)).to_dict()


def _repriced(payload, name):
    """The sealed payload read back and re-priced with the attach row: a chip row fed over the host's link, judged
    against the description readable now, beside the kept host rows judged against the sealed origin."""
    return apply_overrides(PlatformPhysics.from_dict(payload), host_attach_block(name))


def _priced(physics):
    return {t.name: (t.band.low, t.band.nominal, t.band.high)
            for t in price_absolute(quantities(**VIEW), physics).terms}


def _refused(call):
    with pytest.raises(ToolchainRefusal) as why:
        call()
    return str(why.value)


def _origin(**edit):
    return REG.HostOrigin.from_dict({"kind": REG.SHIPPED, "name": NAME, "path": "a/description.json",
                                     "sha256": "0" * 64, **edit})


class TestTheOriginsIdentity:
    def test_one_digest_at_two_paths_is_one_origin_and_each_states_its_path(self):
        first, second = _origin(), _origin(path="b/description.json")
        assert first == second and hash(first) == hash(second) and len({first, second}) == 1
        assert first.to_dict()["path"] == "a/description.json" and second.to_dict()["path"] == "b/description.json"
        assert "a/description.json" in first.label and "b/description.json" in second.label

    @pytest.mark.parametrize("edit", [{"sha256": "1" * 64}, {"kind": REG.REGISTERED}, {"name": "test_other_host"}],
                             ids=["digest", "kind", "name"])
    def test_another_digest_kind_or_name_is_another_origin(self, edit):
        assert _origin() != _origin(**edit) and len({_origin(), _origin(**edit)}) == 2

    def test_the_identity_is_the_kind_the_name_and_the_digest(self):
        assert sorted(f.name for f in dataclasses.fields(REG.HostOrigin) if f.compare) == ["kind", "name", "sha256"]
        assert sorted(f.name for f in dataclasses.fields(REG.HostOrigin)) == sorted(REG.ORIGIN_FIELDS)


class TestTheOneDescriptionRule:
    def test_a_registered_seal_read_back_where_its_bytes_are_registered_from_another_folder_reprices_alike(
            self, tmp_path):
        """The verify's o7: one digest registered from two folders is one description; re-priced with the attach row,
        the physics and its price equal the re-pricing in the first registration, its stated path kept."""
        with registered_copy(tmp_path / "a", NAME, _same, as_name=AS_NAME) as name:
            payload = _sealed_with_store(name)
            first = _repriced(payload, name)
            first_price = _priced(first)
        with registered_copy(tmp_path / "b", NAME, _same, as_name=AS_NAME) as name:
            path = REG.description_path(name)
            assert str(path) != payload[ORIGIN]["path"] and digest(path) == payload[ORIGIN]["sha256"]
            assert PlatformPhysics.from_dict(payload).to_dict() == payload
            again = _repriced(payload, name)
            assert _priced(again) == first_price
        assert again.to_dict() == first.to_dict()
        assert again.to_dict()[ORIGIN] == payload[ORIGIN]

    def test_a_shipped_seal_whose_stated_path_is_edited_reads_back_and_reprices_alike(self):
        """The verify's o7b: the stated path is text; the seal reads back byte for byte and re-prices with the attach
        row to the same physics and price, the edited path still stated."""
        payload = _sealed_with_store(NAME)
        first = _repriced(payload, NAME)
        edited = copy.deepcopy(payload)
        edited[ORIGIN]["path"] = "elsewhere/" + Path(payload[ORIGIN]["path"]).name
        assert PlatformPhysics.from_dict(edited).to_dict() == edited
        again = _repriced(edited, NAME)
        assert again.to_dict()[ORIGIN] == edited[ORIGIN]
        assert {**again.to_dict(), ORIGIN: None} == {**first.to_dict(), ORIGIN: None}
        assert _priced(again) == _priced(first)

    def test_the_control_one_registration_reprices(self, tmp_path):
        with registered_copy(tmp_path / "a", NAME, _same, as_name=AS_NAME) as name:
            payload = _sealed_with_store(name)
            assert _repriced(payload, name).to_dict()[ORIGIN] == payload[ORIGIN]


class TestTwoDescriptionsWithOneNameAndDifferentDigests:
    """Controls of the identity: the digest still decides (they hold on revision 3 and must still hold)."""

    def test_other_bytes_registered_from_another_folder_refuse_the_earlier_seal_by_name(self, tmp_path):
        with registered_copy(tmp_path / "a", NAME, _same, as_name=AS_NAME) as name:
            payload = _sealed_with_store(name)
        with registered_copy(tmp_path / "b", NAME, _scaled, as_name=AS_NAME):
            why = _refused(lambda: PlatformPhysics.from_dict(payload))
        assert AS_NAME in why and payload[ORIGIN]["sha256"] in why and "not held" in why

    def test_other_bytes_at_the_same_path_refuse_the_earlier_seal_by_name(self, tmp_path):
        with registered_copy(tmp_path, NAME, _same, as_name=AS_NAME) as name:
            payload = _sealed_with_store(name)
        with registered_copy(tmp_path, NAME, _scaled, as_name=AS_NAME) as name:
            assert str(REG.description_path(name)) == payload[ORIGIN]["path"]
            why = _refused(lambda: PlatformPhysics.from_dict(payload))
        assert AS_NAME in why and payload[ORIGIN]["sha256"] in why and "not held" in why

    def test_rows_judged_against_two_digests_of_one_name_refuse_by_name(self, shipped_copy):
        """The earlier seal's kept rows beside the attach row judged against the corrected description: two digests
        under one name are two descriptions, at one stated path."""
        shipped, held = shipped_copy
        earlier = _sealed_with_store(NAME)
        correct(shipped, held)
        later = digest(shipped / f"{NAME}.json")
        why = _refused(lambda: _repriced(earlier, NAME))
        assert "one description" in why and earlier[ORIGIN]["sha256"] in why and later in why


class TestTheJudgmentStatesWhereItWasRead:
    def test_a_new_seal_states_the_path_its_description_is_read_from_now(self, tmp_path):
        """One judgment per description (its digest); a read from another path states that path, never the path of
        an earlier read of the same bytes."""
        with registered_copy(tmp_path / "a", NAME, _same, as_name=AS_NAME) as name:
            first = _sealed_with_store(name)
        with registered_copy(tmp_path / "b", NAME, _same, as_name=AS_NAME) as name:
            later = _sealed_with_store(name)
            assert later[ORIGIN]["path"] == str(REG.description_path(name))
        assert later[ORIGIN]["path"] != first[ORIGIN]["path"]
        assert {**later, ORIGIN: None} == {**first, ORIGIN: None}
        assert {**later[ORIGIN], "path": None} == {**first[ORIGIN], "path": None}

    def test_a_sealed_rows_judgment_states_the_sealed_path(self, tmp_path):
        """A read-back judges the kept rows against the sealed origin and states its path, the one the record holds."""
        payload = _sealed_with_store(NAME)
        edited = copy.deepcopy(payload)
        edited[ORIGIN]["path"] = "elsewhere/" + Path(payload[ORIGIN]["path"]).name
        origin = REG.HostOrigin.from_dict(edited[ORIGIN])
        host = next(value["host"] for value in payload["constants"].values() if value.get("host"))
        assert A._origin_of(host, origin).path == edited[ORIGIN]["path"]
        assert A._origin_of(host, None).path == payload[ORIGIN]["path"]


class TestTheReadBackOfASeal:
    def test_a_registered_seal_is_held_by_name_and_digest_wherever_its_bytes_are_registered(self, tmp_path):
        with registered_copy(tmp_path / "a", NAME, _same, as_name=AS_NAME) as name:
            origin = PlatformPhysics.from_dict(_sealed_with_store(name)).host_origin
        with registered_copy(tmp_path / "b", NAME, _same, as_name=AS_NAME) as name:
            assert REG.held_description(origin) == REG.description_path(name).read_bytes()


class TestTheRegistrationsNameRule:
    @pytest.mark.parametrize("where", ["the same file", "a byte copy in another folder"])
    def test_a_registered_name_offered_again_refuses_wherever_its_file_lies(self, tmp_path, where):
        """One name, one registration: the outcome does not depend on the path offered (the same file, or the same
        bytes elsewhere); the first registration is kept."""
        with registered_copy(tmp_path / "a", NAME, _same, as_name=AS_NAME) as name:
            path = REG.description_path(name)
            offered = path
            if where != "the same file":
                offered = tmp_path / "b" / path.name
                offered.parent.mkdir()
                offered.write_bytes(path.read_bytes())
                offered.with_suffix(".md").write_bytes(path.with_suffix(".md").read_bytes())
            why = _refused(lambda: A.register_host_description(offered))
            assert "already read" in why and REG.description_path(name) == path


def _census():
    """(where an origin's path is read: (file, function) of every ``<...origin>.path`` or ``<...origin>['path']``
    read), (where an origin is built: (file, function) of every ``HostOrigin(`` call), over the package's source."""
    reads, builds = set(), set()
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(PACKAGE).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                holder = None
                if isinstance(node, ast.Attribute) and node.attr == "path":
                    holder = node.value
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                        and node.slice.value == "path":
                    holder = node.value
                if holder is not None and ast.unparse(holder).split(".")[-1].split("(")[0].endswith("origin"):
                    reads.add((rel, fn.name))
                if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in ("HostOrigin", "cls") \
                        and rel.endswith("hosts/registry.py"):
                    builds.add((rel, fn.name))
                elif isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "HostOrigin":
                    builds.add((rel, fn.name))
    return reads, builds


class TestTheCensus:
    def test_an_origins_path_is_read_only_where_a_judgment_states_it(self):
        """No site compares, collects or counts origins by their path: the one read of an origin's path outside the
        type is the judgment stating where this read found its description."""
        reads, _builds = _census()
        assert reads == {("deployment_record/platform_physics/hosts/admission.py", "described")}

    def test_an_origin_is_built_only_by_the_registry(self):
        _reads, builds = _census()
        assert builds == {("deployment_record/platform_physics/hosts/registry.py", "read_description"),
                          ("deployment_record/platform_physics/hosts/registry.py", "from_dict")}
