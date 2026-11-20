"""[U-HOST3] The origin seal (rulings 2026-11-11, item 3): one class, "a sealed value whose description cannot be named
again" (the second verify's K-1 and m-3).

A sealed physics carries its description's origin (name, path, digest, shipped or registered) and is judged at
read-back against the description of that digest, held by digest; a description not held refuses by name. The two
consequences: a corrected shipped description leaves an earlier record priceable under its own digest, and a key
resting on a registered description names it. A sealed physics whose rows name a host and that seals no origin (a
version 2 record of revision 2) refuses by name: its description cannot be named again. The registry's two edges
(m-4) refuse by name: a registered file gone or changed, and a name no identity can name.
No description, mode, host or chip is named here: the first read description, its default mode, and the first
registry profile that takes its block.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.hosts import registry as REG
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.admission import (
    register_host_description, unregister_host_description,
)
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HOST_IDENTITY_FIELD, host_attach_block, host_description, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.loader import apply_overrides
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.registered_hosts import declared_rows, registered, registered_copy

NAME = host_descriptions()[0]
ORIGIN = "host_origin"
VIEW = dict(synaptic_events=1e6, host_macs=1e8, compute_op_count=3, cores_physical=8, latency_steps=32, timesteps=4,
            neurons_used=100, segment_cores=1)


def _chip():
    for name in available_profiles():
        try:
            resolve_platform_physics(name, merge_override_blocks({"host": host_description_block(NAME)}))
        except ToolchainRefusal:
            continue
        return name
    raise AssertionError("no registry profile takes the identified host block")


CHIP = _chip()


def _resolve(rows, chip=CHIP):
    return resolve_platform_physics(chip, merge_override_blocks({"host": rows}))


def _sealed(name=NAME):
    return _resolve(host_description_block(name)).to_dict()


def _e2e(physics):
    terms = {t.name: t for t in price_absolute(quantities(**VIEW), physics).terms}
    return terms["e2e_latency_s"].band.nominal


def _refused(call):
    with pytest.raises(ToolchainRefusal) as why:
        call()
    return str(why.value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_row(data):
    """(the rows mapping holding it, the key): the first row the default mode resolves, its band to be corrected."""
    mode = data["modes"][data["default_mode"]].get("constants") or {}
    rows = {**data.get("constants", {}), **mode}
    key = sorted(k for k in rows if "nominal" in rows[k])[0]
    return (mode if key in mode else data["constants"]), key


@pytest.fixture
def shipped_copy(tmp_path, monkeypatch):
    """The shipped directory and its held store as copies (the first description's file pair; its bytes held), so a
    test can correct a shipped description as a release would: the corrected bytes added to the held store, the
    earlier bytes kept there."""
    shipped, held = tmp_path / "shipped", tmp_path / "shipped" / "held"
    held.mkdir(parents=True)
    source = REG.description_path(NAME)
    for suffix in (".json", ".md"):
        shutil.copyfile(source.with_suffix(suffix), shipped / f"{NAME}{suffix}")
    shutil.copyfile(source, held / f"{_digest(source)}.json")
    monkeypatch.setattr(REG, "_SHIPPED", shipped)
    monkeypatch.setattr(REG, "_HELD", held)
    return shipped, held


def _correct(shipped: Path, held: Path, factor=1.01):
    """Correct the shipped description's first row (each corner times ``factor``) and hold the corrected bytes."""
    path = shipped / f"{NAME}.json"
    data = json.loads(path.read_text())
    rows, key = _first_row(data)
    for end in ("low", "nominal", "high"):
        if end in rows[key]:
            rows[key][end] *= factor
    path.write_text(json.dumps(data, indent=1))
    shutil.copyfile(path, held / f"{_digest(path)}.json")
    return key


class TestTheSealedOrigin:
    def test_a_sealed_physics_carries_its_descriptions_origin(self):
        payload = _sealed()
        source = REG.description_path(NAME)
        assert payload[ORIGIN] == {"kind": "shipped", "name": NAME, "path": f"{REG.SHIPPED_DIRECTORY}/{source.name}",
                                   "sha256": _digest(source)}

    def test_the_stated_path_never_names_the_checkouts_location(self):
        assert not Path(_sealed()[ORIGIN]["path"]).is_absolute()

    def test_it_reads_back_byte_for_byte_against_the_held_description_of_its_digest(self):
        payload = _sealed()
        assert PlatformPhysics.from_dict(payload).to_dict() == payload

    def test_a_physics_without_host_rows_seals_no_origin(self):
        assert ORIGIN not in resolve_platform_physics(CHIP, None).to_dict()

    def test_every_shipped_description_is_held_and_every_held_file_hashes_to_its_name(self):
        for name in host_descriptions():
            path = REG.description_path(name)
            assert (REG._HELD / f"{_digest(path)}.json").read_bytes() == path.read_bytes(), name
        for held in sorted(REG._HELD.glob("*.json")):
            assert _digest(held) == held.stem, held.name
            assert host_description(json.loads(held.read_text())["name"]).name  # a description the judge reads


class TestACorrectedShippedDescription:
    def test_an_earlier_record_stays_priceable_under_its_own_digest(self, shipped_copy):
        shipped, held = shipped_copy
        earlier = _sealed()
        price = _e2e(PlatformPhysics.from_dict(earlier))
        key = _correct(shipped, held)
        again = PlatformPhysics.from_dict(earlier)  # judged against the earlier bytes, held by their digest
        assert again.to_dict() == earlier and _e2e(again) == price
        later = _sealed()
        assert later[ORIGIN]["sha256"] != earlier[ORIGIN]["sha256"]
        assert later["constants"][key]["nominal"] == pytest.approx(earlier["constants"][key]["nominal"] * 1.01)

    def test_an_earlier_record_whose_digest_is_not_held_refuses_by_name(self, shipped_copy):
        shipped, held = shipped_copy
        earlier = _sealed()
        _correct(shipped, held)
        (held / f"{earlier[ORIGIN]['sha256']}.json").unlink()
        why = _refused(lambda: PlatformPhysics.from_dict(earlier))
        assert earlier[ORIGIN]["sha256"] in why and NAME in why and "not held" in why

    def test_a_shipped_description_whose_bytes_are_not_held_refuses_at_declaration(self, shipped_copy):
        """A shipped description edited without holding its bytes: a record sealed under it could not be read back,
        so no row naming it is admitted."""
        shipped, held = shipped_copy
        _correct(shipped, held)
        path = shipped / f"{NAME}.json"
        (held / f"{_digest(path)}.json").unlink()
        why = _refused(lambda: _sealed())
        assert "not held" in why and _digest(path) in why

    def test_the_earlier_sealed_rows_beside_a_row_judged_against_the_corrected_one_refuse(self, shipped_copy):
        """One physics rests on one description: the earlier sealed host rows kept beside a row declared now (the
        attach row, which replaces no host row) were judged against two digests."""
        shipped, held = shipped_copy
        chip = next(p for p in available_profiles() if get_platform_physics(p).backing_store is not None)
        blocks = {"host": host_description_block(NAME), "store": ST.store_block(ST.HOST, "a variant (test)")}
        earlier = PlatformPhysics.from_dict(resolve_platform_physics(chip, merge_override_blocks(blocks)).to_dict())
        _correct(shipped, held)
        why = _refused(lambda: apply_overrides(earlier, host_attach_block(NAME)))
        assert "one description" in why


class TestTheSealedOriginIsJudged:
    def test_a_sealed_host_physics_without_an_origin_refuses_by_name(self):
        """The decided reading of a version 2 record without an origin (revision 2 sealed none): its description
        cannot be named again, so it is never judged against the description that holds the name now."""
        payload = _sealed()
        payload.pop(ORIGIN)
        why = _refused(lambda: PlatformPhysics.from_dict(payload))
        assert ORIGIN in why and NAME in why

    def test_an_origin_beside_no_row_naming_a_host_refuses_by_name(self):
        payload = resolve_platform_physics(CHIP, None).to_dict()
        payload[ORIGIN] = _sealed()[ORIGIN]
        assert ORIGIN in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_an_origin_of_another_description_refuses_by_name(self, tmp_path):
        rows = {k: {f: v for f, v in row.items() if f not in (HOST_IDENTITY_FIELD, "energy_plane", "link_ends")}
                for k, row in host_description_block(NAME).items()}
        with registered(tmp_path, "test_other_host", rows):
            other = _resolve(declared_rows("test_other_host", rows)).to_dict()[ORIGIN]
            payload = _sealed()
            payload[ORIGIN] = other
            assert "test_other_host" in _refused(lambda: PlatformPhysics.from_dict(payload))

    @pytest.mark.parametrize("edit", [
        {"kind": "registered"}, {"sha256": "0" * 64}, {"name": NAME + "x"},
    ], ids=["kind", "digest", "name"])
    def test_an_edited_origin_refuses_by_name(self, edit):
        payload = _sealed()
        payload[ORIGIN].update(edit)
        _refused(lambda: PlatformPhysics.from_dict(payload))

    @pytest.mark.parametrize("form", [
        {"drop": "path"}, {"add": ("note", "x")}, {"set": ("kind", "vendor")}, {"set": ("sha256", "ABC")},
        {"set": ("sha256", 7)}, {"set": ("path", "")}, {"set": ("name", "two words")}, {"set": ("name", "")},
        {"whole": "a string"}, {"whole": None},
    ], ids=["missing", "extra", "kind", "digest_form", "digest_type", "path", "name_space", "name_empty", "string",
            "none"])
    def test_an_origin_outside_its_form_refuses_by_name(self, form):
        payload = _sealed()
        origin = payload[ORIGIN]
        if "drop" in form:
            origin.pop(form["drop"])
        if "add" in form:
            origin[form["add"][0]] = form["add"][1]
        if "set" in form:
            origin[form["set"][0]] = form["set"][1]
        if "whole" in form:
            payload[ORIGIN] = form["whole"]
            if form["whole"] is None:  # a None origin beside rows naming a host: no origin sealed
                assert ORIGIN in _refused(lambda: PlatformPhysics.from_dict(payload))
                return
        assert "host origin" in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_a_stated_path_is_a_statement_the_digest_names_the_description(self):
        """The same bytes may be held at another path (a registered file registered again from elsewhere, a shipped
        version kept in the held store): the digest names the description, the path states where it was read."""
        payload = _sealed()
        payload[ORIGIN]["path"] = "elsewhere/" + Path(payload[ORIGIN]["path"]).name
        assert PlatformPhysics.from_dict(payload).to_dict() == payload


class TestAKeyRestingOnARegisteredDescription:
    def test_its_sealed_physics_names_the_registered_description(self, tmp_path):
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_registered_host") as name:
            path = REG.description_path(name)
            payload = _sealed(name)
            assert payload[ORIGIN] == {"kind": "registered", "name": name, "path": str(path),
                                       "sha256": _digest(path)}
            assert PlatformPhysics.from_dict(payload).to_dict() == payload

    def test_read_back_where_it_is_not_held_refuses_naming_it(self, tmp_path):
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_registered_host") as name:
            payload = _sealed(name)
        why = _refused(lambda: PlatformPhysics.from_dict(payload))
        assert "registered" in why and name in why and payload[ORIGIN]["sha256"] in why
        assert payload[ORIGIN]["path"] in why

    def test_registering_the_same_bytes_again_holds_them_again(self, tmp_path):
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_registered_host") as name:
            payload = _sealed(name)
        with registered_copy(tmp_path / "again", NAME, lambda data: None, as_name="test_registered_host"):
            assert PlatformPhysics.from_dict(payload).to_dict() == payload


class TestTheRegistrysTwoEdges:
    def test_a_registered_file_removed_refuses_by_name(self, tmp_path):
        """m-4 (the verify's r7): a FileNotFoundError at revision 2."""
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_removed_host") as name:
            path = REG.description_path(name)
            block = host_description_block(name)
            path.unlink()
            why = _refused(lambda: _resolve(block))
            assert name in why and str(path) in why

    def test_a_registered_file_changed_after_registration_refuses_by_name(self, tmp_path):
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_changed_host") as name:
            path = REG.description_path(name)
            block = host_description_block(name)
            data = json.loads(path.read_text())
            data["display_name"] = data["display_name"] + " (changed after registration)"
            path.write_text(json.dumps(data, indent=1))
            why = _refused(lambda: _resolve(block))
            assert name in why and "changed after registration" in why

    @pytest.mark.parametrize("name", ["test host", "test\thost"], ids=["space", "tab"])
    def test_a_name_no_identity_can_name_refuses_at_registration(self, tmp_path, name):
        """m-4 (the verify's r8): registered at revision 2, though '<description> <mode>' splits at its first
        space."""
        data = json.loads(REG.description_path(NAME).read_text())
        data.update(name=name, description_file=f"{name}.md")
        (tmp_path / f"{name}.md").write_text("# a test description\n")
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        try:
            register_host_description(path)
        except ToolchainRefusal as why:
            assert "one token" in str(why)
        else:
            unregister_host_description(name)
            raise AssertionError(f"a description named {name!r}, which no identity can name, was registered")
        assert name not in REG.description_paths()

    def test_unregistering_a_name_never_registered_refuses_by_name(self):
        assert "test_never_registered" in _refused(lambda: unregister_host_description("test_never_registered"))


def test_a_development_machine_block_registered_outside_a_test_is_named_by_its_key(tmp_path):
    """The verify's r1 and r6: registration is the contract (the ruling's item 3 names such a key rather than
    confining registration); a physics priced under a registered description states so in its sealed origin."""
    rows = {k: {f: v for f, v in row.items() if f not in (HOST_IDENTITY_FIELD, "energy_plane", "link_ends")}
            for k, row in copy.deepcopy(host_description_block(NAME)).items()}
    for row in rows.values():
        row["note"] = "a development machine's number, relabelled (test)"
    with registered(tmp_path, "test_development_machine", rows):
        sealed = _resolve(declared_rows("test_development_machine", rows)).to_dict()
    assert sealed[ORIGIN]["kind"] == "registered" and sealed[ORIGIN]["name"] == "test_development_machine"
    assert sealed[ORIGIN]["path"] == str((tmp_path / "test_development_machine.json").resolve())
