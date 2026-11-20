"""[U-HOST3] m-1 as a class: the hand-made cases the per-condition mutant harness asked for (FU/uhost_pricing3,
gates/cond_run1). Each case names the condition it pins (the harness's mutant ids); each is red on its mutant and green
on the rule. The admission rule's clauses: what a refusal states (the rows named, the plane and the link ends found and
stated, a description's reason only where it states one), how a run's host is judged over a physics sealed under an
earlier digest, a held file that is no description, a description's mode without rows of its own, a derivation whose
first term is another row in another unit, and the block writer's refusals. No description, mode, host or chip is
named here.
"""

from __future__ import annotations

import copy
import json

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics import available_profiles
from mimarsinan.deployment_record.platform_physics.constants import keys_in_group
from mimarsinan.deployment_record.platform_physics.hosts import admission as A
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.platform_physics.hosts import registry as REG
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HOST_IDENTITY_FIELD, HostRowAdmissionError, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import LINK_ENDS_FIELD
from mimarsinan.deployment_record.platform_physics.loader import apply_overrides
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

from unit.deployment_record.held_hosts import correct, digest, hold, shipped_copy  # noqa: F401  (a fixture)
from unit.deployment_record.registered_hosts import declared_rows, registered, registered_copy

NAME = host_descriptions()[0]


def _chip():
    for name in available_profiles():
        try:
            resolve_platform_physics(name, merge_override_blocks({"host": host_description_block(NAME)}))
        except ToolchainRefusal:
            continue
        return name
    raise AssertionError("no registry profile takes the identified host block")


CHIP = _chip()


def _block(name=NAME):
    return copy.deepcopy(host_description_block(name))


def _resolve(rows, chip=CHIP):
    return resolve_platform_physics(chip, merge_override_blocks({"host": rows}))


def _refused(call, kind=ToolchainRefusal):
    with pytest.raises(kind) as why:
        call()
    return str(why.value)


def _bare_rows(name=NAME):
    """The block's rows as a test description declares them (no host, plane or ends fields)."""
    return {k: {f: v for f, v in row.items() if f not in (HOST_IDENTITY_FIELD, "energy_plane", LINK_ENDS_FIELD)}
            for k, row in _block(name).items()}


class TestTheReadDescriptionsARefusalLists:
    def test_a_registered_description_whose_file_is_gone_is_listed_unreadable(self, tmp_path):
        """c002: read_hosts let the registry's refusal escape the refusal it was writing."""
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_gone_host") as name:
            REG.description_path(name).unlink()
            row = {**_block()["p_host"], HOST_IDENTITY_FIELD: "test_no_such_host default"}
            why = _refused(lambda: _resolve({"p_host": row}), HostRowAdmissionError)
            assert "which no read host description declares" in why and "unreadable" in why and name in why

    def test_a_shipped_description_the_judge_refuses_is_listed_unreadable(self, shipped_copy):
        """c001: read_hosts let the judge's refusal escape the refusal it was writing."""
        shipped, _held = shipped_copy
        data = json.loads((shipped / f"{NAME}.json").read_text())
        data.update(name="test_broken_host", format_version=2)
        (shipped / "test_broken_host.json").write_text(json.dumps(data))
        row = {**_block()["p_host"], HOST_IDENTITY_FIELD: "test_no_such_host default"}
        why = _refused(lambda: _resolve({"p_host": row}), HostRowAdmissionError)
        assert "which no read host description declares" in why and "unreadable" in why


class TestTheRulesRefusalNamesEveryRow:
    def test_a_sealed_origin_not_held_refuses_every_row_by_name(self):
        """c011: the held store's refusal escaped as the first row's, never naming the rows."""
        payload = _resolve(_block()).to_dict()
        payload["host_origin"]["sha256"] = "0" * 64
        why = _refused(lambda: PlatformPhysics.from_dict(payload), HostRowAdmissionError)
        assert all(key in why for key in _block()) and "not held" in why

    def test_a_registered_file_changed_refuses_every_declared_row_by_name(self, tmp_path):
        """c012: the registry's refusal escaped as the first row's, never naming the rows."""
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_changed_rows") as name:
            block = _block(name)
            path = REG.description_path(name)
            data = json.loads(path.read_text())
            data["display_name"] += " (changed)"
            path.write_text(json.dumps(data))
            why = _refused(lambda: _resolve(block), HostRowAdmissionError)
            assert all(key in why for key in block) and "changed after registration" in why


class TestAHeldFileThatIsNoDescription:
    def _sealed_on(self, held, data: bytes):
        payload = _resolve(_block()).to_dict()
        payload["host_origin"]["sha256"] = hold(held, data)
        return payload

    def test_a_held_file_the_judge_refuses_refuses_by_name(self, shipped_copy):
        """c021: the judge's refusal escaped the rule."""
        _shipped, held = shipped_copy
        data = json.loads(REG.description_path(NAME).read_text())
        data["format_version"] = 2
        payload = self._sealed_on(held, json.dumps(data).encode())
        assert "format_version" in _refused(lambda: PlatformPhysics.from_dict(payload), HostRowAdmissionError)

    def test_a_held_file_that_is_no_json_refuses_by_name(self, shipped_copy):
        """c022: a JSON error escaped the rule (no refusal by name)."""
        _shipped, held = shipped_copy
        payload = self._sealed_on(held, b"not a description")
        _refused(lambda: PlatformPhysics.from_dict(payload), HostRowAdmissionError)


class TestADescriptionsModeWithoutRowsOfItsOwn:
    def test_a_mode_declaring_no_rows_reads_the_shared_rows(self, tmp_path):
        """c029: a mode stating no 'constants' broke the rows' merge."""
        rows = _bare_rows()
        with registered(tmp_path, "test_shared_rows", rows) as identity:
            path = REG.description_path("test_shared_rows")
            data = json.loads(path.read_text())
            mode = data["default_mode"]
            data["constants"] = data["modes"][mode].pop("constants")
            path.write_text(json.dumps(data, indent=1))
            A.unregister_host_description("test_shared_rows")
            A.register_host_description(path)
            physics = _resolve(declared_rows("test_shared_rows", rows))
            assert all(physics.constants[k].host == identity for k in rows)


class TestWhatARefusalStates:
    def test_a_row_the_mode_does_not_declare_states_no_reason_it_has_none_for(self, tmp_path):
        """c039: the refusal claimed the description states a reason where it states none."""
        rows = _bare_rows()
        missing = sorted(rows)[0]
        kept = {k: v for k, v in rows.items() if k != missing}
        with registered(tmp_path, "test_no_reason", kept):
            row = {**rows[missing], HOST_IDENTITY_FIELD: "test_no_reason test"}
            why = _refused(lambda: _resolve({**declared_rows("test_no_reason", kept), missing: row}))
            assert missing in why and "(the description:" not in why

    @pytest.mark.parametrize("stated", ["system", "attributable"])
    def test_a_plane_refusal_states_the_plane_found(self, stated):
        """c099, c100: the words of a plane refusal."""
        block = _block()
        key = next(k for k, r in sorted(block.items()) if r.get("energy_plane") and r["energy_plane"] != stated)
        block[key]["energy_plane"] = stated
        why = _refused(lambda: _resolve(block))
        found = "system" if stated == "system" else "attributable or none"
        assert f"its energy_plane is {found}, the description states" in why

    def test_a_link_ends_refusal_states_both_ends(self):
        """c101, c103, c104, c105: the words of a link-ends refusal (the physics' ends and the description's)."""
        block = _block()
        key = next(k for k, r in sorted(block.items()) if r.get(LINK_ENDS_FIELD))
        stated = dict(block[key][LINK_ENDS_FIELD])
        other = {c: ["host", "chip"] for c in stated}
        block[key][LINK_ENDS_FIELD] = other
        why = _refused(lambda: _resolve(block))
        assert f"its {LINK_ENDS_FIELD} {{" in why and "are not the description's {" in why

    def test_a_link_ends_refusal_where_the_description_states_none(self, tmp_path):
        """c104: the description's ends read for a row it states none for."""
        rows = _bare_rows()
        key = next(k for k, r in sorted(_block().items()) if r.get(LINK_ENDS_FIELD))
        with registered(tmp_path, "test_no_ends", rows):
            declared = declared_rows("test_no_ends", rows)
            declared[key][LINK_ENDS_FIELD] = dict(_block()[key][LINK_ENDS_FIELD])
            why = _refused(lambda: _resolve(declared))
            assert key in why and "are not the description's None" in why

    def test_a_refusal_beside_no_host_names_no_host(self):
        """c140: the refusal of a host row naming no host, on a physics naming none, claimed rows beside a host."""
        row = {k: v for k, v in _block()["p_host"].items() if k not in (HOST_IDENTITY_FIELD, "energy_plane")}
        why = _refused(lambda: _resolve({"p_host": row}))
        assert "names no host" in why and "beside the rows of host" not in why


class TestARunsHostOverAnEarlierSealedPhysics:
    def test_a_host_block_declared_over_an_earlier_physics_is_judged_now_and_seals_the_new_origin(self, shipped_copy):
        """c112, c121: the declared rows were judged again against the earlier physics' origin."""
        shipped, held = shipped_copy
        earlier = PlatformPhysics.from_dict(_resolve(_block()).to_dict())
        correct(shipped, held)
        later = apply_overrides(earlier, _block())
        assert later.host_origin is not None and later.host_origin.sha256 == digest(shipped / f"{NAME}.json")
        assert later.host_origin.sha256 != earlier.host_origin.sha256


class TestTheBlockWriter:
    def test_a_term_the_mode_does_not_declare_refuses_by_name(self):
        """c144: the block writer reached for an undeclared row's band."""
        undeclared = next(k for k in keys_in_group(HOST) if k not in _block())
        why = _refused(lambda: A.host_derivation_block(NAME, "default", {undeclared: 1.0}), HostRowAdmissionError)
        assert undeclared in why

    def test_a_shipped_description_not_held_refuses_the_block_writer_by_name(self, shipped_copy):
        """c141: the writer read a description it could not judge."""
        shipped, held = shipped_copy
        correct(shipped, held, keep_held=False)
        key = sorted(_block())[0]
        why = _refused(lambda: A.host_derivation_block(NAME, "default", {key: 2.0}), HostRowAdmissionError)
        assert "not held" in why

    def test_a_chip_row_derived_from_a_host_row_takes_its_terms_unit(self):
        """c149, c150: the writer read the derived row's own unit where the description declares no such row (a chip
        row fed over the host's link, the attach row's form)."""
        link = next(k for k, r in sorted(_block().items()) if r.get(LINK_ENDS_FIELD) is None and k.startswith("t_link"))
        rows = A.host_derivation_block(NAME, "default", {"t_program_per_byte": {"of": {link: 1.0}}})
        assert rows["t_program_per_byte"]["unit"] == _block()[link]["unit"]
        assert rows["t_program_per_byte"]["nominal"] == pytest.approx(_block()[link]["nominal"])

    def test_a_derivation_whose_first_term_is_another_row_keeps_the_rows_unit(self):
        """c052: the derived row took its first term's unit."""
        block = _block()
        times = sorted(k for k in block if k.startswith("t_"))
        key = next(k for k in times if any(o < k and block[o]["unit"] != block[k]["unit"] for o in times))
        other = next(o for o in times if o < key and block[o]["unit"] != block[key]["unit"])
        rows = A.host_derivation_block(NAME, "default", {key: {"of": {key: 1.0, other: 1.0}}})
        assert rows[key]["unit"] == block[key]["unit"]
        assert _resolve({**block, **rows}).constants[key].unit == block[key]["unit"]


class TestTheBlockWritersCitation:
    def test_the_citations_of_the_terms_that_state_one_each_once(self, tmp_path):
        """c160, c161: the derived row's citation, the terms' own citations joined (a term without one adds none)."""
        rows = _bare_rows()
        times = sorted(k for k in rows if k.startswith("t_") and rows[k]["evidence_kind"] == "derived")
        key, other = times[0], times[1]
        rows[other]["citation"] = ""
        with registered(tmp_path, "test_citations", rows):
            block = A.host_derivation_block("test_citations", "default", {key: {"of": {key: 1.0, other: 1.0}}})
            assert block[key]["citation"] == rows[key]["citation"]
            assert A.host_derivation_block("test_citations", "default", {other: 2.0})[other]["citation"] == ""


class TestTheDeclarationsForm:
    def _row(self, declaration, key=None):
        key = key or sorted(_block())[0]
        return key, {**_block()[key], "evidence_kind": "derived", "derivation": "a stated scaling (test)",
                     "derived_from": declaration}

    @pytest.mark.parametrize("factor", ["2", float("inf"), None, [2.0]], ids=["string", "infinite", "none", "list"])
    def test_a_factor_outside_finite_positive_numbers_refuses_by_name(self, factor):
        """c163 (a string read as its number), c165 (an infinite factor)."""
        key, row = self._row({"of": {sorted(_block())[0]: factor}})
        assert "factor" in _refused(lambda: _resolve({**_block(), key: row}))

    def test_an_integer_factor_is_a_factor(self):
        """c167: an int factor was refused."""
        key = sorted(_block())[0]
        rows = A.host_derivation_block(NAME, "default", {key: 2})
        assert _resolve({**_block(), **rows}).constants[key].derived_from

    @pytest.mark.parametrize("declaration", [["of"], {"combine": "sum"}, {"of": ["x"]}],
                             ids=["a_list", "no_of", "of_a_list"])
    def test_a_declaration_outside_the_one_form_refuses_by_name(self, declaration):
        """c172, c174, c175: a list declaration, a declaration without 'of', and 'of' as a list crashed."""
        key, row = self._row(declaration)
        assert "derived_from" in _refused(lambda: _resolve({**_block(), key: row}))

    def test_an_empty_of_on_a_chip_row_naming_a_host_refuses_by_name(self):
        """c176: an empty 'of' on a row that need not name itself crashed."""
        chip = next(p for p in available_profiles()
                    if resolve_platform_physics(p, None).backing_store is not None)
        from mimarsinan.deployment_record.platform_physics.hosts import store as ST
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_attach_block
        (key, attach), = host_attach_block(NAME).items()
        attach = {**attach, "derived_from": {"of": {}}}
        blocks = {"host": _block(), "host_attach": {key: attach}, "store": ST.store_block(ST.HOST, "a variant (test)")}
        assert "at least one row" in _refused(lambda: resolve_platform_physics(chip, merge_override_blocks(blocks)))


class TestTheWordsAgainstTheDeclaration:
    def _derived(self, text, factor=2.0):
        key = sorted(_block())[0]
        return key, {**A.host_derivation_block(NAME, "default", {key: factor})[key], "derivation": text}

    def test_a_foreign_row_is_named_as_such_and_no_factor_claimed(self):
        """c215, c216: the refusal's words for a text naming a row that is no term."""
        other = sorted(_block())[1]
        key, row = self._derived(f"the description's {other} (test)")
        why = _refused(lambda: _resolve({**_block(), key: row}))
        assert f"it names ['{other}'], no term of it" in why and "it states the factors" not in why

    def test_a_foreign_factor_is_named_as_such_and_no_row_claimed(self):
        """c217, c218: the refusal's words for a text stating a factor no term declares."""
        key, row = self._derived("the row x 3 (test)")
        why = _refused(lambda: _resolve({**_block(), key: row}))
        assert "it states the factors ['3'], no term's" in why and "it names" not in why

    def test_a_factor_stated_to_more_digits_than_it_differs_in_is_the_factor(self):
        """c204: a text stating the declared factor exactly (within the comparison's tolerance) to more digits than
        rounding the factor to them reproduces."""
        factor = 0.1000000000000004
        key, row = self._derived("the row x 0.1000000000000005 (test)", factor)
        assert _resolve({**_block(), key: row}).constants[key].derived_from


class TestThePrintedDeclarationOfAMalformedText:
    @pytest.mark.parametrize("text", ["not a declaration", '{"combine": "sum"}', "[1, 2]", '{"of": {"x": null}}'],
                             ids=["no_json", "no_of", "a_list", "a_null_factor"])
    def test_a_text_outside_the_form_is_printed_as_it_stands(self, text):
        """c225, c226, c227: a malformed sealed declaration's basis raised instead of printing it (the rule refuses
        the row; its basis may be printed before)."""
        from mimarsinan.deployment_record.platform_physics.hosts.rows import declaration_statement
        assert declaration_statement(text) == f"derived_from: {text}; "

    def test_no_declaration_prints_nothing(self):
        from mimarsinan.deployment_record.platform_physics.hosts.rows import declaration_statement
        assert declaration_statement("") == ""


def _write(tmp_path, name, data):
    (tmp_path / f"{name}.md").write_text(f"# {name}\n")
    data = {**data, "name": name, "description_file": f"{name}.md"}
    (tmp_path / f"{name}.json").write_text(json.dumps(data))
    return tmp_path / f"{name}.json"


def _description(name="test_judged"):
    from unit.deployment_record.registered_hosts import description_data
    return description_data(name, _bare_rows())


class TestTheDescriptionsJudge:
    def test_a_chip_row_in_a_description_refuses_at_registration(self, tmp_path):
        """c229: a non-host row was admitted into a description."""
        data = _description()
        mode = data["default_mode"]
        physics = resolve_platform_physics(CHIP, None)
        chip_row = next(k for k in sorted(physics.constants) if A.spec_for(k).group != HOST)
        data["modes"][mode]["constants"][chip_row] = {f: v for f, v in physics.constants[chip_row].to_dict().items()
                                                      if f not in ("key", "overridden")}
        assert "not a host row" in _refused(lambda: A.register_host_description(_write(tmp_path, "test_judged", data)))

    def test_a_datasheet_row_is_admitted(self, tmp_path):
        """c236: the class 'datasheet' was refused."""
        data = _description()
        mode = data["default_mode"]
        key = sorted(data["modes"][mode]["constants"])[0]
        data["modes"][mode]["constants"][key].update(evidence_kind="datasheet", citation="a datasheet (test)")
        name = A.register_host_description(_write(tmp_path, "test_datasheet", data))
        A.unregister_host_description(name)

    def test_a_shared_measured_row_refuses_at_registration(self, tmp_path):
        """c261: the rows every mode shares were not judged."""
        data = _description()
        mode = data["default_mode"]
        key = sorted(data["modes"][mode]["constants"])[0]
        data["constants"] = {key: {**data["modes"][mode]["constants"].pop(key), "evidence_kind": "measured",
                                   "note": "measured on a machine (test)"}}
        assert "measured" in _refused(lambda: A.register_host_description(_write(tmp_path, "test_shared_measured",
                                                                                  data)))

    def test_a_default_mode_that_is_no_mode_refuses_at_registration(self, tmp_path):
        """c244."""
        data = {**_description(), "default_mode": "test_no_such_mode"}
        assert "test_no_such_mode" in _refused(lambda: A.register_host_description(_write(tmp_path, "test_mode", data)))

    def test_a_name_field_unlike_its_file_refuses_at_registration(self, tmp_path):
        """c251."""
        path = _write(tmp_path, "test_named", _description())
        data = json.loads(path.read_text())
        data["name"] = "test_other_name"
        path.write_text(json.dumps(data))
        assert "test_other_name" in _refused(lambda: A.register_host_description(path))

    def test_link_ends_of_a_row_it_does_not_declare_refuse_at_registration(self, tmp_path):
        """c248."""
        data = _description()
        mode = data["default_mode"]
        link = next(k for k, r in sorted(_block().items()) if r.get(LINK_ENDS_FIELD))
        data["modes"][mode]["constants"].pop(link)
        data["energy_attribution"]["rows"].pop(link, None)
        data[LINK_ENDS_FIELD] = {"rows": {link: _block()[link][LINK_ENDS_FIELD]}, "basis": "a test (test)"}
        assert link in _refused(lambda: A.register_host_description(_write(tmp_path, "test_stray_ends", data)))

    def test_a_description_without_energy_rows_needs_no_attribution(self, tmp_path):
        """c256: a description stating no energy_attribution (and no energy row) broke the judge."""
        data = _description()
        mode = data["default_mode"]
        data["modes"][mode]["constants"] = {k: v for k, v in data["modes"][mode]["constants"].items()
                                            if not k.startswith(("p_", "e_"))}
        data.pop("energy_attribution")
        name = A.register_host_description(_write(tmp_path, "test_no_energy", data))
        A.unregister_host_description(name)

    def test_link_ends_stating_no_rows_read_as_none(self, tmp_path):
        data = {**_description(), LINK_ENDS_FIELD: {"basis": "a test (test)"}}
        name = A.register_host_description(_write(tmp_path, "test_ends_no_rows", data))
        A.unregister_host_description(name)

    def test_a_shipped_description_named_with_a_space_refuses_by_name(self, shipped_copy):
        """c242: the token rule on a shipped description (registration checks its own)."""
        shipped, _held = shipped_copy
        data = json.loads((shipped / f"{NAME}.json").read_text())
        data.update(name="test host", description_file="test host.md")
        (shipped / "test host.md").write_text("# a test\n")
        (shipped / "test host.json").write_text(json.dumps(data))
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_description
        assert "one token" in _refused(lambda: host_description("test host"))


class TestTheAttachRowsFields:
    def test_it_carries_the_link_rows_citation_and_its_derivation_in_the_note(self):
        """c303 (the citation dropped), c310 (the link row's own arithmetic dropped from the note)."""
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_attach_block
        (_key, row), = host_attach_block(NAME).items()
        link = _block()["t_link_per_byte"]
        assert row["citation"] == link["citation"] and link["derivation"] in row["note"]

    def test_a_link_row_without_citation_or_derivation_gives_an_attach_row_without_them(self, tmp_path):
        """c308 (a field the link row lacks read anyway), c311 (an absent derivation printed as 'False')."""
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_attach_block
        rows = _bare_rows()
        rows["t_link_per_byte"] = {k: v for k, v in rows["t_link_per_byte"].items() if k not in ("citation",
                                                                                               "derivation")}
        rows["t_link_per_byte"].update(evidence_kind="estimated", note="a test's link (test)")
        with registered(tmp_path, "test_bare_link", rows):
            (_key, row), = host_attach_block("test_bare_link").items()
            assert "citation" not in row and "False" not in row["note"]


def _profiles():
    from mimarsinan.deployment_record.platform_physics.registry import available_profiles as names
    from pathlib import Path
    import mimarsinan.deployment_record.platform_physics as package
    directory = Path(package.__file__).parent / "profiles"
    return {name: json.loads((directory / f"{name}.json").read_text()) for name in names()}


class TestTheProfilesEntry:
    def test_a_profile_of_another_version_refuses(self):
        """c312."""
        from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict
        name, data = sorted(_profiles().items())[0]
        with pytest.raises(ValueError, match="format_version"):
            profile_from_dict({**data, "format_version": 2})

    def test_an_external_component_naming_a_constant_the_profile_declares_refuses(self):
        """c315, c325."""
        from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict
        name, data = next((n, d) for n, d in sorted(_profiles().items()) if d.get("external_components"))
        component = data["external_components"][0]
        key = sorted(component["constants"])[0]
        own = next(iter(data["constants"].values()))
        data = {**data, "constants": {**data["constants"], key: {**own}}}
        with pytest.raises(ValueError, match="external component"):
            profile_from_dict(data)

    def test_a_profiles_chip_components_converter_and_absent_constants_are_read(self):
        """c318 (a profile stating no constants), c319 (its chip components), c328 (its converter)."""
        from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict
        profiles = sorted(_profiles().items())
        _name, with_parts = next((n, d) for n, d in profiles if d.get("chip_components"))
        assert profile_from_dict(with_parts).chip_components
        _name, with_converter = next((n, d) for n, d in profiles if d.get("converter"))
        assert profile_from_dict(with_converter).converter is not None
        _name, plain = next((n, d) for n, d in profiles if not d.get("chip_components") and not d.get("carry_buffer"))
        assert profile_from_dict({k: v for k, v in plain.items() if k != "constants"}).constants == {}


class TestTheRunsDeclaredRowsReplaceTheirRowWhole:
    def _identified(self):
        return _resolve(_block())

    def test_a_declared_plane_replaces_the_physics_plane(self):
        """c362: the earlier physics' system plane survived a declaration stating another."""
        physics = self._identified()
        key = next(k for k in sorted(physics.system_energy_rows))
        row = {**_block()[key], "energy_plane": "attributable"}
        assert key in _refused(lambda: apply_overrides(physics, {key: row}))

    def test_a_declared_link_row_without_ends_does_not_keep_the_physics_ends(self):
        """c364: the earlier physics' link ends survived a declaration stating none."""
        physics = self._identified()
        key = next(iter(sorted(physics.link_ends)))
        row = {k: v for k, v in _block()[key].items() if k != LINK_ENDS_FIELD}
        assert key in _refused(lambda: apply_overrides(physics, {key: row}))

    def test_a_host_replacing_the_earlier_one_drops_its_link_ends(self, tmp_path):
        """c363: the replaced host's link ends survived beside the new host's rows."""
        physics = self._identified()
        link = next(iter(sorted(physics.link_ends)))
        rows = {k: v for k, v in _bare_rows().items() if k != link}
        with registered(tmp_path, "test_no_link", rows):
            later = apply_overrides(physics, declared_rows("test_no_link", rows))
            assert link not in later.constants and link not in later.link_ends

    def test_a_declared_host_rows_note_names_its_host_and_a_chip_rows_does_not(self):
        """c407 (the host's name dropped from its row's note), c388 (a row naming no host counted as one)."""
        physics = self._identified()
        key = sorted(_block())[0]
        assert physics.constants[key].note.startswith(f"[host {_block()[key][HOST_IDENTITY_FIELD]}] ")
        chip_key = next(k for k in sorted(physics.constants) if A.spec_for(k).group != HOST)
        own = physics.constants[chip_key]
        later = apply_overrides(physics, {chip_key: {"nominal": own.nominal, "low": own.low, "high": own.high,
                                                     "unit": own.unit, "evidence_kind": "estimated",
                                                     "note": "a test's value (test)"}})
        assert later.constants[chip_key].note == "a test's value (test)"

    def test_an_override_without_a_unit_takes_the_existing_rows_unit(self):
        """c384: the vocabulary's display unit replaced the existing row's."""
        from mimarsinan.deployment_record.platform_physics import get_platform_physics
        found = next((p, k) for p in available_profiles() for k, v in sorted(get_platform_physics(p).constants.items())
                     if v.unit != A.spec_for(k).display_unit)
        profile, key = found
        own = get_platform_physics(profile).constants[key]
        later = apply_overrides(get_platform_physics(profile), {key: {"nominal": own.nominal,
                                                                      "note": "a test's value (test)"}})
        assert later.constants[key].unit == own.unit


class TestTheSealedPhysicsStatesEachOptionalPartIffItHasIt:
    FIELDS = ("external_components", "converter", "carry_buffer", "chip_components", "system_energy_rows",
              "backing_store", "host_origin")

    def test_each_profile_and_the_identified_host_seal_their_parts_and_read_back(self):
        """c416 to c423: an optional part of the sealed physics dropped, or stated empty where absent."""
        from mimarsinan.deployment_record.platform_physics import get_platform_physics
        physics_set = [get_platform_physics(p) for p in available_profiles()] + [_resolve(_block())]
        seen = set()
        for physics in physics_set:
            payload = physics.to_dict()
            for field in self.FIELDS:
                value = getattr(physics, field)
                assert (field in payload) == (value is not None and value != () and value != {}), (physics.name, field)
                seen |= {field} if field in payload else set()
            assert PlatformPhysics.from_dict(payload).to_dict() == payload
        assert seen == set(self.FIELDS)


class TestASealedPhysicsWithoutConstants:
    def test_it_reads_back(self):
        """c455: a sealed physics stating no constant broke the read."""
        from dataclasses import replace as _replace
        bare = _replace(resolve_platform_physics(CHIP, None), constants={}, carry_buffer=None, converter=None,
                        chip_components=(), backing_store=None, transfer_rows={}, system_energy_rows=())
        payload = bare.to_dict()
        assert PlatformPhysics.from_dict(payload).to_dict() == payload


class TestThePrintedBasis:
    def _value(self, **fields):
        from mimarsinan.deployment_record.platform_physics.constants import PhysicsConstantValue
        key = sorted(_block())[0]
        base = dict(key=key, low=1.0, nominal=1.0, high=1.0, unit=_block()[key]["unit"], evidence_kind="derived",
                    derivation="a stated derivation (test)")
        return PhysicsConstantValue(**{**base, **fields})

    def test_a_row_without_a_citation_prints_its_derivation(self):
        """c457."""
        assert self._value().basis == "derived: a stated derivation (test)"

    def test_an_operators_row_says_so_and_a_profiles_does_not(self):
        """c459, c460."""
        assert self._value(overridden=True).basis.startswith("derived (operator override): ")
        assert "(operator override)" not in self._value().basis


class TestTheSealedOriginsForm:
    @pytest.mark.parametrize("origin", [["kind", "name", "path", "sha256"],
                                        {"kind": "shipped", "name": NAME, "path": 7, "sha256": "0" * 64},
                                        {"kind": "shipped", "name": 7, "path": "p", "sha256": "0" * 64}],
                             ids=["a_list_of_its_fields", "a_path_no_string", "a_name_no_string"])
    def test_it_refuses_by_name(self, origin):
        """c471, c476, c491."""
        from mimarsinan.deployment_record.platform_physics.hosts.registry import HostOrigin, HostOriginError
        with pytest.raises(HostOriginError, match="host origin"):
            HostOrigin.from_dict(origin)


class TestTheRegistrysReads:
    def test_a_listed_file_gone_before_its_read_is_no_description(self, monkeypatch, tmp_path):
        """c513: a listed shipped file removed before its read raised FileNotFoundError."""
        monkeypatch.setattr(REG, "description_path", lambda name: tmp_path / f"{name}.json")
        assert REG.read_description("test_gone_since_listed") is None

    def test_a_held_file_that_does_not_hash_to_its_name_refuses(self, shipped_copy):
        """c520: a held file's bytes were trusted under a name that is not their digest."""
        _shipped, held = shipped_copy
        source = REG.description_path(NAME)
        wrong = "f" * 64
        (held / f"{wrong}.json").write_bytes(source.read_bytes())
        payload = _resolve(_block()).to_dict()
        payload["host_origin"]["sha256"] = wrong
        assert "not its name" in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_a_registered_name_holds_only_the_digest_it_was_registered_with(self, tmp_path):
        """c526: a registered origin of another digest read the description registered under its name now."""
        with registered_copy(tmp_path / "first", NAME, lambda data: None, as_name="test_digest_host") as name:
            payload = _resolve(_block(name)).to_dict()

        def edit(data):
            data["display_name"] += " (registered again, other bytes)"

        with registered_copy(tmp_path / "second", NAME, edit, as_name="test_digest_host"):
            assert "not held" in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_a_description_file_that_is_no_json_file_refuses_at_registration(self, tmp_path):
        """c533."""
        source = REG.description_path(NAME)
        data = json.loads(source.read_text())
        data.update(name="test_suffix", description_file="test_suffix.md")
        (tmp_path / "test_suffix.md").write_text("# a test\n")
        path = tmp_path / "test_suffix.txt"
        path.write_text(json.dumps(data))
        assert "<name>.json" in _refused(lambda: A.register_host_description(path))


class TestTheWallsForm:
    def test_an_integer_wall_is_a_wall(self):
        """c557: an int wall (a whole number of seconds) was refused."""
        from mimarsinan.chip_simulation.hybrid_run.stage_timing import host_wall
        wall = host_wall([{"name": "op", "wall_s_total": 1, "invocations": 1, "samples_per_call": 2}])
        assert wall.per_inference_s == 0.5 and wall.raw_total_s == 1.0

    def test_a_wall_is_timed_iff_some_row_was(self):
        """c565."""
        from mimarsinan.chip_simulation.hybrid_run.stage_timing import host_wall
        assert host_wall([{"name": "op", "wall_s_total": 0.5, "invocations": 1, "samples_per_call": 1}]).timed
        assert not host_wall([{"name": "op", "wall_s_total": None}]).timed


class TestTheFoldsOtherClauses:
    def _two_op_schedule(self, same_name=False):
        from dataclasses import replace as _replace
        from mimarsinan.deployment_record.schema import ComputeOpRecord
        from unit.deployment_record.record_fixtures import make_full_record
        record = make_full_record()
        first = next(s for s in record.schedule.stages if isinstance(s, ComputeOpRecord))
        last = max(s.stage_index for s in record.schedule.stages)
        second = _replace(first, name=first.name if same_name else first.name + "_second", stage_index=last + 1)
        return _replace(record.schedule, stages=record.schedule.stages + (second,)), first.name

    def _row(self, name):
        return {"stage_index": 0, "name": name, "wall_s_total": 0.5, "invocations": 1, "samples_per_call": 1}

    def test_a_wall_naming_no_compute_op_refuses(self):
        """c591."""
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import fold_compute_walls
        schedule, _name = self._two_op_schedule()
        with pytest.raises(ValueError, match="match no ComputeOp"):
            fold_compute_walls(schedule, [self._row("test_no_such_op")])

    def test_a_wall_whose_name_two_compute_ops_carry_refuses(self):
        """c593, c604."""
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import fold_compute_walls
        schedule, name = self._two_op_schedule(same_name=True)
        with pytest.raises(ValueError, match="multiple ComputeOp"):
            fold_compute_walls(schedule, [self._row(name)])

    def test_an_untimed_compute_op_stays_untimed_beside_a_timed_one(self):
        """c608: the fold read a wall for every ComputeOp, timed or not."""
        from mimarsinan.deployment_record.schema import ComputeOpRecord
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import fold_compute_walls
        schedule, name = self._two_op_schedule()
        folded = fold_compute_walls(schedule, [self._row(name)])
        ops = {s.name: s for s in folded.stages if isinstance(s, ComputeOpRecord)}
        assert ops[name].wall_s_total == 0.5 and ops[name + "_second"].wall_s_total is None
        assert ops[name + "_second"].invocations is None and ops[name + "_second"].samples_per_call is None


class TestTheRecordsOtherReads:
    def test_a_version_1_record_whose_latency_is_no_mapping_refuses_by_name(self):
        """c624: a latency stated as a string broke the rule's field check."""
        import copy as _copy
        from mimarsinan.deployment_record.schema import record as R
        from unit.deployment_record.record_fixtures import make_full_record
        old = _copy.deepcopy(make_full_record().to_dict())
        old["format_version"] = 1
        old["identity"]["format_version"] = 1
        old["timing"]["latency"] = "host_ops_s host_ops_s_per_pass"
        with pytest.raises(ToolchainRefusal, match="timing.latency.host_ops_s"):
            R.DeploymentRecord.from_dict(old)

    def test_its_optional_parts_are_read_into_their_types(self):
        """c635, c637, c639: the output stage, the declared departures and the device samples kept as sealed."""
        from mimarsinan.deployment_record.schema import record as R
        from unit.deployment_record.record_fixtures import make_full_record
        payload = make_full_record().to_dict()
        payload.update(output_stage={"kind": "a test stage"}, declared_departures=[{"what": "a test departure"}],
                       device_samples=[{"sample": 0}])
        record = R.DeploymentRecord.from_dict(payload)
        assert isinstance(record.declared_departures, tuple) and isinstance(record.device_samples, tuple)
        assert record.output_stage == payload["output_stage"] and record.output_stage is not payload["output_stage"]
        assert record.to_dict() == payload
